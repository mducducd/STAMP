import logging
from collections.abc import Collection, Iterable
from pathlib import Path
from typing import cast, no_type_check

import h5py
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import openslide
import torch
from jaxtyping import Float, Integer
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from packaging.version import Version
from PIL import Image
from torch import Tensor
from torch.func import jacrev  # pyright: ignore[reportPrivateImportUsage]

from stamp.modeling.data import get_coords, get_stride
from stamp.modeling.lightning_model import LitVisionTransformer
from stamp.modeling.vision_transformer import VisionTransformer
from stamp.preprocessing import supported_extensions
from stamp.preprocessing.tiling import get_slide_mpp_
from stamp.types import DeviceLikeType, Microns, SlideMPP, TilePixels

_logger = logging.getLogger("stamp")


def sinh_arcsinh_transform(
    values: torch.Tensor,
    epsilon: float = 0.0,  # skewness
    delta: float = 1.0,  # tail weight (>0)
) -> torch.Tensor:
    """
    Sinh–Arcsinh transformation (Jones & Pewsey, 2009).

    Args:
        values: Input tensor.
        epsilon: Skewness parameter.
        delta: Tail weight parameter (>0).

    Returns:
        Transformed tensor (same shape as input).
    """
    return torch.sinh(delta * torch.arcsinh(values) - epsilon)


def _normalize_gradcam_log(
    values: torch.Tensor,
    clamp_percent: float | None = None,  # set None to disable clamping
    remap: bool = False,  # if True → min–max after transform
    mode: str = "log",  # "log" | "symlog"
    scale: float = 1.0,  # only used for symlog
) -> torch.Tensor:
    """
    Global log scaling for Grad-CAM across all tiles+categories.

    Args:
        values: tensor [tile, category]
        clamp_percent: optional percentile clamp (e.g. 1.0 → 1–99),
                       or None for no clamping
        remap: if True, min–max scale to [0,1]; if False, return transformed values
        mode: "log" (sign*log1p) or "symlog" (normalized symmetric log in [-1,1])
        scale: controls compression in symlog (higher = more compression)

    Returns:
        tensor with same shape as `values`
    """
    v = values.detach().cpu().numpy().reshape(-1)

    # 1. transform
    if mode == "log":
        v = np.sign(v) * np.log1p(np.abs(v))
    elif mode == "symlog":
        v = np.sign(v) * np.log1p(scale * np.abs(v)) / np.log1p(scale)
    else:
        raise ValueError(f"Unknown mode {mode}, must be 'log' or 'symlog'")

    # 2. optional clamp
    if clamp_percent is not None and clamp_percent > 0:
        lo = np.percentile(v, clamp_percent)
        hi = np.percentile(v, 100.0 - clamp_percent)
        v = np.clip(v, lo, hi)
    else:
        lo, hi = v.min(), v.max()

    # 3. optional remap to [0,1]
    if remap:
        v = (v - lo) / (hi - lo + 1e-8)

    v = v.reshape(values.shape)
    return torch.from_numpy(v).to(values.device, dtype=values.dtype)


def log_transform(values: torch.Tensor) -> torch.Tensor:
    """sign(x) * log1p(|x|)"""
    return torch.sign(values) * torch.log1p(values.abs())


def boxcox_transform(values: torch.Tensor, lam: float = 0.0) -> torch.Tensor:
    """
    Box–Cox transform (requires values > 0).
    """
    if lam == 0:
        return torch.log(values)
    else:
        return (values.pow(lam) - 1) / lam


def symmetric_boxcox_transform(values: torch.Tensor, lam: float = 0.0) -> torch.Tensor:
    """
    Symmetric Box–Cox transform using sign(|x|).
    """
    abs_vals = values.abs() + 1e-8  # avoid 0 issues
    if lam == 0:
        out = torch.log(abs_vals)
    else:
        out = (abs_vals.pow(lam) - 1) / lam
    return torch.sign(values) * out


def yeojohnson_transform(values: torch.Tensor, lam: float = 0.0) -> torch.Tensor:
    """
    Yeo–Johnson transform (works for both positive and negative values).
    """
    pos = values >= 0
    out = torch.zeros_like(values)

    if lam == 0:
        out[pos] = torch.log1p(values[pos])
    else:
        out[pos] = ((values[pos] + 1).pow(lam) - 1) / lam

    if lam == 2:
        out[~pos] = -torch.log1p(-values[~pos])
    else:
        out[~pos] = -(((-values[~pos] + 1).pow(2 - lam) - 1) / (2 - lam))

    return out


def mulaw_transform(values: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Mu-law companding: sign(x) * log1p(scale*|x|) / log1p(scale)"""
    return (
        torch.sign(values)
        * torch.log1p(scale * values.abs())
        / torch.log1p(torch.tensor(scale, dtype=values.dtype, device=values.device))
    )


def sqrt_transform(values: torch.Tensor) -> torch.Tensor:
    """
    Symmetric square-root transform.
    """
    return torch.sign(values) * torch.sqrt(values.abs())


def webber_transform(values: torch.Tensor, C: float = -2.0) -> torch.Tensor:
    """Webber 2012 symlog: sign(x) * log10(1 + |x| / 10^C)"""
    return torch.sign(values) * torch.log10(1 + values.abs() / (10.0**C))


def arcsinh_transform(values: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Arcsinh scaling: arcsinh(alpha * x)"""
    return torch.arcsinh(alpha * values)


def shifted_log_transform(values: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    Shifted symmetric log transform.

    Args:
        values: Input tensor.
        alpha: Shift parameter (>0).

    Returns:
        Transformed tensor (same shape).
    """
    return torch.sign(values) * torch.log1p(alpha * values.abs())


def normalize_gradcam(
    values: torch.Tensor,
    mode: str = "log",  # "log" | "mulaw" | "webber" | "arcsinh"
    clamp_percent: float | None = None,
    remap: bool = False,
    scale: float = 10.0,  # for "mulaw"
    C: float = -2.0,  # for "webber"
    alpha: float = 1.0,  # for "arcsinh"
) -> torch.Tensor:
    """
    Normalize Grad-CAM values with different symmetric log-like transforms.
    """
    v = values.detach().cpu().numpy().reshape(-1)

    if mode == "log":
        v = np.sign(v) * np.log1p(np.abs(v))

    elif mode == "mulaw":
        v = np.sign(v) * np.log1p(scale * np.abs(v)) / np.log1p(scale)

    elif mode == "webber":
        v = np.sign(v) * np.log10(1 + np.abs(v) / (10.0**C))

    elif mode == "arcsinh":
        v = np.arcsinh(alpha * v)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # clamp
    if clamp_percent is not None and clamp_percent > 0:
        lo = np.percentile(v, clamp_percent)
        hi = np.percentile(v, 100.0 - clamp_percent)
        v = np.clip(v, lo, hi)
    else:
        lo, hi = v.min(), v.max()

    # optional remap
    if remap:
        v = (v - lo) / (hi - lo + 1e-8)

    v = v.reshape(values.shape)
    return torch.from_numpy(v).to(values.device, dtype=values.dtype)

def _gradcam_per_category(
    model: VisionTransformer,
    feats: Float[Tensor, "tile feat"],
    coords: Float[Tensor, "tile 2"],
) -> Float[Tensor, "tile category"]:
    feat = -1  # feats dimension

    return (
        (
            feats
            * jacrev(
                lambda bags: torch.softmax(
                    model.forward(
                        bags=bags.unsqueeze(0),
                        coords=coords.unsqueeze(0),
                        mask=None,
                    ),
                    dim=1,
                ).squeeze(0)
            )(feats)
        )
        .mean(feat)  # type: ignore
        .abs()
    ).permute(-1, -2)


def _vals_to_im(
    scores: Float[Tensor, "tile feat"],
    coords_norm: Integer[Tensor, "tile coord"],
) -> Float[Tensor, "width height category"]:
    """Arranges scores in a 2d grid according to coordinates"""
    size = coords_norm.max(0).values.flip(0) + 1
    im = torch.zeros((*size.tolist(), *scores.shape[1:])).type_as(scores)

    flattened_im = im.flatten(end_dim=1)
    flattened_coords = coords_norm[:, 1] * im.shape[1] + coords_norm[:, 0]
    flattened_im[flattened_coords] = scores

    im = flattened_im.reshape_as(im)

    return im


def _show_thumb(
    slide, thumb_ax: Axes, attention: Tensor, default_slide_mpp: SlideMPP | None
) -> np.ndarray:
    mpp = get_slide_mpp_(slide, default_mpp=default_slide_mpp)
    dims_um = np.array(slide.dimensions) * mpp
    thumb = slide.get_thumbnail(np.round(dims_um * 8 / 256).astype(int))
    thumb_ax.imshow(np.array(thumb)[: attention.shape[0] * 8, : attention.shape[1] * 8])
    return np.array(thumb)[: attention.shape[0] * 8, : attention.shape[1] * 8]


@no_type_check  # beartype<=0.19.0 breaks here for some reason
def _show_class_map(
    class_ax: Axes,
    top_score_indices: Integer[Tensor, "width height"],
    gradcam_2d: Float[Tensor, "width height category"],
    categories: Collection[str],
) -> tuple[np.ndarray, list[Patch]]:
    """Returns the class map image and legend patches for saving separately"""
    cmap = plt.get_cmap("Pastel1")
    classes = cast(np.ndarray, cmap(top_score_indices.cpu().numpy()))
    classes[..., -1] = (gradcam_2d.sum(-1) > 0).detach().cpu().numpy() * 1.0
    class_ax.imshow(classes)

    legend_patches = [
        Patch(facecolor=cmap(i), label=cat) for i, cat in enumerate(categories)
    ]
    class_ax.legend(handles=legend_patches)

    return classes, legend_patches


def _create_overlay(
    thumb: np.ndarray,
    score_im: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Creates an overlay of the heatmap over the thumbnail."""
    # Resize score_im to match thumbnail size
    thumb_height, thumb_width = thumb.shape[:2]
    score_resized = Image.fromarray(np.uint8(score_im * 255)).resize(
        (thumb_width, thumb_height), resample=Image.Resampling.NEAREST
    )
    score_resized = np.array(score_resized) / 255.0

    # Convert thumbnail to float for blending
    thumb_float = thumb.astype(float) / 255.0

    # Create overlay where heatmap alpha channel > 0
    mask = score_resized[..., -1] > 0
    overlay = thumb_float.copy()

    # Blend heatmap with thumbnail where mask is True
    overlay[mask] = alpha * score_resized[mask, :3] + (1 - alpha) * thumb_float[mask]

    return (overlay * 255).astype(np.uint8)


def _create_plotted_overlay(
    thumb: np.ndarray,
    score_im: np.ndarray,
    category: str,
    slide_score: float,
    alpha: float,
) -> tuple[Figure, Axes]:
    """Creates a plotted overlay with title and legend."""
    overlay = _create_overlay(thumb, score_im, alpha)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(overlay)
    ax.set_title(f"{category} - Slide Score: {slide_score:.3f}", fontsize=16, pad=20)
    ax.axis("off")

    # Create legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="red", alpha=0.7, label="Positive"),
        Patch(facecolor="blue", alpha=0.7, label="Negative"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.98, 0.98))

    plt.tight_layout()
    return fig, ax


def heatmaps_(
    *,
    feature_dir: Path,
    wsi_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    slide_paths: Iterable[Path] | None,
    device: DeviceLikeType,
    default_slide_mpp: SlideMPP | None,
    opacity: float,
    # top tiles
    topk: int,
    bottomk: int,
) -> None:
    # Collect slides to generate heatmaps for
    if slide_paths is not None:
        wsis_to_process = (wsi_dir / slide for slide in slide_paths)
    else:
        wsis_to_process = (
            p for ext in supported_extensions for p in wsi_dir.glob(f"**/*{ext}")
        )

    for wsi_path in wsis_to_process:
        h5_path = feature_dir / wsi_path.with_suffix(".h5").name

        if not h5_path.exists():
            _logger.info(f"could not find matching h5 file at {h5_path}. Skipping...")
            continue

        slide_output_dir = output_dir / h5_path.stem
        # Create organized folder structure
        plots_dir = slide_output_dir / "plots"
        raw_dir = slide_output_dir / "raw"
        tiles_dir = slide_output_dir / "tiles"

        for dir_path in [plots_dir, raw_dir, tiles_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)

        _logger.info(f"creating heatmaps for {wsi_path.name}")

        slide = openslide.open_slide(wsi_path)
        slide_mpp = get_slide_mpp_(slide, default_mpp=default_slide_mpp)
        assert slide_mpp is not None, "could not determine slide MPP"

        with h5py.File(h5_path) as h5:
            feat_type = h5.attrs.get("feat_type", None)
            if feat_type is not None and feat_type != "tile":
                raise ValueError(
                    f"Feature file {h5_path} is a slide or patient level feature. Heatmaps are currently supported for tile-level features only."
                )
            feats = (
                torch.tensor(
                    h5["feats"][:]  # pyright: ignore[reportIndexIssue]
                )
                .float()
                .to(device)
            )
            coords_info = get_coords(h5)
            coords_um = torch.from_numpy(coords_info.coords_um).float()
            stride_um = Microns(get_stride(coords_um))

            tile_size_slide_px = TilePixels(
                int(round(float(coords_info.tile_size_um) / slide_mpp))
            )

        # grid coordinates, i.e. the top-left most tile is (0, 0), the one to its right (0, 1) etc.
        coords_norm = (coords_um / stride_um).round().long()

        # coordinates as used by OpenSlide
        coords_tile_slide_px = torch.round(coords_um / slide_mpp).long()

        model = (
            LitVisionTransformer.load_from_checkpoint(checkpoint_path).to(device).eval()
        )

        # TODO: Update version when a newer model logic breaks heatmaps.
        # if Version(model.stamp_version) < Version("2.3.0"):
        #     raise ValueError(
        #         f"model has been built with stamp version {model.stamp_version} "
        #         f"which is incompatible with the current version."
        #     )

        # Score for the entire slide
        slide_score = (
            model.vision_transformer(
                bags=feats.unsqueeze(0),
                coords=coords_um.unsqueeze(0),
                mask=None,
            )
            .squeeze(0)
            .softmax(0)
        )

        # Find the class with highest probability
        highest_prob_class_idx = slide_score.argmax().item()

        gradcam = _gradcam_per_category(
            model=model.vision_transformer,
            feats=feats,
            coords=coords_um,
        )  # shape: [tile, category]
        gradcam = _normalize_gradcam_log(gradcam, clamp_percent=5.0)
        gradcam_2d = _vals_to_im(
            gradcam,
            coords_norm,
        ).detach()  # shape: [width, height, category]
        # gradcam = normalize_gradcam(gradcam, clamp_percent=0.1, mode="log")
        scores = torch.softmax(
            model.vision_transformer.forward(
                bags=feats.unsqueeze(-2),
                coords=coords_um.unsqueeze(-2),
                mask=torch.zeros(len(feats), 1, dtype=torch.bool, device=device),
            ),
            dim=1,
        )  # shape: [tile, category]

        scores_2d = _vals_to_im(
            scores, coords_norm
        ).detach()  # shape: [width, height, category]

        fig, axs = plt.subplots(
            nrows=2, ncols=max(2, len(model.categories)), figsize=(12, 8)
        )

        # Generate class map and save it separately
        classes_img, legend_patches = _show_class_map(
            class_ax=axs[0, 1],
            top_score_indices=scores_2d.topk(2).indices[:, :, 0],
            gradcam_2d=gradcam_2d,
            categories=model.categories,
        )

        # Save class map to raw folder
        target_size = np.array(classes_img.shape[:2][::-1]) * 8
        Image.fromarray(np.uint8(classes_img * 255)).resize(
            tuple(target_size), resample=Image.Resampling.NEAREST
        ).save(raw_dir / f"{h5_path.stem}-classmap.png")

        # Generate overview thumbnail first (moved up)
        thumb = _show_thumb(
            slide=slide,
            thumb_ax=axs[0, 0],
            attention=_vals_to_im(
                torch.zeros(len(feats), 1).to(device),  # placeholder for initial call
                coords_norm,
            ).squeeze(-1),
            default_slide_mpp=default_slide_mpp,
        )

        attention = None
        for ax, (pos_idx, category) in zip(axs[1, :], enumerate(model.categories)):
            ax: Axes
            top2 = scores.topk(2)
            # Calculate the distance of the "hot" class
            # to the class with the highest score apart from the hot class
            category_support = torch.where(
                top2.indices[..., 0] == pos_idx,
                scores[..., pos_idx] - top2.values[..., 1],
                scores[..., pos_idx] - top2.values[..., 0],
            )  # shape: [tile]
            assert ((category_support >= -1) & (category_support <= 1)).all()

            # So, if we have a pixel with scores (.4, .4, .2) and would want to get the heat value for the first class,
            # we would get a neutral color, because it is matched with the second class
            # But if our scores were (.4, .3, .3), it would be red,
            # because now our class is .1 above its nearest competitor

            attention = torch.where(
                top2.indices[..., 0] == pos_idx,
                gradcam[..., pos_idx] / gradcam.max(),
                (
                    others := gradcam[
                        ..., list(set(range(len(model.categories))) - {pos_idx})
                    ]
                    .max(-1)
                    .values
                )
                / others.max(),
            )  # shape: [tile]

            category_score = (
                category_support * attention / attention.max()
            )  # shape: [tile]

            score_im = cast(
                np.ndarray,
                plt.get_cmap("RdBu_r")(
                    _vals_to_im(category_score.unsqueeze(-1) / 2 + 0.5, coords_norm)
                    .squeeze(-1)
                    .cpu()
                    .detach()
                    .numpy()
                ),
            )

            score_im[..., -1] = (
                (_vals_to_im(attention.unsqueeze(-1), coords_norm).squeeze(-1) > 0)
                .cpu()
                .numpy()
            )

            ax.imshow(score_im)
            ax.set_title(f"{category} {slide_score[pos_idx].item():1.2f}")
            target_size = np.array(score_im.shape[:2][::-1]) * 8

            Image.fromarray(np.uint8(score_im * 255)).resize(
                tuple(target_size), resample=Image.Resampling.NEAREST
            ).save(
                raw_dir / f"{h5_path.stem}-{category}={slide_score[pos_idx]:0.2f}.png"
            )

            # Create and save overlay to raw folder
            overlay = _create_overlay(thumb=thumb, score_im=score_im, alpha=opacity)
            Image.fromarray(overlay).save(
                raw_dir / f"raw-overlay-{h5_path.stem}-{category}.png"
            )

            # Create and save plotted overlay to plots folder
            overlay_fig, overlay_ax = _create_plotted_overlay(
                thumb=thumb,
                score_im=score_im,
                category=category,
                slide_score=slide_score[pos_idx].item(),
                alpha=opacity,
            )
            overlay_fig.savefig(
                plots_dir / f"overlay-{h5_path.stem}-{category}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(overlay_fig)

            # Only extract tiles for the highest probability class
            if pos_idx == highest_prob_class_idx:
                # Top tiles
                for i, (score, index) in enumerate(zip(*category_score.topk(topk))):
                    (
                        slide.read_region(
                            tuple(coords_tile_slide_px[index].tolist()),
                            0,
                            (tile_size_slide_px, tile_size_slide_px),
                        )
                        .convert("RGB")
                        .save(
                            tiles_dir
                            / f"top_{i + 1:02d}-{h5_path.stem}-{category}={score:0.2f}.jpg"
                        )
                    )
                # Bottom tiles
                for i, (score, index) in enumerate(
                    zip(*(-category_score).topk(bottomk))
                ):
                    (
                        slide.read_region(
                            tuple(coords_tile_slide_px[index].tolist()),
                            0,
                            (tile_size_slide_px, tile_size_slide_px),
                        )
                        .convert("RGB")
                        .save(
                            tiles_dir
                            / f"bottom_{i + 1:02d}-{h5_path.stem}-{category}={-score:0.2f}.jpg"
                        )
                    )

        assert attention is not None, (
            "attention should have been set in the for loop above"
        )

        # Save thumbnail to raw folder
        Image.fromarray(thumb).save(raw_dir / f"thumbnail-{h5_path.stem}.png")

        for ax in axs.ravel():
            ax.axis("off")

        # Save overview plot to plots folder
        fig.savefig(plots_dir / f"overview-{h5_path.stem}.png")
        plt.close(fig)