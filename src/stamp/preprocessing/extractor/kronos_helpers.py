# import os

# import torch

# from stamp.preprocessing.extractor.kronos import create_model_from_pretrained

# # --- Load pretrained Kronos model ---
# model, precision, embedding_dim = create_model_from_pretrained(
#     "hf_hub:MahmoodLab/KRONOS",
#     cfg={
#         "model_type": "vits16",
#         "token_overlap": True,
#     },
# )

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device).eval()

# # --- Dummy multiplex input ---
# batch_size = 2
# num_markers = 18  # your marker count
# height = width = 224  # Kronos default

# x = torch.randn(batch_size, num_markers, height, width, device=device)
# marker_ids = [torch.arange(num_markers, device=device) for _ in range(batch_size)]

# print("Running forward pass...")
# with torch.no_grad():
#     patch_features, marker_features, patch_token_features = model(
#         x, marker_ids=marker_ids
#     )

# print("✅ Forward pass successful!")
# print(f"Patch-level feature shape: {patch_features.shape}")
# print(f"Marker-level feature shape: {marker_features.shape}")
# print(f"Token-level feature shape: {patch_token_features.shape}")

"""
Kronos feature extractor wrapper for STAMP.
Integrates the pretrained Kronos ViT-S16 model into the STAMP Extractor interface.
"""

import torch
from torch import nn
from torchvision import transforms

from stamp.preprocessing.config import ExtractorName
from stamp.preprocessing.extractor import Extractor
from stamp.preprocessing.extractor.kronos import create_model_from_pretrained

__author__ = "Minh Duc Nguyen, Marko van Treeck"
__copyright__ = "Copyright (C) 2025"
__license__ = "MIT"


# ----------------------------------------------------------------------
# Kronos Wrapper
# ----------------------------------------------------------------------


class KRONOS(nn.Module):
    """A wrapper around Kronos to make it compatible with STAMP Extractor.

    This ensures that Kronos returns only patch_features (CLS token embeddings)
    instead of a tuple of three outputs.
    """

    def __init__(self, kronos_model: nn.Module):
        super().__init__()
        self.kronos_model = kronos_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, num_markers, H, W)

        Returns:
            patch_features: Tensor of shape (B, D)
        """
        B, num_markers, _, _ = x.shape
        marker_ids = [torch.arange(num_markers, device=x.device) for _ in range(B)]
        patch_features, _, _ = self.kronos_model(x, marker_ids=marker_ids)
        return patch_features


def kronos() -> Extractor:
    """Return Kronos ViT-S16 extractor compatible with STAMP."""
    # Load Kronos pretrained model
    model, _, _ = create_model_from_pretrained(
        checkpoint_path="hf_hub:MahmoodLab/KRONOS",
        cfg={
            "model_type": "vits16",
            "token_overlap": True,
        },
    )

    # Wrap Kronos to return only patch_features
    model = KRONOS(model)

    # Define the same type of transform as other extractors
    transform = transforms.Compose([])

    return Extractor(
        model=model,
        transform=transform,
        identifier=ExtractorName.KRONOS,  # add to ExtractorName Enum
    )
