import os
import platform
import shutil
from pathlib import Path
from typing import Annotated, Literal

import torch
from pydantic import Field

# Define your safe workspace root (absolute path)
WORKSPACE_ROOT = os.path.abspath("agent_workspace")
base_dir = "./"
base = Path(base_dir).resolve()


def manage_workspace_dirs_(
    action: Annotated[
        Literal["create", "remove"],
        Field(
            description="Action to perform: 'create' to make a folder, 'remove' to delete it."
        ),
    ],
    folder_name: Annotated[
        str, Field(description="Relative path of the folder inside the workspace.")
    ],
) -> str:
    """
    Create or remove folders safely inside the predefined workspace.

    Example:
        manage_workspace_dirs(action='create', folder_name='feats')
    """
    # --- resolve safe paths ---
    target_path = os.path.abspath(os.path.join(WORKSPACE_ROOT, folder_name))
    if not target_path.startswith(WORKSPACE_ROOT):
        return f"❌ Access denied: '{folder_name}' is outside workspace."

    # --- perform the action ---
    try:
        if action == "create":
            os.makedirs(target_path, exist_ok=True)
            return f"📁 Created (or already exists): {target_path}"

        elif action == "remove":
            if os.path.isdir(target_path):
                shutil.rmtree(target_path)
                return f"🗑️ Removed folder: {target_path}"
            else:
                return f"⚠️ Folder not found: {target_path}"

        else:
            return f"❌ Invalid action '{action}'. Use 'create' or 'remove'."

    except Exception as e:
        return f"❌ Error while performing {action} on '{folder_name}': {e}"


def _resolve_path(subpath: str) -> Path:
    requested = (base / subpath).resolve()
    if base not in requested.parents and requested != base:
        raise PermissionError(f"Access denied: {subpath}")
    return requested


def read_file(path: str) -> str:
    """
    Read the contents of a file inside the allowed folder.

    Args:
        path (str): Relative path to the file.

    Returns:
        str: Content of the file.
    """
    safe_path = _resolve_path(path)
    with open(safe_path, "r", encoding="utf-8") as f:
        return f.read()


def list_files(subdir: str = "") -> list:
    """
    List all files and directories under the given subdirectory (default is root), recursively,
    returning paths relative to the base directory.

    Args:
        subdir (str): Relative subdirectory path to list files from.

    Returns:
        list: List of relative file paths found.
    """
    safe = _resolve_path(subdir)
    if not safe.is_dir():
        raise FileNotFoundError(f"Subdirectory does not exist: {subdir}")
    results = []
    base_len = len(str(base)) + 1  # To slice off base path + separator
    for root, dirs, files in os.walk(safe):
        rel_root = str(root)[base_len:]  # relative path under base_dir
        for d in dirs:
            path = os.path.join(rel_root, d)
            results.append(path + "/")
        for f in files:
            path = os.path.join(rel_root, f)
            results.append(path)
    return sorted(results)


def check_available_devices() -> str:
    """
    Check which computation devices are available on the system.
    This includes checking for cuda (NVIDIA GPUs) and mps (Apple Silicon GPUs).

    Returns:
        A string describing the available devices.
    """
    devices = []

    # Check for CUDA availability
    if torch.cuda.is_available():
        devices.append("cuda")

    # Check for MPS availability (Apple Silicon GPUs)
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        devices.append("mps")

    # Check for CPU (always available)
    devices.append("cpu")

    # Format the result
    if devices:
        return f"Available devices: {', '.join(devices)}"
    else:
        return "No computation devices are available."
