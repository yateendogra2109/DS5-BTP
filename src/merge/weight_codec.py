from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .types import LatentBundle


def _ensure_ls_merge_on_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    ls_merge_root = root / "external" / "ls-merge"
    if not ls_merge_root.exists():
        raise FileNotFoundError(f"ls-merge not found at {ls_merge_root}")
    path_str = str(ls_merge_root)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    return ls_merge_root


def extract_weight_chunks(
    state_dict: Dict[str, torch.Tensor],
    chunk_size: int,
    *,
    skip_if_contains: Sequence[str] = ("bias", "norm", "ln"),
    select_layers: Optional[Sequence[str]] = None,
    normalize: Optional[str] = None,
    scale: float = 1.0,
    include_mask: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[dict], int]:
    """
    Convert a model state_dict into VAE-ready chunks using ls-merge utilities.

    Returns:
      chunks: (num_chunks, chunk_size)
      mask:   (num_chunks, chunk_size)
      metadata, total_numel
    """
    _ensure_ls_merge_on_path()
    from zoodatasets.base_datasets import collect_flat_weights_with_metadata

    chunks, metadata, total_numel = collect_flat_weights_with_metadata(
        state_dict,
        chunk_size=chunk_size,
        normalize=normalize,
        skip_if_contains=skip_if_contains,
        select_layers=select_layers,
    )
    if chunks is None:
        raise ValueError("No weights extracted from state_dict")

    # The VAE encoder does not consume this mask.  Visualization can omit it
    # to avoid an additional full-size allocation for large models.
    mask = torch.ones_like(chunks) if include_mask else None
    if scale != 1.0:
        chunks = chunks / scale

    return chunks, mask, metadata, total_numel


def reassemble_state_dict(
    reconstructed_chunks: torch.Tensor,
    metadata: List[dict],
    total_numel: int,
    original_state_dict: Dict[str, torch.Tensor],
    *,
    scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Map decoded chunks back into a HuggingFace-compatible state_dict."""
    _ensure_ls_merge_on_path()
    from zoodatasets.base_datasets import reassemble_state_dict as _reassemble

    chunks = reconstructed_chunks
    if scale != 1.0:
        chunks = chunks * scale

    return _reassemble(chunks, metadata, total_numel, original_state_dict)
