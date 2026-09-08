from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from omegaconf import OmegaConf

from .types import LatentBundle
from .weight_codec import extract_weight_chunks, reassemble_state_dict


def _ensure_ls_merge_on_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    ls_merge_root = root / "external" / "ls-merge"
    if not ls_merge_root.exists():
        raise FileNotFoundError(f"ls-merge not found at {ls_merge_root}")
    path_str = str(ls_merge_root)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    return ls_merge_root


class VaeBridge:
    """Thin wrapper around ls-merge's AutoencoderKL for encode/decode."""

    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: Optional[str | Path] = None,
        device: str = "cpu",
    ) -> None:
        _ensure_ls_merge_on_path()
        from utils.util import instantiate_from_config

        self.device = torch.device(device)
        config = OmegaConf.load(str(config_path))
        self.model = instantiate_from_config(config.model).to(self.device)
        self.model.eval()

        self.chunk_size = int(config.model.params.enconfig.chunk_size)
        self.input_length = int(config.model.params.enconfig.length)
        self.extract_chunk_size = self.input_length

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        ckpt = torch.load(str(checkpoint_path), map_location=self.device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        self.model.load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def encode_chunks(
        self,
        chunks: torch.Tensor,
        *,
        deterministic: bool = True,
        batch_size: int = 1,
        return_moments: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Encode weight chunks.

        chunks: (num_chunks, chunk_size) or (num_chunks, n_tok, length)
        Returns sampled latents (or mu), mu, logvar.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        latent_tensor = None
        mu_tensor = None
        logvar_tensor = None
        for start in range(0, chunks.shape[0], batch_size):
            stop = min(start + batch_size, chunks.shape[0])
            x = chunks[start:stop].to(self.device).float().reshape(stop - start, -1)
            z, mu, logvar = self.model.encode(x)
            chosen_is_mu = deterministic and mu is not None
            chosen = mu if chosen_is_mu else z
            chosen = chosen.detach().cpu()
            if latent_tensor is None:
                latent_tensor = torch.empty((chunks.shape[0], *chosen.shape[1:]), dtype=chosen.dtype)
            latent_tensor[start:stop] = chosen

            if return_moments and mu is not None:
                if chosen_is_mu:
                    mu_tensor = latent_tensor
                elif mu_tensor is None:
                    mu_tensor = torch.empty((chunks.shape[0], *mu.shape[1:]), dtype=chosen.dtype)
                if mu_tensor is not latent_tensor:
                    mu_tensor[start:stop] = mu.detach().cpu()
            if return_moments and logvar is not None:
                if logvar_tensor is None:
                    logvar_tensor = torch.empty((chunks.shape[0], *logvar.shape[1:]), dtype=chosen.dtype)
                logvar_tensor[start:stop] = logvar.detach().cpu()

        if latent_tensor is None:
            raise ValueError("No chunks supplied for encoding")
        return latent_tensor, mu_tensor, logvar_tensor

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor, reference_chunks: torch.Tensor) -> torch.Tensor:
        """Decode merged latents back to weight chunks."""
        decoded = []
        for idx in range(latents.shape[0]):
            z = latents[idx].to(self.device).float()
            if z.ndim == 1:
                z = z.unsqueeze(0)
            out = self.model.decode(z)
            out = out.reshape(reference_chunks[idx].shape)
            decoded.append(out.detach().cpu())
        return torch.stack(decoded, dim=0)

    def encode_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        source_id: str,
        scale: float = 1.0,
        skip_if_contains=("bias", "norm", "ln"),
        select_layers=None,
        deterministic: bool = True,
        batch_size: int = 1,
    ) -> LatentBundle:
        chunks, mask, metadata, total_numel = extract_weight_chunks(
            state_dict,
            chunk_size=self.extract_chunk_size,
            skip_if_contains=skip_if_contains,
            select_layers=select_layers,
            scale=scale,
        )
        latents, mu, logvar = self.encode_chunks(
            chunks, deterministic=deterministic, batch_size=batch_size
        )
        return LatentBundle(
            latents=latents,
            chunks=chunks,
            mask=mask,
            mu=mu,
            logvar=logvar,
            metadata=metadata,
            total_numel=total_numel,
            scale=scale,
            source_id=source_id,
            original_state_dict=state_dict,
        )

    def decode_bundle(
        self,
        latents: torch.Tensor,
        bundle: LatentBundle,
    ) -> Dict[str, torch.Tensor]:
        if bundle.original_state_dict is None:
            raise ValueError("original_state_dict is required for reassembly")
        reconstructed = self.decode_latents(latents, bundle.chunks)
        return reassemble_state_dict(
            reconstructed,
            bundle.metadata,
            bundle.total_numel,
            bundle.original_state_dict,
            scale=bundle.scale,
        )
