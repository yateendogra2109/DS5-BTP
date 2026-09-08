#!/usr/bin/env python3
"""Visualize saved latents, or encode a trusted model ``.pth`` with a trained VAE."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.visualization import visualize_latent_pca
from src.merge import VaeBridge
from src.merge.weight_codec import extract_weight_chunks


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Load a trusted raw state_dict or common checkpoint wrapper from ``path``."""
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(artifact, dict):
        for key in ("state_dict", "model_state_dict"):
            candidate = artifact.get(key)
            if isinstance(candidate, dict):
                artifact = candidate
                break
    if not isinstance(artifact, dict) or not artifact or not all(
        isinstance(name, str) and isinstance(weight, torch.Tensor)
        for name, weight in artifact.items()
    ):
        raise ValueError(
            "--model-pth must contain a raw state_dict or a 'state_dict' / "
            "'model_state_dict' mapping of parameter names to tensors"
        )
    return artifact


def load_huggingface_state_dict(model_id: str) -> dict[str, torch.Tensor]:
    """Load a Hugging Face causal-LM state dict without entering merge code."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    state_dict = {name: weight.detach().cpu() for name, weight in model.state_dict().items()}
    del model
    return state_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="PCA of ordered latent weight chunks")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--latent-file", type=Path, help="Trusted torch-saved LatentBundle or bundle-like dict"
    )
    input_group.add_argument(
        "--model-pth", type=Path, help="Trusted model state_dict / checkpoint to encode"
    )
    input_group.add_argument("--model-id", help="Hugging Face model ID or local model directory")
    parser.add_argument("--vae-config", type=Path, help="VAE YAML config; required with --model-pth")
    parser.add_argument(
        "--vae-checkpoint", type=Path, help="Trained VAE .pth checkpoint; required with --model-pth"
    )
    parser.add_argument("--source-id", help="Label stored in the CSV and plot title")
    parser.add_argument("--scale", type=float, default=0.025, help="Weight scale used during VAE training")
    parser.add_argument("--device", default="cpu", help="VAE device, e.g. cpu or cuda")
    parser.add_argument("--batch-size", type=int, default=32, help="Chunks per VAE encoder forward pass")
    parser.add_argument("--pca-batch-size", type=int, default=1024, help="Rows processed per PCA pass")
    parser.add_argument("--output-dir", type=Path, default=Path("results/latent_visualization"))
    args = parser.parse_args()

    if args.model_pth or args.model_id:
        if args.vae_config is None or args.vae_checkpoint is None:
            parser.error("--model-pth/--model-id requires both --vae-config and --vae-checkpoint")
        state_dict = load_state_dict(args.model_pth) if args.model_pth else load_huggingface_state_dict(args.model_id)
        source_id = args.source_id or (args.model_pth.stem if args.model_pth else args.model_id)
        vae = VaeBridge(args.vae_config, checkpoint_path=args.vae_checkpoint, device=args.device)
        chunks, _, metadata, total_numel = extract_weight_chunks(
            state_dict, chunk_size=vae.extract_chunk_size, scale=args.scale, include_mask=False
        )
        latents, _, _ = vae.encode_chunks(
            chunks, deterministic=True, batch_size=args.batch_size, return_moments=False
        )
        del chunks
        paths = visualize_latent_pca(
            latents, metadata, args.output_dir,
            source_id=source_id, total_numel=total_numel, chunk_size=vae.extract_chunk_size,
            pca_batch_size=args.pca_batch_size,
            extra_metadata={
                "model_pth": args.model_pth.resolve() if args.model_pth else None,
                "model_id": args.model_id,
                "vae_config": args.vae_config.resolve(),
                "vae_checkpoint": args.vae_checkpoint.resolve(),
                "scale": args.scale,
                "skip_if_contains": ["bias", "norm", "ln"],
                "vae_batch_size": args.batch_size,
            },
        )
    else:
        artifact = torch.load(args.latent_file, map_location="cpu", weights_only=False)
        get = (lambda key, default=None: artifact.get(key, default)) if isinstance(artifact, dict) else (
            lambda key, default=None: getattr(artifact, key, default)
        )
        paths = visualize_latent_pca(
            get("mu") if get("mu") is not None else get("latents"), get("metadata", []), args.output_dir,
            source_id=args.source_id or get("source_id", args.latent_file.stem),
            total_numel=get("total_numel"),
            chunk_size=int(get("chunks").shape[1]) if get("chunks") is not None else None,
        )
    print("Saved " + ", ".join(str(path) for path in paths.values()))


if __name__ == "__main__":
    main()
