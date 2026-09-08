"""Synthetic, offline test for the latent PCA visualization."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.visualization import visualize_latent_pca
from src.merge.vae_bridge import VaeBridge


def _visualization_script_module():
    import importlib.util

    script_path = ROOT / "scripts" / "visualize_latent_pca.py"
    spec = importlib.util.spec_from_file_location("visualize_latent_pca_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_synthetic_latent_pca() -> None:
    latents = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)
    metadata = [
        {"name": "blocks.0.attn.weight", "offset": 0, "total_numel": 5},
        {"name": "blocks.1.mlp.weight", "offset": 5, "total_numel": 9},
    ]
    with tempfile.TemporaryDirectory() as temporary_directory:
        paths = visualize_latent_pca(
            latents, metadata, temporary_directory, source_id="synthetic", total_numel=14, chunk_size=4
        )
        assert all(path.exists() for path in paths.values())
        with paths["csv"].open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 4
        assert [int(row["chunk_index"]) for row in rows] == [0, 1, 2, 3]
        assert rows[1]["tensor_names"] == "blocks.0.attn.weight | blocks.1.mlp.weight"
        result_metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        assert result_metadata["latent_shape"] == [4, 2, 3]
        assert result_metadata["pca_feature_shape"] == [4, 6]
        assert result_metadata["chunk_records"][1]["tensor_count"] == 2


def test_model_checkpoint_loader() -> None:
    script = _visualization_script_module()
    expected = {"layers.0.weight": torch.ones(2, 2)}
    with tempfile.TemporaryDirectory() as temporary_directory:
        checkpoint_path = Path(temporary_directory) / "model.pth"
        torch.save({"model_state_dict": expected, "epoch": 1}, checkpoint_path)
        loaded = script.load_state_dict(checkpoint_path)
    assert torch.equal(loaded["layers.0.weight"], expected["layers.0.weight"])


def test_batched_encoding() -> None:
    class FakeVae:
        def encode(self, x):
            z = x.reshape(x.shape[0], 2, 2)
            return z, z + 1, z + 2

    bridge = object.__new__(VaeBridge)
    bridge.device = torch.device("cpu")
    bridge.model = FakeVae()
    chunks = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    latents, mu, logvar = bridge.encode_chunks(chunks, batch_size=2)
    assert latents.shape == (5, 2, 2)
    assert torch.equal(latents, chunks.reshape(5, 2, 2) + 1)
    assert mu is latents
    assert torch.equal(logvar, chunks.reshape(5, 2, 2) + 2)


if __name__ == "__main__":
    test_synthetic_latent_pca()
    test_model_checkpoint_loader()
    test_batched_encoding()
    print("Synthetic latent PCA test passed.")
