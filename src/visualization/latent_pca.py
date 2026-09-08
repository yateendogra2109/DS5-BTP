"""PCA visualization of one model's ordered latent weight chunks.

Each plotted point is one *input weight chunk*.  If a VAE returns multiple
latent tokens per chunk, those trailing dimensions are flattened only for PCA;
the point still represents the original input chunk, not a model layer.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _json_value(value: Any) -> Any:
    """Convert tensor metadata to values suitable for a reproducibility JSON."""
    if isinstance(value, torch.Size):
        return list(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _chunk_records(
    num_chunks: int,
    metadata: Sequence[Mapping[str, Any]],
    total_numel: int | None,
    chunk_size: int | None,
) -> list[dict[str, Any]]:
    """Map global chunk ranges back to all original tensors they overlap."""
    if chunk_size is None:
        chunk_size = int(np.ceil(total_numel / num_chunks)) if total_numel else None

    records: list[dict[str, Any]] = []
    for index in range(num_chunks):
        start = index * chunk_size if chunk_size is not None else None
        end = min(start + chunk_size, total_numel) if start is not None and total_numel is not None else None
        overlaps = []
        if start is not None and end is not None:
            for tensor in metadata:
                tensor_start = int(tensor.get("offset", -1))
                tensor_end = tensor_start + int(tensor.get("total_numel", 0))
                if tensor_start < end and tensor_end > start:
                    overlaps.append(str(tensor.get("name", "<unnamed>")))

        records.append(
            {
                "chunk_index": index,
                "global_start": start,
                "global_end": end,
                # Metadata has no canonical layer field. Keep tensor provenance
                # instead of inferring a layer from a parameter-name convention.
                "primary_tensor": overlaps[0] if overlaps else "",
                "tensor_names": " | ".join(overlaps),
                "tensor_count": len(overlaps),
            }
        )
    return records


def _pca_2d(
    features: np.ndarray,
    *,
    batch_size: int = 1024,
    oversamples: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run deterministic randomized PCA without materializing a centered matrix.

    The calculation retains one 2-D coordinate per row while processing feature
    batches.  It avoids the prohibitive full SVD of a large chunk-latent matrix.
    """
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError("PCA requires at least two chunk feature vectors")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    rank = min(features.shape[0], features.shape[1], 2 + oversamples)
    rng = np.random.default_rng(0)
    projection = rng.standard_normal((features.shape[1], rank), dtype=np.float32)
    sketch = np.empty((features.shape[0], rank), dtype=np.float32)
    total_variance = 0.0
    for start in range(0, features.shape[0], batch_size):
        stop = min(start + batch_size, features.shape[0])
        centered = features[start:stop] - mean
        sketch[start:stop] = centered @ projection
        total_variance += float(np.square(centered, dtype=np.float32).sum(dtype=np.float64))

    orthogonal, _ = np.linalg.qr(sketch, mode="reduced")
    compressed = np.zeros((rank, features.shape[1]), dtype=np.float32)
    for start in range(0, features.shape[0], batch_size):
        stop = min(start + batch_size, features.shape[0])
        compressed += orthogonal[start:stop].T @ (features[start:stop] - mean)
    _, singular_values, right_vectors = np.linalg.svd(compressed, full_matrices=False)
    components = right_vectors[:2]
    coordinates = np.empty((features.shape[0], 2), dtype=np.float32)
    for start in range(0, features.shape[0], batch_size):
        stop = min(start + batch_size, features.shape[0])
        coordinates[start:stop] = (features[start:stop] - mean) @ components.T
    explained = np.zeros(2) if total_variance == 0 else np.square(singular_values[:2]) / total_variance
    if explained.size < 2:
        explained = np.pad(explained, (0, 2 - explained.size))
    return coordinates, explained, mean


def visualize_latent_pca(
    latents: torch.Tensor | np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    source_id: str = "model",
    total_numel: int | None = None,
    chunk_size: int | None = None,
    annotate_every: int | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
    pca_batch_size: int = 1024,
) -> dict[str, Path]:
    """Save ordered-chunk PCA coordinates, a PNG trajectory, and metadata.

    ``latents`` must have chunks along axis 0. All remaining axes are treated
    as features of that chunk for PCA.
    """
    latent_array = torch.as_tensor(latents).detach().cpu().float().numpy()
    if latent_array.ndim < 2:
        raise ValueError("latents must have shape (num_chunks, ...)")
    num_chunks = latent_array.shape[0]
    features = latent_array.reshape(num_chunks, -1)
    coordinates, explained, feature_mean = _pca_2d(features, batch_size=pca_batch_size)
    records = _chunk_records(num_chunks, metadata, total_numel, chunk_size)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "model_latent_pca.csv"
    plot_path = destination / "model_latent_pca.png"
    metadata_path = destination / "metadata.json"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["source_id", "pca_1", "pca_2", *records[0].keys()]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record, point in zip(records, coordinates):
            writer.writerow({"source_id": source_id, "pca_1": point[0], "pca_2": point[1], **record})

    fig, axis = plt.subplots(figsize=(8, 6))
    order = np.arange(num_chunks)
    axis.plot(coordinates[:, 0], coordinates[:, 1], color="0.65", linewidth=1, zorder=1)
    scatter = axis.scatter(
        coordinates[:, 0], coordinates[:, 1], c=order, cmap="viridis", s=42, zorder=2
    )
    step = annotate_every or max(1, int(np.ceil(num_chunks / 20)))
    for record, point in zip(records[::step], coordinates[::step]):
        axis.annotate(str(record["chunk_index"]), point, xytext=(4, 4), textcoords="offset points", fontsize=7)
    axis.set_title(f"Latent PCA by ordered weight chunk: {source_id}")
    axis.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    axis.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
    fig.colorbar(scatter, ax=axis, label="chunk index")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    reproducibility = {
        "source_id": source_id,
        "visualization_unit": "global weight chunk",
        "latent_shape": list(latent_array.shape),
        "pca_feature_shape": list(features.shape),
        "pca_method": "deterministic randomized PCA on flattened per-chunk latent features",
        "pca_batch_size": pca_batch_size,
        "explained_variance_ratio": explained.tolist(),
        "feature_mean": feature_mean.tolist(),
        "chunk_size": chunk_size,
        "total_numel": total_numel,
        "metadata_per_tensor": _json_value(list(metadata)),
        "chunk_records": records,
    }
    if extra_metadata:
        reproducibility["input_artifacts"] = _json_value(dict(extra_metadata))
    metadata_path.write_text(json.dumps(reproducibility, indent=2), encoding="utf-8")
    return {"csv": csv_path, "plot": plot_path, "metadata": metadata_path}
