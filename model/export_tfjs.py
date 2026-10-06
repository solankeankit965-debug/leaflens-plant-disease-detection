"""Export the trained Keras model as a browser-compatible TF.js LayersModel.

The model has a fixed Sequential architecture. This lightweight exporter writes
the HDF5 weights in TensorFlow.js's documented binary-manifest format, avoiding
the optional SavedModel dependencies pulled in by the full converter package.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "model" / "artifacts" / "tomato_leaf_cnn.h5"
OUTPUT = ROOT / "public" / "model"


WEIGHT_PATHS = {
    "condition/kernel": "model_weights/condition/leaflens_tomato_cnn/condition/kernel",
    "condition/bias": "model_weights/condition/leaflens_tomato_cnn/condition/bias",
    "conv2d/kernel": "model_weights/conv2d/leaflens_tomato_cnn/conv2d/kernel",
    "conv2d/bias": "model_weights/conv2d/leaflens_tomato_cnn/conv2d/bias",
    "conv2d_1/kernel": "model_weights/conv2d_1/leaflens_tomato_cnn/conv2d_1/kernel",
    "conv2d_1/bias": "model_weights/conv2d_1/leaflens_tomato_cnn/conv2d_1/bias",
    "conv2d_2/kernel": "model_weights/conv2d_2/leaflens_tomato_cnn/conv2d_2/kernel",
    "conv2d_2/bias": "model_weights/conv2d_2/leaflens_tomato_cnn/conv2d_2/bias",
    "dense/kernel": "model_weights/dense/leaflens_tomato_cnn/dense/kernel",
    "dense/bias": "model_weights/dense/leaflens_tomato_cnn/dense/bias",
}


def write_shards(data: bytes, paths: list[str]) -> None:
    shard_size = (len(data) + len(paths) - 1) // len(paths)
    for index, path in enumerate(paths):
        start = index * shard_size
        (OUTPUT / path).write_bytes(data[start : start + shard_size])


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT / "model.json"
    model_json = json.loads(model_path.read_text())
    manifest = model_json["weightsManifest"]
    if len(manifest) != 1:
        raise RuntimeError("Expected exactly one TensorFlow.js weight group.")

    ordered_weights: list[bytes] = []
    with h5py.File(SOURCE, "r") as source:
        for entry in manifest[0]["weights"]:
            name = entry["name"]
            weight = np.asarray(source[WEIGHT_PATHS[name]], dtype="<f4")
            if list(weight.shape) != entry["shape"]:
                raise RuntimeError(f"Unexpected shape for {name}: {weight.shape}")
            ordered_weights.append(weight.tobytes(order="C"))
    write_shards(b"".join(ordered_weights), manifest[0]["paths"])

    labels = ROOT / "model" / "artifacts" / "labels.json"
    (OUTPUT / "labels.json").write_text(labels.read_text())

    print(
        f"Exported {model_json['format']} with "
        f"{len(model_json['weightsManifest'][0]['weights'])} weight tensors."
    )


if __name__ == "__main__":
    main()
