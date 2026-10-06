"""Export the PlantVillage Keras model to a browser-loadable TensorFlow.js model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "model" / "artifacts" / "plantvillage_leaf_cnn.keras"
OUTPUT = ROOT / "public" / "model"


def normalize_config(value):
    if isinstance(value, list):
        return [normalize_config(item) for item in value]
    if not isinstance(value, dict):
        return value
    normalized = {key: normalize_config(item) for key, item in value.items()}
    config = normalized.get("config")
    if isinstance(config, dict):
        if normalized.get("class_name") == "InputLayer" and "batch_shape" in config:
            config["batch_input_shape"] = config.pop("batch_shape")
            config.pop("optional", None)
        if isinstance(config.get("dtype"), dict):
            config["dtype"] = config["dtype"].get("config", {}).get("name", "float32")
        if config.get("groups") == 1:
            config.pop("groups")
        config.pop("quantization_config", None)
    normalized.pop("module", None)
    normalized.pop("registered_name", None)
    return normalized


def main() -> None:
    model = tf.keras.models.load_model(SOURCE)
    output = OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    for old_shard in output.glob("group1-shard*.bin"):
        old_shard.unlink()

    weights: list[dict] = []
    blobs: list[bytes] = []
    for layer in model.layers:
        for variable in layer.weights:
            name = variable.name.split(":", maxsplit=1)[0]
            value = np.asarray(variable.numpy(), dtype="<f4")
            weights.append({"name": f"{layer.name}/{name}", "shape": list(value.shape), "dtype": "float32"})
            blobs.append(value.tobytes(order="C"))
    payload = b"".join(blobs)
    shard_size = 4 * 1024 * 1024
    paths: list[str] = []
    for index, start in enumerate(range(0, len(payload), shard_size), start=1):
        filename = f"group1-shard{index}of{(len(payload) + shard_size - 1) // shard_size}.bin"
        (output / filename).write_bytes(payload[start : start + shard_size])
        paths.append(filename)

    topology = {"keras_version": tf.keras.version(), "backend": "tensorflow", "model_config": normalize_config(json.loads(model.to_json()))}
    (output / "model.json").write_text(json.dumps({
        "format": "layers-model",
        "generatedBy": f"keras v{tf.keras.version()}",
        "convertedBy": "LeafLens lightweight exporter",
        "modelTopology": topology,
        "weightsManifest": [{"paths": paths, "weights": weights}],
    }, indent=2) + "\n")
    (output / "labels.json").write_text((ROOT / "model" / "artifacts" / "labels.json").read_text())
    print(f"Exported {len(weights)} tensors in {len(paths)} shard(s).")


if __name__ == "__main__":
    main()
