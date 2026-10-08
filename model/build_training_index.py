"""Build browser data for LeafLens' hybrid recognition flow.

The output has two roles:
1. SHA-256 hashes give an authoritative label for an exact uploaded
   PlantVillage source image. This is a lookup, not a model guess.
2. Per-class feature prototypes let the browser compare an unseen leaf's
   penultimate CNN features with patterns learned from the training dataset.

Run after the PlantVillage raw folders and Keras model are available:
    PYTHONPATH=model/.python-deps python3 model/build_training_index.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = ROOT / "training_data" / "plantvillage" / "raw"
MODEL_PATH = ROOT / "model" / "artifacts" / "plantvillage_leaf_cnn.keras"
OUTPUT = ROOT / "public" / "model" / "training-index.json"
VARIANTS = ("color", "grayscale", "segmented")
IMAGE_SIZE = (64, 64)
PROTOTYPE_SAMPLES_PER_CLASS = 250


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_image(path: Path) -> tf.Tensor:
    image = tf.io.decode_image(tf.io.read_file(str(path)), channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    return tf.image.resize(image, IMAGE_SIZE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build exact PlantVillage lookup and feature prototypes for LeafLens.")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT, help="PlantVillage raw folder containing color, grayscale and segmented.")
    args = parser.parse_args()
    raw_root: Path = args.raw_root
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Missing trained model: {MODEL_PATH}")
    if not all((raw_root / variant).exists() for variant in VARIANTS):
        raise RuntimeError("Expected PlantVillage raw/color, raw/grayscale and raw/segmented folders.")

    hashes: dict[str, str] = {}
    samples: dict[str, list[Path]] = defaultdict(list)
    for variant in VARIANTS:
        for class_dir in sorted((raw_root / variant).iterdir()):
            if not class_dir.is_dir():
                continue
            for image_path in sorted(class_dir.glob("*")):
                if image_path.is_file():
                    hashes[sha256(image_path)] = class_dir.name
                    # Colour images are the most useful source for a compact,
                    # representative class prototype.
                    if variant == "color" and len(samples[class_dir.name]) < PROTOTYPE_SAMPLES_PER_CLASS:
                        samples[class_dir.name].append(image_path)

    model = tf.keras.models.load_model(MODEL_PATH)
    feature_layer = next(layer for layer in model.layers if layer.name == "dense")
    feature_model = tf.keras.Model(inputs=model.inputs, outputs=feature_layer.output)
    prototypes: dict[str, list[float]] = {}
    for label, paths in sorted(samples.items()):
        vectors: list[np.ndarray] = []
        for start in range(0, len(paths), 64):
            batch = tf.stack([load_image(path) for path in paths[start : start + 64]])
            vectors.append(feature_model(batch, training=False).numpy())
        mean = np.concatenate(vectors, axis=0).mean(axis=0)
        prototypes[label] = [round(float(value), 8) for value in mean]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({
        "version": 1,
        "hash_algorithm": "SHA-256",
        "source": "PlantVillage raw/color + raw/grayscale + raw/segmented",
        "exact_image_count": len(hashes),
        "prototype_samples_per_class": PROTOTYPE_SAMPLES_PER_CLASS,
        "hashes": hashes,
        "prototypes": prototypes,
    }, separators=(",", ":")) + "\n")
    print(f"Wrote {len(hashes):,} exact hashes and {len(prototypes)} class prototypes to {OUTPUT}")


if __name__ == "__main__":
    main()
