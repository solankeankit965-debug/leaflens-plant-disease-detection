"""Train LeafLens on every official PlantVillage raw image variant.

This includes raw/color, raw/grayscale and raw/segmented. Variants with the
same image UUID are always put in the same split, preventing the validation
set from containing a transformed copy of a training leaf.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = Path(os.environ.get("PLANTVILLAGE_RAW_DIR", "/private/tmp/plantvillage-all-raw/raw"))
ARTIFACTS = ROOT / "model" / "artifacts"
VARIANTS = ("color", "grayscale", "segmented")
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 128
SEED = 42
EPOCHS = 5


def pretty_label(label: str) -> str:
    crop, condition = label.split("___", maxsplit=1)
    crop = crop.replace("_", " ").replace(",", "").replace("(", "").replace(")", "").strip()
    condition = condition.replace("_", " ").strip()
    return f"{crop} — {condition}"


def leaf_group(path: Path) -> str:
    """Use the UUID shared by RGB, grayscale and segmented leaf variants."""
    return path.name.split("___", maxsplit=1)[0]


def belongs_to_validation(label: str, group: str) -> bool:
    digest = hashlib.sha256(f"{SEED}:{label}:{group}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % 5 == 0


def collect_examples() -> tuple[list[str], list[int], list[str], list[int], list[str]]:
    color_root = RAW_ROOT / "color"
    labels = sorted(directory.name for directory in color_root.iterdir() if directory.is_dir())
    label_index = {label: index for index, label in enumerate(labels)}
    train_paths: list[str] = []
    train_labels: list[int] = []
    validation_paths: list[str] = []
    validation_labels: list[int] = []

    for variant in VARIANTS:
        variant_root = RAW_ROOT / variant
        for label in labels:
            for image in variant_root.joinpath(label).glob("*.*"):
                target_paths = validation_paths if belongs_to_validation(label, leaf_group(image)) else train_paths
                target_labels = validation_labels if belongs_to_validation(label, leaf_group(image)) else train_labels
                target_paths.append(str(image))
                target_labels.append(label_index[label])
    return train_paths, train_labels, validation_paths, validation_labels, labels


def dataset(paths: list[str], labels: list[int], training: bool) -> tf.data.Dataset:
    data = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        data = data.shuffle(min(len(paths), 20_000), seed=SEED, reshuffle_each_iteration=True)

    def decode(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMAGE_SIZE)
        return image, label

    return data.map(decode, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_model(class_count: int) -> tf.keras.Model:
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3), name="leaf_image"),
        tf.keras.layers.Rescaling(1.0 / 255.0),
        tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(class_count, activation="softmax", name="condition"),
    ], name="leaflens_plantvillage_all_raw_cnn")


def main() -> None:
    if not all((RAW_ROOT / variant).exists() for variant in VARIANTS):
        raise RuntimeError(f"Expected color, grayscale and segmented folders inside {RAW_ROOT}")

    tf.keras.utils.set_random_seed(SEED)
    train_paths, train_labels, validation_paths, validation_labels, class_names = collect_examples()
    print(f"Training images: {len(train_paths):,}; validation images: {len(validation_paths):,}")
    train = dataset(train_paths, train_labels, training=True)
    validation = dataset(validation_paths, validation_labels, training=False)
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomContrast(0.08),
    ])
    train = train.map(lambda images, labels: (augment(images, training=True), labels), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    model = build_model(len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=7e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top_3_accuracy")],
    )
    history = model.fit(
        train, validation_data=validation, epochs=EPOCHS,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
        verbose=2,
    )
    results = model.evaluate(validation, return_dict=True, verbose=0)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    model.save(ARTIFACTS / "plantvillage_leaf_cnn.keras")
    (ARTIFACTS / "labels.json").write_text(json.dumps({
        "classes": class_names,
        "display_names": {name: pretty_label(name) for name in class_names},
        "image_size": list(IMAGE_SIZE),
        "source": "PlantVillage raw/color + raw/grayscale + raw/segmented",
        "variants": list(VARIANTS),
    }, indent=2) + "\n")
    (ARTIFACTS / "metrics.json").write_text(json.dumps({
        "images": len(train_paths) + len(validation_paths),
        "classes": len(class_names),
        "variants": list(VARIANTS),
        "split": "80/20 deterministic split grouped by original leaf UUID",
        "epochs": len(history.history["loss"]),
        "validation_accuracy": round(float(results["accuracy"]), 6),
        "validation_top_3_accuracy": round(float(results["top_3_accuracy"]), 6),
        "validation_loss": round(float(results["loss"]), 6),
        "warning": "PlantVillage images are controlled laboratory-style images. Validate with local field images before using predictions for crop-management decisions.",
    }, indent=2) + "\n")
    print(f"Validation accuracy: {results['accuracy']:.4f}")
    print(f"Validation top-3 accuracy: {results['top_3_accuracy']:.4f}")


if __name__ == "__main__":
    main()
