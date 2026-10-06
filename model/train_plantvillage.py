"""Train a 38-class PlantVillage CNN for the LeafLens browser application.

The source root is the official PlantVillage ``raw/color`` folder. Each source
folder is a real crop/disease label; unlike the earlier demo, no artificial
severity labels are created.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
DATASET = Path(os.environ.get("PLANTVILLAGE_COLOR_DIR", "/private/tmp/plantvillage-source/raw/color"))
ARTIFACTS = ROOT / "model" / "artifacts"
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 64
SEED = 42


def pretty_label(label: str) -> str:
    crop, condition = label.split("___", maxsplit=1)
    crop = crop.replace("_", " ").replace(",", "").replace("(", "").replace(")", "").strip()
    condition = condition.replace("_", " ").strip()
    return f"{crop} — {condition}"


def build_model(class_count: int) -> tf.keras.Model:
    return tf.keras.Sequential(
        [
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
        ],
        name="leaflens_plantvillage_cnn",
    )


def main() -> None:
    if not DATASET.exists():
        raise RuntimeError(f"Dataset directory does not exist: {DATASET}")

    tf.keras.utils.set_random_seed(SEED)
    train = tf.keras.utils.image_dataset_from_directory(
        DATASET, validation_split=0.2, subset="training", seed=SEED,
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode="int",
    )
    validation = tf.keras.utils.image_dataset_from_directory(
        DATASET, validation_split=0.2, subset="validation", seed=SEED,
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode="int",
    )
    class_names = train.class_names

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomContrast(0.08),
    ])
    train = train.map(lambda images, labels: (augment(images, training=True), labels), num_parallel_calls=tf.data.AUTOTUNE)
    train = train.prefetch(tf.data.AUTOTUNE)
    validation = validation.prefetch(tf.data.AUTOTUNE)

    model = build_model(len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=7e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top_3_accuracy")],
    )
    history = model.fit(
        train,
        validation_data=validation,
        epochs=8,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
        verbose=2,
    )
    results = model.evaluate(validation, return_dict=True, verbose=0)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    model.save(ARTIFACTS / "plantvillage_leaf_cnn.keras")
    (ARTIFACTS / "labels.json").write_text(json.dumps({
        "classes": class_names,
        "display_names": {name: pretty_label(name) for name in class_names},
        "image_size": list(IMAGE_SIZE),
        "source": "PlantVillage raw/color",
    }, indent=2) + "\n")
    (ARTIFACTS / "metrics.json").write_text(json.dumps({
        "images": sum(1 for _ in DATASET.glob("*/*")),
        "classes": len(class_names),
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
