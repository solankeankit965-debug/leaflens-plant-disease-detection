"""Train the three-class tomato-leaf CNN used by LeafLens."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "training_data" / "tomato"
ARTIFACTS = ROOT / "model" / "artifacts"
IMAGE_SIZE = (128, 128)
CLASS_NAMES = ["healthy", "semi_diseased", "diseased"]
SEED = 42


def load_dataset() -> tuple[np.ndarray, np.ndarray, list[str]]:
    images: list[np.ndarray] = []
    labels: list[int] = []
    files: list[str] = []

    for class_index, class_name in enumerate(CLASS_NAMES):
        for image_path in sorted((DATASET / class_name).glob("*.jpg")):
            image = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
            images.append(tf.keras.utils.img_to_array(image))
            labels.append(class_index)
            files.append(str(image_path.relative_to(ROOT)))

    class_counts = [labels.count(index) for index in range(len(CLASS_NAMES))]
    if len(images) < 200 or min(class_counts) < 60:
        raise RuntimeError("Expected at least 200 images and at least 60 images in every class folder.")

    return np.asarray(images, dtype=np.float32), np.asarray(labels), files


def build_model() -> tf.keras.Model:
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
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(96, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax", name="condition"),
        ],
        name="leaflens_tomato_cnn",
    )


def stratified_split(labels: np.ndarray, validation_fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    train_indices: list[int] = []
    validation_indices: list[int] = []
    rng = np.random.default_rng(SEED)
    for class_index in range(len(CLASS_NAMES)):
        indices = np.where(labels == class_index)[0]
        rng.shuffle(indices)
        validation_count = max(1, round(len(indices) * validation_fraction))
        validation_indices.extend(indices[:validation_count])
        train_indices.extend(indices[validation_count:])
    return np.asarray(train_indices), np.asarray(validation_indices)


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.config.experimental.enable_op_determinism()

    images, labels, files = load_dataset()
    train_indices, validation_indices = stratified_split(labels)
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=8e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        images[train_indices],
        labels[train_indices],
        validation_data=(images[validation_indices], labels[validation_indices]),
        epochs=80,
        batch_size=12,
        shuffle=True,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)],
        verbose=2,
    )

    train_loss, train_accuracy = model.evaluate(images[train_indices], labels[train_indices], verbose=0)
    validation_loss, validation_accuracy = model.evaluate(images[validation_indices], labels[validation_indices], verbose=0)
    probabilities = model.predict(images[validation_indices], verbose=0)
    predicted = probabilities.argmax(axis=1)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    model.save(ARTIFACTS / "tomato_leaf_cnn.keras")
    model.save(ARTIFACTS / "tomato_leaf_cnn.h5")

    (ARTIFACTS / "labels.json").write_text(
        json.dumps({"classes": CLASS_NAMES, "image_size": list(IMAGE_SIZE)}, indent=2) + "\n"
    )
    (ARTIFACTS / "metrics.json").write_text(
        json.dumps(
            {
                "images": len(images),
                "class_counts": {name: int((labels == index).sum()) for index, name in enumerate(CLASS_NAMES)},
                "epochs": len(history.history["loss"]),
                "training_accuracy": round(float(train_accuracy), 6),
                "training_loss": round(float(train_loss), 6),
                "validation_accuracy": round(float(validation_accuracy), 6),
                "validation_loss": round(float(validation_loss), 6),
                "validation_images": int(len(validation_indices)),
                "correct_on_validation_images": int((predicted == labels[validation_indices]).sum()),
                "warning": "Validation images are from the same PlantVillage source and proxy severity labels are not expert annotations.",
                "predictions": [
                    {
                        "file": files[int(validation_indices[index])],
                        "expected": CLASS_NAMES[int(expected)],
                        "predicted": CLASS_NAMES[int(actual)],
                        "confidence": round(float(probabilities[index, actual]), 6),
                    }
                    for index, (expected, actual) in enumerate(zip(labels[validation_indices], predicted))
                ],
            },
            indent=2,
        )
        + "\n"
    )

    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Validation accuracy: {validation_accuracy:.4f} ({int((predicted == labels[validation_indices]).sum())}/{len(validation_indices)} correct)")
    print(f"Saved model artifacts to: {ARTIFACTS}")


if __name__ == "__main__":
    main()
