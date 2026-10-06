"""Expand the LeafLens demo dataset to 210 independent PlantVillage images.

The original dataset has authoritative PlantVillage labels for healthy and
Tomato Early Blight only. The app's semi_diseased and diseased labels are
therefore reproducible *severity proxy* groups, created by ranking Early Blight
images by the amount of yellow/brown lesion-like colour in the leaf image.
They are not expert severity annotations.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "training_data" / "tomato"
SOURCE = Path("/private/tmp/plantvillage-source/raw/color")
PER_CLASS = 70


def lesion_score(image_path: Path) -> float:
    """Return a simple, reproducible colour-based lesion proxy score."""
    image = Image.open(image_path).convert("RGB").resize((128, 128))
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    yellow_or_brown = (
        (red > 0.23)
        & (green > 0.10)
        & (red > green * 1.08)
        & (green > blue * 1.12)
    )
    return float(yellow_or_brown.mean())


def listed_source_names() -> set[str]:
    with (DATASET / "manifest.csv").open(newline="") as file:
        return {row["source_filename"] for row in csv.DictReader(file)}


def copy_images(paths: list[Path], label: str, start: int, rows: list[dict[str, str]], origin: str) -> None:
    destination = DATASET / label
    for offset, source in enumerate(paths, start=start):
        target_name = f"{label}_{offset:03d}.jpg"
        target = destination / target_name
        shutil.copy2(source, target)
        rows.append(
            {
                "project_file": f"{label}/{target_name}",
                "project_label": label,
                "source_class": source.parent.name,
                "source_filename": source.name,
                "severity_label_origin": origin,
            }
        )


def main() -> None:
    if not (SOURCE / "Tomato___healthy").exists() or not (SOURCE / "Tomato___Early_blight").exists():
        raise RuntimeError("PlantVillage source folders are missing. Download them before expanding the dataset.")

    current = {name: sorted((DATASET / name).glob("*.jpg")) for name in ("healthy", "semi_diseased", "diseased")}
    if any(len(files) != 5 for files in current.values()):
        raise RuntimeError("This script expects the original five images in each class folder.")

    used = listed_source_names()
    healthy = [path for path in sorted((SOURCE / "Tomato___healthy").glob("*")) if path.name not in used]
    early_blight = [path for path in sorted((SOURCE / "Tomato___Early_blight").glob("*")) if path.name not in used]
    if len(healthy) < PER_CLASS - 5 or len(early_blight) < 2 * (PER_CLASS - 5):
        raise RuntimeError("Not enough independent PlantVillage images to reach 70 images per class.")

    new_rows: list[dict[str, str]] = []
    copy_images(healthy[: PER_CLASS - 5], "healthy", 6, new_rows, "PlantVillage class")

    ranked = sorted(((lesion_score(path), path) for path in early_blight), key=lambda item: (item[0], item[1].name))
    count = PER_CLASS - 5
    copy_images([path for _, path in ranked[:count]], "semi_diseased", 6, new_rows, "automated visual severity proxy")
    copy_images([path for _, path in ranked[-count:]], "diseased", 6, new_rows, "automated visual severity proxy")

    manifest = DATASET / "manifest.csv"
    with manifest.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["project_file", "project_label", "source_class", "source_filename", "severity_label_origin"])
        writer.writerows(new_rows)

    print("Dataset expanded to 210 independent images: 70 healthy, 70 semi_diseased, 70 diseased.")


if __name__ == "__main__":
    main()
