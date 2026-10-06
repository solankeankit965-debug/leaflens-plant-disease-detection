"""Create practical, class-specific next-step guidance for LeafLens."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "model" / "artifacts" / "labels.json"
OUTPUT = ROOT / "public" / "model" / "advisories.json"


def advice(label: str) -> dict[str, str | list[str]]:
    crop, condition = label.split("___", 1)
    crop = crop.replace("_", " ").replace(",", "").replace("(", "").replace(")", "").strip()
    disease = condition.replace("_", " ").strip()
    if condition.lower() == "healthy":
        return {"crop": crop, "condition": "Healthy", "summary": "No matching disease pattern was found in this image.", "actions": ["Continue regular crop monitoring.", "Photograph a single leaf in even light if symptoms appear.", "Keep records of any change in colour, spots, or leaf curl."], "urgency": "Low"}
    if "Yellow Leaf Curl" in condition or "mosaic" in condition.lower():
        actions = ["Isolate or remove clearly affected plants to reduce spread.", "Check nearby plants and manage insect vectors such as whiteflies.", "Use clean planting material and ask a local agriculture extension officer for crop-specific action."]
        urgency = "High"
    elif "Spider_mites" in condition:
        actions = ["Inspect the undersides of leaves for mites and fine webbing.", "Reduce plant stress and avoid dusty, dry conditions.", "Use only locally approved mite-control methods after confirming the pest."]
        urgency = "Medium"
    elif "Haunglongbing" in condition:
        actions = ["Inspect the full tree for uneven yellowing and fruit symptoms.", "Report suspected citrus greening through your local citrus or agriculture authority.", "Manage citrus psyllid vectors only with locally approved guidance."]
        urgency = "High"
    elif "Bacterial" in condition:
        actions = ["Avoid handling plants while leaves are wet.", "Remove badly affected leaves and clean tools between plants.", "Use disease-free seed or transplants and ask a local extension officer about registered control options."]
        urgency = "Medium"
    else:
        actions = ["Remove severely affected leaves and dispose of them away from the crop.", "Improve airflow and avoid prolonged leaf wetness or overhead watering.", "Confirm the diagnosis locally before using any crop-protection product."]
        urgency = "Medium"
    return {"crop": crop, "condition": disease, "summary": f"The image most closely matches {disease} on {crop}.", "actions": actions, "urgency": urgency}


def main() -> None:
    labels = json.loads(LABELS.read_text())["classes"]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({label: advice(label) for label in labels}, indent=2) + "\n")
    print(f"Wrote advice for {len(labels)} PlantVillage classes.")


if __name__ == "__main__":
    main()
