# LeafLens — Plant Disease Detection

LeafLens is a browser-based plant disease classifier and advisory system. A user
uploads a leaf image; the website runs a TensorFlow.js CNN locally and displays
the predicted crop/disease class, confidence, alternatives, and practical next
steps.

## Current model

- Dataset: PlantVillage `raw/color`
- Images: 54,305
- Classes: 38 crop/disease labels
- Crops recognised: Apple, Blueberry, Cherry, Corn (maize), Grape, Orange,
  Peach, Bell pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato
- Input: 64 × 64 RGB leaf image
- Held-out validation accuracy: 73.38%
- Held-out top-3 accuracy: 91.31%

The model is trained on controlled PlantVillage images. Its predictions are an
image-based indication, not a replacement for local agricultural diagnosis.
It shows a crop and disease result only when the image has a confident match to
one of the 38 trained labels; it does not guess a label for unsupported crops.

## Run locally

```bash
npm install
npm run dev
```

Open `http://localhost:3000`.

## Training and browser export

See [model/README.md](model/README.md) and
[training_data/plantvillage/README.md](training_data/plantvillage/README.md)
for the dataset, training, validation, and export workflow.

## Source dataset

PlantVillage Dataset by Mohanty, Hughes, and Salathé:
https://github.com/spMohanty/PlantVillage-Dataset
