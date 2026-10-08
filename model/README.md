# LeafLens CNN models

The current browser model is a 38-class CNN trained from every official
PlantVillage raw image folder: `raw/color`, `raw/grayscale`, and
`raw/segmented` (162,916 images). It predicts the source crop/disease classes
and is exported to TensorFlow.js for local browser inference.

The older three-class tomato demonstration assets remain only as historical
project material; they are not used by the current website model.

## Recreate the full model

```bash
PYTHONPATH=model/.python-deps python3 model/train_plantvillage_all_raw.py
PYTHONPATH=model/.python-deps python3 model/export_plantvillage_tfjs.py
PYTHONPATH=model/.python-deps python3 model/generate_advisories.py
```

The Keras model and metrics are stored in `model/artifacts`. The browser model,
labels, and care guidance are stored in `public/model`.

The training script groups alternate image variants of the same original leaf
by UUID before making its deterministic 80/20 split. This avoids data leakage.

## Accuracy warning

PlantVillage images have relatively controlled backgrounds. Its held-out
validation score does not establish real-world field accuracy. Confirm a
suspected plant disease locally before applying any crop-protection product.
