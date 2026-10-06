# LeafLens CNN models

The current browser model is a 38-class CNN trained from the official
PlantVillage `raw/color` image folders. It predicts the source crop/disease
classes and is exported to TensorFlow.js for local browser inference.

The older three-class tomato demonstration assets remain only as historical
project material; they are not used by the current website model.

## Recreate the full model

```bash
PYTHONPATH=model/.python-deps python3 model/train_plantvillage.py
PYTHONPATH=model/.python-deps python3 model/export_plantvillage_tfjs.py
PYTHONPATH=model/.python-deps python3 model/generate_advisories.py
```

The Keras model and metrics are stored in `model/artifacts`. The browser model,
labels, and care guidance are stored in `public/model`.

## Accuracy warning

PlantVillage images have relatively controlled backgrounds. Its held-out
validation score does not establish real-world field accuracy. Confirm a
suspected plant disease locally before applying any crop-protection product.
