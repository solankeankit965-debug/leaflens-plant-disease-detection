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

## Hybrid recognition index

`model/build_training_index.py` adds a second, safe recognition path without
replacing the CNN:

- An image whose uploaded bytes exactly match a known PlantVillage source image
  is found through a SHA-256 lookup and receives that dataset's authoritative
  crop/disease label.
- A new image is checked for usable detail, evaluated by the CNN, and (when
  generated) compared with per-class CNN feature prototypes. Strong CNN matches
  are shown as results; medium matches are labelled as similarities to verify;
  weak matches are rejected as unsupported.

Run the index builder after downloading the official raw folders and training
the Keras model:

```bash
PYTHONPATH=model/.python-deps python3 model/build_training_index.py
```

It writes `public/model/training-index.json`. This can be a sizeable browser
asset because it contains one secure hash per dataset image; the website loads
it only when the user starts an analysis, not when the page first opens.

The training script groups alternate image variants of the same original leaf
by UUID before making its deterministic 80/20 split. This avoids data leakage.

## Accuracy warning

PlantVillage images have relatively controlled backgrounds. Its held-out
validation score does not establish real-world field accuracy. Confirm a
suspected plant disease locally before applying any crop-protection product.
