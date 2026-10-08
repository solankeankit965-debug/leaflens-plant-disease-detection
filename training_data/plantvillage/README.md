# PlantVillage full classification dataset

LeafLens now trains from every official PlantVillage `raw` image folder rather
than from the earlier 15-image tomato demonstration set.

- Source: https://github.com/spMohanty/PlantVillage-Dataset
- Images: 162,916 colour, grayscale, and segmented leaf images
- Variants: `raw/color` (54,305), `raw/grayscale` (54,305), and
  `raw/segmented` (54,306)
- Classes: 38 official crop/disease folders
- Crop types: Apple, Blueberry, Cherry, Corn (maize), Grape, Orange, Peach,
  Bell pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato
- Input: 64 × 64 RGB
- Validation: deterministic 80/20 split grouped by original leaf UUID, so
  alternate versions of the same leaf cannot leak between training and test

## Important limits

PlantVillage images use relatively controlled backgrounds. A validation score
from this source does **not** prove accuracy on field photos, different camera
conditions, or early symptoms. Each app result is an image-based suggestion;
users should confirm a suspected disease with a local agriculture extension
service before applying any crop-protection product.

## Recreate the model

```bash
PYTHONPATH=model/.python-deps python3 model/train_plantvillage_all_raw.py
PYTHONPATH=model/.python-deps python3 model/export_plantvillage_tfjs.py
PYTHONPATH=model/.python-deps python3 model/generate_advisories.py
```

The browser uses `public/model/model.json`, `public/model/labels.json`, and
`public/model/advisories.json`.

## Raw-image variants

The source repository provides `raw/color`, `raw/grayscale`, and
`raw/segmented`. They are alternate representations of the same leaf
photographs, not new crop or disease labels. LeafLens now trains on all three
forms, while grouping each original leaf's variants together during validation.
