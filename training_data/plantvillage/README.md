# PlantVillage full classification dataset

LeafLens now trains from the official PlantVillage `raw/color` folder rather
than from the earlier 15-image tomato demonstration set.

- Source: https://github.com/spMohanty/PlantVillage-Dataset
- Images: 54,305 RGB leaf images
- Classes: 38 official crop/disease folders
- Crop types: Apple, Blueberry, Cherry, Corn (maize), Grape, Orange, Peach,
  Bell pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato
- Input: 64 × 64 RGB
- Validation: deterministic 80/20 split by image file

## Important limits

PlantVillage images use relatively controlled backgrounds. A validation score
from this source does **not** prove accuracy on field photos, different camera
conditions, or early symptoms. Each app result is an image-based suggestion;
users should confirm a suspected disease with a local agriculture extension
service before applying any crop-protection product.

## Recreate the model

```bash
PYTHONPATH=model/.python-deps python3 model/train_plantvillage.py
PYTHONPATH=model/.python-deps python3 model/export_plantvillage_tfjs.py
PYTHONPATH=model/.python-deps python3 model/generate_advisories.py
```

The browser uses `public/model/model.json`, `public/model/labels.json`, and
`public/model/advisories.json`.

## Why `raw/color` is used

The source repository also provides `raw/grayscale` and `raw/segmented`.
Those are alternate representations of the same leaf photographs, not new crop
or disease labels. LeafLens trains on the original RGB `raw/color` images so a
normal colour photo uploaded to the website matches the training input.
