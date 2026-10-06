# PlantVillage full classification dataset

LeafLens now trains from the official PlantVillage `raw/color` folder rather
than from the earlier 15-image tomato demonstration set.

- Source: https://github.com/spMohanty/PlantVillage-Dataset
- Images: 54,305 RGB leaf images
- Classes: 38 official crop/disease folders
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
