# ClayTerratorch

End-to-end LULC prediction pipeline for Uttarakhand using Meta's **Clay v1.5** foundation model as a frozen encoder and **TerraTorch's UNetDecoder** as a trainable segmentation head.

---

## How it works

```
Landsat-C2-L2 (Planetary Computer)
        │
        ▼
  Paired tiles: 224×224 px, 6 bands + LULC mask
        │
        ├── train/val/test split (stratified by dominant class)
        │
        ▼
  Clay v1.5 encoder  [FROZEN — pretrained on global satellite data]
        │
        ▼
  UNetDecoder        [TRAINED on your labeled tiles]
        │
        ▼
  20-class LULC map  →  GeoTIFF
```

---

## Pipeline

### 1. Config files

| File | Controls |
|------|----------|
| `configs/source_to_target.yaml` | STAC query, AOI, bands, quality filters, paths |
| `configs/terratorch_segmentation_clay.yaml` | Model, decoder, loss, optimizer, callbacks |
| `configs/metadata.yaml` | Per-band normalization stats by platform |
| `configs/model_profiles.yaml` | Runtime profiles (mac / gpu / cloud) |

### 2. Commands (run in order)

```bash
# Create project directories
clayterractorch setup

# Clip LULC masks to AOI + reproject
clayterractorch prep    --config configs/source_to_target.yaml

# Fetch Landsat tiles from Planetary Computer → .npz cubes
clayterractorch cubes   --config configs/source_to_target.yaml

# Stratified train / val / test split
clayterractorch splits  --config configs/source_to_target.yaml

# Sync band stats (means/stds) → configs/stats/*.txt
clayterractorch stats   --config configs/source_to_target.yaml

# Train Clay + UNetDecoder via TerraTorch Lightning
clayterractorch train   --config configs/terratorch_segmentation_clay.yaml

# Predict LULC on a new scene
clayterractorch predict --config configs/terratorch_segmentation_clay.yaml \
                        --ckpt   outputs/terratorch_clay/checkpoints/best.ckpt \
                        --scene  /path/to/scene.tif \
                        --out    outputs/predictions/lulc.tif
```

---

## Setup

```bash
# Create env
micromamba env create -f environment.yml -y
micromamba activate clayterratorch

# Authenticate Planetary Computer
planetarycomputer configure  # or set PC_SDK_SUBSCRIPTION_KEY env var

# HuggingFace login (for Clay weights)
huggingface-cli login
```

Update `backbone_ckpt` in `configs/terratorch_segmentation_clay.yaml` to point at your local `clay-v1.5.ckpt`.

---

## Project structure

```
ClayTerratorch/
├── clayterractorch.py           ← unified CLI entry point
├── lulc_legend.py               ← 20-class LULC legend + colors
├── configs/
│   ├── source_to_target.yaml    ← data pipeline config
│   ├── terratorch_segmentation_clay.yaml  ← model config (edit this)
│   ├── metadata.yaml            ← band stats by platform
│   ├── model_profiles.yaml      ← runtime profiles
│   └── stats/                   ← auto-generated means/stds txt files
├── scripts/
│   ├── 01_prepare_lulc_masks.py ← clip + reproject LULC GeoTIFFs
│   ├── 02_generate_tiles_from_stac.py ← fetch + save tile cubes
│   ├── 03_make_splits.py        ← stratified train/val/test split
│   ├── 04_compute_stats.py      ← sync band stats from metadata
│   └── list_terratorch_backbones.py  ← utility: list valid backbones
├── data/
│   ├── raw/                     ← raw LULC GeoTIFFs + shapefiles
│   └── processed/
│       ├── lulc_masks/          ← clipped + reprojected masks
│       └── tiles/
│           ├── all/             ← all generated tiles
│           └── {train,val,test}/images/ + masks/
└── outputs/
    ├── terratorch_clay/         ← checkpoints + logs
    └── predictions/             ← output LULC GeoTIFFs
```

---

## 20 LULC Classes

| Code | Class |
|------|-------|
| 0 | No data (ignored) |
| 1 | Built-up |
| 2 | Kharif crops |
| 3 | Rabi crops |
| 4 | Zaid crops |
| 5 | Orchards |
| 6 | Plantations |
| 7 | Evergreen forest |
| 8 | Deciduous forest |
| 9 | Shrubland |
| 10 | Grassland |
| 11 | Scrubland |
| 12 | Barren land |
| 13 | Fallow land |
| 14 | Marshy / Swamp |
| 15 | Inland water |
| 16 | River / Stream |
| 17 | Snow / Ice |
| 18 | Cloud / Shadow |
| 19 | Other |

---

## Key config knobs

**Freeze vs fine-tune backbone** (`terratorch_segmentation_clay.yaml`):
```yaml
freeze_backbone: true   # fast, less GPU — start here
freeze_backbone: false  # end-to-end fine-tune — after frozen head converges
```

**Adjust training** (`terratorch_segmentation_clay.yaml`):
```yaml
trainer:
  max_epochs: 60
  precision: 16-mixed

model:
  init_args:
    loss:
      ce: 0.7
      dice: 0.3
```

**Change area / dates** (`source_to_target.yaml`):
```yaml
paths:
  aoi_shapefile: data/raw/uk_shape_files/uttarakhand_Boundary.shp
source:
  date_ranges:
    "2015": "2014-01-01/2016-12-31"
```
