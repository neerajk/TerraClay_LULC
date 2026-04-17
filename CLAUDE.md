# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

Setup environment:
```bash
clayterractorch setup
```

Generate data cubes from STAC:
```bash
clayterractorch gen-cubes --config configs/source_to_target.yaml
```

Extract embeddings (CLAY default):
```bash
clayterractorch extract-emb \
  --config configs/source_to_target.yaml \
  --checkpoint /path/to/clay-v1.5.ckpt \
  --model-type clay \
  --batch-size 32
```

Train segmentation head (embedding-based DEFAULT):
```bash
clayterractorch train \
  --embedding-dir data/embeddings \
  --output-dir outputs/training \
  --max-epochs 50
```

Standard TerraTorch training (alternative):
```bash
clayterractorch train \
  --config configs/terratorch_segmentation.yaml \
  --standard
```

Predict LULC (bilinear upsampling DEFAULT):
```bash
clayterractorch predict \
  --scene-tif /path/to/input_scene.tif \
  --out-tif /path/to/output_lulc.tif \
  --encoder-checkpoint /path/to/clay-v1.5.ckpt \
  --decoder-checkpoint /path/to/best_training_model.ckpt \
  --tile-size 256 \
  --stride 128 \
  --batch-size 16 \
  --prediction-method bilinear
```

Predict with internal Terratorch decoder:
```bash
clayterractorch predict \
  --scene-tif /path/to/input_scene.tif \
  --out-tif /path/to/output_lulc.tif \
  --encoder-checkpoint /path/to/clay-v1.5.ckpt \
  --decoder-checkpoint /path/to/terratorch_decoder.ckpt \
  --encoder-type clay \
  --prediction-method internal
```

## Architecture Overview

ClayTerratorch implements a 4-stage LULC prediction pipeline:

1. **Cube Generation** (02_generate_tiles_from_stac.py): 
   - Searches Planetary Computer STAC for satellite scenes
   - Filters by date, cloud cover, AOI
   - Loads LULC masks and creates analysis-ready cubes (.npz)
   - Output: pixels (6,H,W), mask (H,W), lat/lon/time metadata

2. **Embedding Extraction** (05_extract_embeddings.py):
   - Normalizes cube pixels
   - Passes through CLAY/Terratorch encoder
   - Saves embeddings and masks
   - Output: embeddings (257,1024) [CLS+patches], mask (256,256)

3. **Model Training** (13_train_embedding_head.py DEFAULT or 10_train_terratorch.sh):
   - Embedding-based: Trains UNet head on embeddings → segmentation
   - Standard: Full Terratorch training on raw pixels
   - Output: Model checkpoints, training logs, visualizations

4. **Prediction** (06_predict_lulc.py):
   - Bilinear: Scene → CLAY embeddings → segmentation head → upsample → probabilities
   - Internal: Scene → CLAY embeddings → Terratorch decoder → upsample → probabilities
   - Sliding window with 50% overlap (tile=256, stride=128)
   - Output: 20-band probability GeoTIFF + optional visualizations

Key components:
- Unified CLI (clayterractorch.py) coordinates all stages
- LULC legend (lulc_legend.py) provides 20-class mapping and colors
- Configuration via YAML files in configs/
- Visualization and metadata generation during training/prediction