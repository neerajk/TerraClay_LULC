# ClayTerratorch End-to-End Pipeline Flowchart

## Overview
```
INPUT (STAC Query Parameters)
          ↓
[1] Generate Cubes from STAC
          ↓
  Image Tiles (6 bands) + LULC Mask
          ↓
[2] Extract Embeddings (CLAY/Terratorch Encoder)
          ↓
  Embeddings (256x1024) + LULC Mask
          ↓
[3] Train Segmentation Head
          ↓
  Trained Decoder Model (.ckpt)
          ↓
[4] Predict LULC on New Scene
          ↓
OUTPUT: LULC Probability Map (GeoTIFF)
```

## Detailed Flow

### Phase 1: Data Preparation
```
STAC Query (AOI, date range, collections, filters)
          ↓
02_generate_tiles_from_stac.py
          ↓
  ├─ Search Planetary Computer STAC
  ├─ Filter by quality (cloud cover, black pixels, etc.)
  ├─ Load LULC mask tiles
  └─ Save as .npz cubes:
       - pixels: (6, H, W) - surface reflectance
       - mask: (H, W) - LULC labels (20 classes)
       - lat_norm: (2,) - normalized latitude
       - lon_norm: (2,) - normalized longitude  
       - week_norm: (2,) - normalized week of year
       - hour_norm: (2,) - normalized hour of day
          ↓
data/processed/tiles/all/cubes/*.npz
```

### Phase 2: Embedding Extraction
```
Cubes (.npz files)
          ↓
05_extract_embeddings.py
          ↓
  ├─ Load cubes in batches
  ├─ Normalize pixels using metadata statistics
  ├─ Prepare time/lat/lon tensors for CLAY/Terratorch
  ├─ Run encoder forward pass (no gradients)
  └─ Save embeddings:
       - embeddings: (257, 1024) [CLS + 16x16 patches]
       - mask: (256, 256) - LULC labels
          ↓
data/embeddings/*/year/*.npz
```

### Phase 3: Model Training
```
Embeddings (.npz files)
          ↓
13_train_embedding_head.py (or 10_train_terratorch.sh)
          ↓
  ├─ Create train/val split
  ├─ Dataset: EmbeddingDataset (loads embeddings + masks)
  ├─ Model: SimpleUNetHead or Terratorch SegmentationHead
  ├─ Loss: CrossEntropyLoss (ignore index=0 for background)
  ├─ Optimizer: AdamW with LR scheduler
  ├─ Metrics: mIoU, per-class IoU
  └─ Save best model checkpoints
          ↓
outputs/embedding_training/*.ckpt
```

### Phase 4: Inference/Prediction
```
Input Scene (GeoTIFF)
          ↓
06_predict_lulc.py
          ↓
  ├─ Slide window over scene (tile_size x stride)
  ├─ For each tile:
      │  ├─ Extract and normalize pixels
      │  ├─ Prepare time/lat/lon (center point approximation)
      │  ├─ Run encoder → get embeddings
      │  ├─ Run decoder → get class logits
      │  ├─ Apply softmax → get probabilities
      │  └─ Accumulate predictions (overlap handling)
  ├─ Average overlapping predictions
  └─ Save multi-class probability GeoTIFF
          ↓
outputs/predictions/scene_model.tif
```

## Cloud Server Optimization Notes

1. **Batch Processing**: All GPU operations use batching for efficiency
2. **Memory Management**: 
   - Automatic cleanup with `del` and `torch.cuda.empty_cache()`
   - Pin memory for faster CPU↔GPU transfer
   - Num workers configurable for data loading
3. **Disk Usage**:
   - Intermediate cubes can be deleted after embedding extraction
   - Embeddings are much smaller than raw cubes (~10x compression)
   - Only final model checkpoints and predictions need long-term storage
4. **Reproducibility**:
   - All random seeds fixed
   - Deterministic algorithms where possible
   - Full configuration in YAML files

## CLI Usage Summary

```bash
# 1. Prepare data (run once per AOI/date range)
clayterractorch gen-cubes --config configs/source_to_target.yaml

# 2. Extract embeddings (run once per dataset)
clayterractorch extract-emb \
  --config configs/source_to_target.yaml \
  --checkpoint /path/to/clay-v1.5.ckpt

# 3. Train model (run when new labels available)
clayterractorch train --embedding-based --embedding-dir data/embeddings

# 4. Predict (run for each new scene)
clayterractorch predict \
  --scene-tif /path/to/input.tif \
  --out-tif /path/to/output.tif \
  --encoder-checkpoint /path/to/clay-v1.5.ckpt \
  --decoder-checkpoint /path/to/best_model.ckpt
```