# ClayTerratorch - Quick Start Guide

## 🚀 5-Minute Setup (Micromamba)

```bash
# 1. Get the code
git clone https://github.com/yourusername/ClayTerratorch.git
cd ClayTerratorch

# 2. Create environment
micromamba create -f environment.yml
micromamba activate clay_terratorch

# 3. Verify installation
./INSTALL_TEST.sh
```

## 📋 Complete Workflow

### Step 1: Configure your AOI
```bash
cp example_config.yaml configs/source_to_target.yaml
# Edit configs/source_to_target.yaml for your:
#   - AOI (optional shapefile)
#   - Date range
#   - LULC reference data
```

### Step 2: Generate Analysis Cubes
```bash
clayterractorch gen-cubes --config configs/source_to_target.yaml
# Output: data/processed/tiles/all/cubes/*.npz
```

### Step 3: Extract Features
```bash
clayterractorch extract-emb \
  --config configs/source_to_target.yaml \
  --checkpoint /path/to/clay-v1.5.ckpt
# Output: data/embeddings/YEAR/emb_*.npz
```

### Step 4: Train Model
```bash
# Option A: Fast embedding-based training (recommended)
clayterractorch train \
  --embedding-based \
  --embedding-dir data/embeddings \
  --output-dir outputs/training

# Option B: Standard TerraTorch training
clayterractorch train \
  --config configs/terratorch_segmentation.yaml
# Output: outputs/training/*.ckpt
```

### Step 5: Predict New Scenes
```bash
clayterractorch predict \
  --scene-tif /path/to/new_scene.tif \
  --out-tif /path/to/prediction.tif \
  --encoder-checkpoint /path/to/clay-v1.5.ckpt \
  --decoder-checkpoint /path/to/best_model.ckpt
# Output: Multi-class probability GeoTIFF
```

## 🔧 Troubleshooting

**Missing dependencies?** Run the full install test:
```bash
micromamba create -f environment.yml
micromamba activate clay_terratorch
./INSTALL_TEST.sh
```

**Out of memory?** Reduce batch size:
```bash
clayterractorch extract-emb --batch-size 16 ...
clayterractorch train --batch-size 8 ...
clayterractorch predict --batch-size 4 ...
```

**Need GPU?** Specify device:
```bash
clayterractorch predict --device cuda ...
```

## 📁 Output Locations

- **Intermediate cubes**: `data/processed/tiles/all/cubes/`
- **Embeddings**: `data/embeddings/YEAR/`
- **Models**: `outputs/training/` or `outputs/embedding_training/`
- **Predictions**: `outputs/predictions/`
- **Logs**: `logs/` (auto-generated)

## 🌐 Cloud Deployment

See [README.md](README.md) for Docker, AWS Lambda, and HPC examples.

## ❓ Help

```bash
clayterractorch --help          # Main help
clayterractorch <subcommand> --help  # Subcommand help
```

## 📚 Documentation

- [README.md](README.md) - Full documentation
- [FLOWCHART.md](FLOWCHART.md) - Pipeline visualization
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Technical details