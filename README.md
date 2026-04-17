# ClayTerratorch

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TerraTorch](https://img.shields.io/badge/TerraTorch-0.99+-orange.svg)](https://github.com/terrastackai/terratorch)

**End-to-End LULC Prediction Pipeline** using CLAY/Terratorch embeddings with unified CLI interface.

## 🚀 Overview

ClayTerratorch replaces custom training/inference glue with a streamlined, configurable pipeline that:

- **Curates data** from Planetary Computer STAC using spatial/temporal/query filters
- **Generates analysis-ready cubes** with satellite bands, LULC masks, and geospatial/temporal metadata
- **Extracts powerful embeddings** from foundation models (CLAY/Terratorch)
- **Trains lightweight segmentation heads** on embeddings for efficient fine-tuning
- **Predicts LULC** on new scenes using sliding window inference
- **Everything controllable via unified CLI** - no more scattered scripts

## 📋 Table of Contents

- [Features](#-features)
- [System Requirements](#-system-requirements)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Pipeline Stages](#-pipeline-stages)
- [Configuration](#-configuration)
- [Examples](#-examples)
- [Cloud Deployment](#-cloud-deployment)
- [License](#-license)

## ✨ Features

- **Unified CLI**: Single entry point with `gen-cubes`, `extract-emb`, `train`, `predict` subcommands
- **STAC-First**: Native integration with Planetary Computer for search-ready data curation
- **Embedding-Powered**: Leverages CLAY/Terratorch encoders for rich feature representations
- **Metadata Aware**: Preserves lat/lon/time/LULC throughout pipeline for advanced analysis
- **Cloud Optimized**: Batch processing, memory efficient, suitable for serverless/containerized deployment
- **Reproducible**: Fixed seeds, deterministic operations, version-controlled configs
- **Flexible**: Works with CLAY, Terratorch backbones, and custom checkpoints
- **Production Ready**: Comprehensive error handling, logging, and checkpointing

## 🔧 System Requirements

- Python 3.8+
- GPU recommended (CUDA 11.x) for training/inference, CPU fallback supported
- ~10GB disk space for processing (scales with AOI size)
- Dependencies listed in `environment.yml` or `requirements.txt`

## 📦 Installation

### Option 1: Micromamba (Recommended for Cloud)

```bash
# Clone repository
git clone https://github.com/yourusername/ClayTerratorch.git
cd ClayTerratorch

# Create environment from file
micromamba create -f environment.yml
micromamba activate clay_terratorch

# Verify installation
python -c "import torch, rasterio, terratorch; print('All core imports OK')"
```

### Option 2: Conda

```bash
conda env create -f environment.yml
conda activate clay_terratorch
```

### Option 3: Pip (if base env already has CUDA/torch)

```bash
pip install -r requirements.txt
# Plus: micromamba install -c conda-forge pytorch-lightning torchmetrics
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
clayterractorch setup
```
Creates necessary directory structure and validates configuration.

### 2. Generate Data Cubes

```bash
# Edit configs/source_to_target.yaml for your AOI, date range, filters
clayterractorch gen-cubes --config configs/source_to_target.yaml
```
Output: `data/processed/tiles/all/cubes/*.npz`

### 3. Extract Embeddings

```bash
clayterractorch extract-emb \
  --config configs/source_to_target.yaml \
  --checkpoint /path/to/clay-v1.5.ckpt \
  --model-type clay \
  --batch-size 32
```
Output: `data/embeddings/YEAR/emb_*.npz`

### 4. Train Model

```bash
# Option A: Embedding-based training (recommended for efficiency)
clayterractorch train \
  --embedding-based \
  --embedding-dir data/embeddings \
  --output-dir outputs/training \
  --max-epochs 50

# Option B: Standard TerraTorch training (uses raw pixels)
clayterractorch train \
  --config configs/terratorch_segmentation.yaml
```
Output: `outputs/training/*.ckpt`

### 5. Predict on New Scene

```bash
clayterractorch predict \
  --scene-tif /path/to/input_scene.tif \
  --out-tif /path/to/output_lulc.tif \
  --encoder-checkpoint /path/to/clay-v1.5.ckpt \
  --decoder-checkpoint /path/to/best_training_model.ckpt \
  --tile-size 256 \
  --stride 128 \
  --batch-size 16
```
Output: Multi-class probability GeoTIFF (20 bands for 20 LULC classes)

## 🔄 Pipeline Stages

See [FLOWCHART.md](FLOWCHART.md) for detailed visualization.

### Stage 1: Cube Generation
**Input**: STAC query parameters (AOI, date, collections, filters)  
**Process**: Search STAC → Filter scenes → Load LULC masks → Save cubes  
**Output**: `.npz` files containing:
- `pixels`: (6, H, W) - Surface reflectance (B, G, R, NIR, SWIR1, SWIR2)
- `mask`: (H, W) - LULC ground truth (0=background, 1-20=classes)
- `lat_norm`, `lon_norm`: (2,) - Normalized coordinates [-1, 1]
- `week_norm`, `hour_norm`: (2,) - Normalized temporal cycles [-1, 1]

### Stage 2: Embedding Extraction
**Input**: Data cubes from Stage 1  
**Process**: Batch normalize → CLAY/Terratorch encoder → Save features  
**Output**: `.npz` files containing:
- `embeddings`: (257, 1024) - [CLS token + 16x16 patch embeddings]
- `mask`: (256, 256) - LULC labels for supervised learning

### Stage 3: Model Training
**Input**: Embeddings + masks from Stage 2  
**Process**: Train/val split → Forward pass → Loss → Backprop → Checkpoint  
**Output**: 
- Best model: `outputs/training/best-{epoch}-{val_mIoU:.4f}.ckpt`
- Last epoch: `outputs/training/last.ckpt`
- Training logs: TensorBoard compatible

### Stage 4: Prediction
**Input**: New scene GeoTIFF + trained models  
**Process**: Sliding window → Encode → Decode → Softmax → Accumulate → Average  
**Output**: GeoTIFF with:
- Band 1: Class 0 probability (background)
- Band 2: Class 1 probability 
- ...
- Band 20: Class 19 probability
- Same georeference/projection as input

## ⚙️ Configuration

### Main Config: `configs/source_to_target.yaml`
```yaml
source:
  stac_api: "https://planetarycomputer.microsoft.com/api/stac/v1"
  collection: "landsat-c2-l2"
  date_ranges:
    "2020": "2020-01-01/2020-12-31"
  allowed_months: [4, 5, 6, 7, 8, 9, 10]  # Growing season
  max_catalog_cloud_cover: 10.0
  max_alternates: 6
  query:
    eo:cloud_cover:
      lte: 10.0

bands:
  order: [red, green, blue, nir08, swir16, swir22]

quality:
  min_lulc_coverage_pct: 30.0   # Minimum valid LULC pixels in tile
  max_nodata_pct: 40.0
  max_black_pct: 50.0
  max_blue_mean: 15000.0

tiling:
  tile_size: 256
  stride: 128  # 50% overlap for seamless prediction

paths:
  aoi_shapefile: "data/raw/aoi.shp"           # Optional AOI constraint
  source_lulc:
    "2020": "data/raw/lulc_2020.tif"          # LULC reference data
  prepared_masks_dir: "data/processed/lulc_masks"
  tiles_all_cubes_dir: "data/processed/tiles/all/cubes"
  tiles_metadata_csv: "data/processed/tiles/all/tiles_metadata.csv"
```

### Model Profiles: `configs/model_profiles.yaml`
Predefined configurations for different backbones:
- `clay_lulc_best`: Legacy CLAY + CNN decoder (from clay_LULC)
- `terratorch_prithvi`: Terratorch with Prithvi backbone
- `terratorch_dofa`: Terratorch with DOFA backbone  
- Add your own by following the template

## 💡 Examples

### Extract DOFA Embeddings
```bash
clayterractorch extract-emb \
  --config configs/source_to_target.yaml \
  --checkpoint /path/to/terratorch_dofa.ckpt \
  --model-type clay \
  --batch-size 64
```

### Train with Custom Learning Rate
```bash
clayterractorch train \
  --embedding-based \
  --embedding-dir data/embeddings \
  --lr 0.0005 \
  --weight-decay 0.001 \
  --max-epochs 100
```

### Predict with Overlap Blending
```bash
clayterractorch predict \
  --scene-tif landsat_scene.tif \
  --out-tif prediction.tif \
  --encoder-checkpoint clay-v1.5.ckpt \
  --decoder-checkpoint best_model.ckpt \
  --tile-size 512 \
  --stride 384  # 25% overlap
  --batch-size 8
```

## ☁️ Cloud Deployment

### AWS Lambda / Google Cloud Run / Azure Container Instances
1. Containerize with Docker:
   ```dockerfile
   FROM continuumio/micromamba
   COPY environment.yml /
   RUN micromamba create -f /environment.yml && micromamba clean --all
   ENV PATH /opt/conda/envs/clay_terratorch/bin:$PATH
   COPY . /app
   WORKDIR /app
   ENTRYPOINT ["clayterractorch"]
   ```
2. Mount volumes for:
   - Input: AOI shapefiles, LULC references, scenes to predict
   - Output: Predictions, logs, temporary processing
   - Config: Your customized YAML files
3. Set environment variables:
   - `OMP_NUM_THREADS=1` (prevent oversubscription)
   - `NUMBA_NUM_THREADS=1`

### HPC / SLURM Clusters
Use the CLI directly in job scripts:
```bash
#!/bin/bash
#SBATCH --job-name=lulc_predict
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

micromamba activate clay_terratorch
clayterractorch predict --scene-tif $SCENE --out-tif $OUT ...
```

## 📁 Directory Structure

```
ClayTerratorch/
├── clayterractorch.py          # Unified CLI entry point
├── FLOWCHART.md                # Pipeline visualization
├── IMPLEMENTATION_SUMMARY.md   # Technical implementation details
├── environment.yml             # Micromamba/conda environment
├── requirements.txt            # Pip requirements
├── configs/                    # YAML configuration files
│   ├── source_to_target.yaml   # STAC/curation/settings
│   ├── metadata.yaml           # Sensor normalization stats
│   ├── terratorch_segmentation.yaml  # Standard training
│   └── model_profiles.yaml     # Pretrained model definitions
├── scripts/                    # Individual pipeline stages
│   ├── 02_generate_tiles_from_stac.py   # Stage 1
│   ├── 05_extract_embeddings.py         # Stage 2
│   ├── 13_train_embedding_head.py       # Stage 3 (embedding-based)
│   ├── 06_predict_lulc.py               # Stage 4
│   └── 10_train_terratorch.sh           # Stage 3 (standard)
├── data/                       # Data directory (created by setup)
│   ├── raw/                    # AOI, LULC references
│   └── processed/              # Intermediate products
├── outputs/                    # Results directory (created by setup)
│   ├── predictions/            # Final LULC predictions
│   ├── embedding_training/     # Model checkpoints
│   └── terratorch_runs/        # Standard training outputs
└── logs/                       # Processing logs
```

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce `--batch-size`
   - Increase `--stride` (less overlap)
   - Use `--device cpu` for debugging
   - Add `export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`

2. **Missing Dependencies**
   - Run `clayterractorch setup` first
   - Check `micromamba list` or `conda list`
   - Install missing: `micromamba install PACKAGE_NAME`

3. **STAC Access Issues**
   - Verify internet connectivity
   - Planetary Computer occasionally has downtime
   - Try reducing `max_alternates` to decrease load

4. **Shape Mismatch Errors**
   - Ensure input scene bands match config `--band-indices`
   - Check that LULC mask classes match `num_classes` (default 20)
   - Verify tile size compatibility with model expectations

## 📚 References

- [CLAY Foundation Model](https://github.com/ironjr/clay)
- [TerraTorch Geospatial ML Library](https://github.com/terrastackai/terratorch)
- [Planetary Computer STAC API](https://planetarycomputer.microsoft.com/)
- [International Land Cover Classification System](https://www.fao.org/land-water/land/land-governance/land-resources-planning-and-management/lccs/en/)

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built upon the [clay_LULC](https://github.com/ironjr/clay_LULC) foundation
- Uses Planetary Computer for free access to satellite imagery
- Leverages TerraTorch for geospatial deep learning utilities
- Inspired by modern MLOps practices for reproducible ML pipelines

---

**Ready for production use. Contributions welcome!**  
For questions or issues, please open a GitHub Issue.