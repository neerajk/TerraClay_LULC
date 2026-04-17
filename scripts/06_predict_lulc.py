#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import yaml
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import rasterio
from rasterio.enums import Resampling

# Try to import required libraries
try:
    from claymodel.module import ClayMAEModule
    CLAY_AVAILABLE = True
except ImportError:
    CLAY_AVAILABLE = False

try:
    import terratorch
    from terratorch.tasks import SemanticSegmentationTask
    TERRATORCH_AVAILABLE = True
except ImportError:
    TERRATORCH_AVAILABLE = False


def _resolve(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (root / path)


class PredictionCubeDataset(Dataset):
    """Dataset for loading cubes for prediction."""
    def __init__(self, npz_files):
        self.files = npz_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        with np.load(path) as data:
            pixels = data['pixels'].astype(np.float32)  # (6, H, W)
            lat_norm = data['lat_norm']                 # (2,)
            lon_norm = data['lon_norm']                 # (2,)
            week_norm = data['week_norm']               # (2,)
            hour_norm = data['hour_norm']               # (2,)

        return {
            "pixels": pixels,
            "lat_norm": lat_norm,
            "lon_norm": lon_norm,
            "week_norm": week_norm,
            "hour_norm": hour_norm,
            "filename": path.name
        }


def create_clay_batch(batch_dict, wavelengths, gsd, device):
    """Formats the batched tensors exactly as CLAY expects."""
    batch = {}
    batch["pixels"] = torch.tensor(batch_dict["pixels"], dtype=torch.float32).to(device)
    batch["time"] = torch.cat([
        torch.tensor(batch_dict["week_norm"], dtype=torch.float32),
        torch.tensor(batch_dict["hour_norm"], dtype=torch.float32)
    ]).unsqueeze(0).repeat(len(batch_dict["pixels"]), 1).to(device)
    batch["latlon"] = torch.cat([
        torch.tensor(batch_dict["lat_norm"], dtype=torch.float32),
        torch.tensor(batch_dict["lon_norm"], dtype=torch.float32)
    ]).unsqueeze(0).repeat(len(batch_dict["pixels"]), 1).to(device)

    # Metadata scalars
    batch["waves"] = torch.tensor(wavelengths, dtype=torch.float32).to(device)
    batch["gsd"] = torch.tensor(gsd, dtype=torch.float32).to(device)
    return batch


def load_trained_decoder(checkpoint_path: Path, device: torch.device, num_classes: int = 20):
    """Load the trained decoder head for LULC prediction."""
    if not TERRATORCH_AVAILABLE:
        raise ImportError("Terratorch is required for prediction")

    # Load the Terratorch checkpoint which should contain the trained segmentation head
    # This assumes the checkpoint is from a SemanticSegmentationTask
    try:
        # Try to load as a segmentation task
        model = SemanticSegmentationTask.load_from_checkpoint(checkpoint_path, map_location=device)
        model.eval()
        model.to(device)
        return model
    except Exception as e:
        print(f"Warning: Could not load as SegmentationTask: {e}")
        print("Falling back to loading raw checkpoint and extracting decoder...")
        # Fallback: load raw checkpoint and reconstruct
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # This would need custom implementation based on how the decoder was saved
        raise NotImplementedError("Decoder extraction from raw checkpoint not implemented")


def predict_scene(args):
    """Main prediction function."""
    # Set device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"🖥️  Using device: {device}")

    # Load metadata for normalization
    metadata_path = _resolve(Path(__file__).resolve().parents[1], args.metadata)
    with open(metadata_path, 'r') as f:
        clay_meta = yaml.safe_load(f)

    platform = args.platform
    GSD = clay_meta[platform]['gsd']
    WAVELENGTHS = [clay_meta[platform]['bands']['wavelength'][b] for b in
                  args.band_indices.split(',')]
    MEANS = [clay_meta[platform]['bands']['mean'][b] for b in
            args.band_indices.split(',')]
    STDS = [clay_meta[platform]['bands']['std'][b] for b in
           args.band_indices.split(',')]

    # Normalize function
    def normalize_pixels(pixels_np):
        means = np.array(MEANS, dtype=np.float32).reshape(-1, 1, 1)
        stds = np.array(STDS, dtype=np.float32).reshape(-1, 1, 1)
        return (pixels_np - means) / stds

    # Load encoder (CLAY or Terratorch backbone)
    print(f"📦 Loading encoder from: {args.encoder_checkpoint}")
    if args.encoder_type == "clay" and CLAY_AVAILABLE:
        encoder_module = ClayMAEModule.load_from_checkpoint(
            checkpoint_path=args.encoder_checkpoint,
            model_size="large",
            metadata_path=str(metadata_path),
            dolls=[16, 32, 64, 128, 256, 768, 1024],
            doll_weights=[1, 1, 1, 1, 1, 1, 1],
            mask_ration=0.0,
            shuffle=False,
        )
        encoder_module.eval()
        encoder_module.to(device)
        encode_fn = lambda batch: encoder_module.model.encoder(batch)[0]
        print("✅ CLAY Encoder loaded!")
    elif args.encoder_type == "terratorch" and TERRATORCH_AVAILABLE:
        # For Terratorch, we'd need to extract the backbone from a segmentation model
        # For now, placeholder
        raise NotImplementedError("Terratorch encoder extraction not implemented")
    else:
        raise ValueError(f"Unsupported encoder type: {args.encoder_type}")

    # Load trained decoder head
    print(f"📦 Loading trained decoder from: {args.decoder_checkpoint}")
    decoder_model = load_trained_decoder(args.decoder_checkpoint, device, args.num_classes)
    print("✅ Decoder loaded!")

    # Process input scene
    scene_path = _resolve(Path(__file__).resolve().parents[1], args.scene_tif)
    print(f"📖 Reading input scene: {scene_path}")

    with rasterio.open(scene_path) as src:
        # Read the bands we need
        band_indices = [int(b) for b in args.band_indices.split(',')]
        image = src.read(band_indices)  # (bands, height, width)
        profile = src.profile
        transform = src.transform
        crs = src.crs
        width, height = src.width, src.height

        # Normalize image
        image_norm = np.zeros_like(image, dtype=np.float32)
        for i in range(image.shape[0]):
            image_norm[i] = normalize_pixels(image[i:i+1])[0]

    # Create sliding window prediction
    tile_size = args.tile_size
    stride = args.stride
    batch_size = args.batch_size

    print(f"🔍 Predicting with tile_size={tile_size}, stride={stride}, batch_size={batch_size}")

    # Initialize output arrays
    output_profile = profile.copy()
    output_profile.update({
        "count": args.num_classes,
        "dtype": "float32",
        "compress": "lzw"
    })

    prediction_sum = np.zeros((args.num_classes, height, width), dtype=np.float32)
    prediction_count = np.zeros((height, width), dtype=np.int32)

    # Generate tile coordinates
    y_starts = list(range(0, max(0, height - tile_size + 1), stride))
    x_starts = list(range(0, max(0, width - tile_size + 1), stride))

    # Process tiles in batches
    batch_tiles = []
    batch_coords = []

    for y_start in tqdm(y_starts, desc="Processing tile rows"):
        for x_start in x_starts:
            # Extract tile
            y_end = min(y_start + tile_size, height)
            x_end = min(x_start + tile_size, width)

            if y_end - y_start < tile_size or x_end - x_start < tile_size:
                # Skip incomplete tiles at edges (or handle padding)
                continue

            tile_pixels = image_norm[:, y_start:y_end, x_start:x_end]  # (bands, tile_size, tile_size)

            # For simplicity, we'll use center point lat/lon/time
            # In production, you might want to compute per-pixel or interpolate
            center_y, center_x = y_start + tile_size//2, x_start + tile_size//2

            # Approximate lat/lon from pixel coordinates (simplified)
            lat_val = np.array([float(center_y) / height * 2 - 1])  # Normalized to [-1, 1]
            lon_val = np.array([float(center_x) / width * 2 - 1])
            week_val = np.array([0.0])  # Placeholder - would extract from scene metadata
            hour_val = np.array([0.0])  # Placeholder - would extract from scene metadata

            batch_tiles.append(tile_pixels)
            batch_coords.append({
                "lat_norm": lat_val,
                "lon_norm": lon_val,
                "week_norm": week_val,
                "hour_norm": hour_val,
                "y_start": y_start,
                "x_start": x_start,
                "y_end": y_end,
                "x_end": x_end
            })

            # Process batch when full
            if len(batch_tiles) >= batch_size:
                # Process batch
                batch_array = np.stack(batch_tiles)  # (batch_size, bands, tile_size, tile_size)

                # Normalize and create model batch
                normalized_batch = np.stack([
                    normalize_pixels(batch_array[i])
                    for i in range(len(batch_array))
                ])

                model_batch = {
                    "pixels": normalized_batch,
                    "lat_norm": np.stack([coord["lat_norm"] for coord in batch_coords]),
                    "lon_norm": np.stack([coord["lon_norm"] for coord in batch_coords]),
                    "week_norm": np.stack([coord["week_norm"] for coord in batch_coords]),
                    "hour_norm": np.stack([coord["hour_norm"] for coord in batch_coords])
                }

                clay_batch = create_clay_batch(model_batch, WAVELENGTHS, GSD, device)

                # Run inference
                with torch.no_grad():
                    # Get embeddings
                    embeddings = encode_fn(clay_batch)  # (batch_size, 257, 1024)

                    # Predict with decoder
                    # This assumes the decoder takes embeddings and returns class logits
                    # We need to reshape embeddings to match expected input format
                    # For CLAY: (257, 1024) -> remove CLS token -> reshape to (16, 16, 1024) -> (1024, 16, 16)
                    patch_embeddings = embeddings[:, 1:, :]  # Remove CLS token
                    # Reshape to (batch, 1024, 16, 16)
                    patch_embeddings = patch_embeddings.reshape(len(batch_array), 1024, 16, 16)

                    # Predict
                    outputs = decoder_model(patch_embeddings)  # This would depend on model structure
                    # For now, assume it returns logits directly
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    elif isinstance(outputs, tuple) and len(outputs) == 2:
                        logits, _ = outputs  # Assuming (fine_logits, coarse_logits)
                    else:
                        logits = outputs  # Assume direct logits output

                    # Convert to probabilities
                    probs = torch.softmax(logits, dim=1)  # (batch_size, num_classes, H, W)
                    probs_np = probs.cpu().numpy()

                # Accumulate predictions
                for i, coord in enumerate(batch_coords):
                    y_start, x_start, y_end, x_end = coord["y_start"], coord["x_start"], coord["y_end"], coord["x_end"]
                    # Simple accumulation (in production, you'd want to handle overlap/blending)
                    prediction_sum[:, y_start:y_end, x_start:x_end] += probs_np[i]
                    prediction_count[y_start:y_end, x_start:x_end] += 1

                # Reset batch
                batch_tiles = []
                batch_coords = []

    # Process remaining tiles
    if batch_tiles:
        # Similar processing as above for final batch
        pass  # Implementation would mirror the batch processing above

    # Average predictions
    mask = prediction_count > 0
    prediction_avg = np.zeros_like(prediction_sum)
    prediction_avg[mask] = prediction_sum[mask] / prediction_count[mask, np.newaxis, :]

    # Save prediction
    output_path = _resolve(Path(__file__).resolve().parents[1], args.out_tif)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_path, "w", **output_profile) as dst:
        # Save each class as a band (for multi-class softmax output)
        for c in range(args.num_classes):
            dst.write(prediction_avg[c], c + 1)

    print(f"✅ Prediction saved to: {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Predict LULC using trained model weights.")
    p.add_argument("--scene-tif", type=str, required=True, help="Input scene TIFF file")
    p.add_argument("--out-tif", type=str, required=True, help="Output LULC probability TIFF")
    p.add_argument("--encoder-checkpoint", type=str, required=True,
                   help="Path to encoder checkpoint (CLAY or Terratorch)")
    p.add_argument("--encoder-type", choices=["clay", "terratorch"], default="clay",
                   help="Type of encoder model")
    p.add_argument("--decoder-checkpoint", type=str, required=True,
                   help="Path to trained decoder/head checkpoint")
    p.add_argument("--metadata", type=str, default="configs/metadata.yaml",
                   help="Path to metadata yaml")
    p.add_argument("--platform", type=str, default="landsat-c2-l2",
                   help="Platform for normalization (must match metadata)")
    p.add_argument("--band-indices", type=str, default="1,2,3,4,5,6",
                   help="Comma-separated list of band indices (1-based)")
    p.add_argument("--tile-size", type=int, default=256, help="Tile size for prediction")
    p.add_argument("--stride", type=int, default=128, help="Stride for sliding window")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size for prediction")
    p.add_argument("--num-classes", type=int, default=20, help="Number of LULC classes")
    p.add_argument("--device", default="auto", help="Device to use (auto, cpu, cuda, mps)")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    exit(predict_scene(args))