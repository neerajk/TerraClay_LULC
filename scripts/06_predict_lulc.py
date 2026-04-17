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

# Import LULC legend for Decadal LULC India
try:
    from lulc_legend import LULC_CLASS_MAP, LULC_VALID_CLASS_IDS, class_colors
    LULC_LEGEND_AVAILABLE = True
except ImportError:
    LULC_LEGEND_AVAILABLE = False
    # Fallback definitions
    LULC_CLASS_MAP = {}
    LULC_VALID_CLASS_IDS = tuple(range(20))
    class_colors = ["#000000"] * 20


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


def load_embedding_decoder(checkpoint_path: Path, device: torch.device, num_classes: int = 20):
    """Load the trained decoder head for embedding-based prediction."""
    try:
        # Try to load as a PyTorch Lightning module first
        model = torch.load(checkpoint_path, map_location=device)
        if hasattr(model, 'model') and hasattr(model.model, 'forward'):
            # It's a LightningModule - extract the actual model
            actual_model = model.model
            actual_model.eval()
            actual_model.to(device)
            return actual_model
        elif hasattr(model, 'forward'):
            # It's already a raw nn.Module
            model.eval()
            model.to(device)
            return model
        else:
            # It's a raw state dict, we need to reconstruct the model
            raise NotImplementedError("Raw state dict loading not implemented - use Lightning checkpoints")
    except Exception as e:
        print(f"Error loading embedding decoder: {e}")
        raise


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

    if args.prediction_method == "bilinear":
        # Original embedding-based method with bilinear upsampling
        return _predict_bilinear(
            args, device, metadata_path, clay_meta,
            WAVELENGTHS, GSD, MEANS, STDS, normalize_pixels
        )
    else:
        # Internal Terratorch decoder method
        return _predict_internal(
            args, device, metadata_path, clay_meta,
            WAVELENGTHS, GSD, MEANS, STDS, normalize_pixels
        )


def _predict_bilinear(args, device, metadata_path, clay_meta,
                     WAVELENGTHS, GSD, MEANS, STDS, normalize_pixels):
    """Original prediction method using embedding extraction + bilinear upsampling."""
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

    # Load trained decoder head (embedding-based)
    print(f"📦 Loading trained decoder from: {args.decoder_checkpoint}")
    decoder_model = load_embedding_decoder(args.decoder_checkpoint, device, args.num_classes)
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
                    outputs = decoder_model(patch_embeddings)
                    # Handle different output types
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    elif isinstance(outputs, tuple) and len(outputs) == 2:
                        logits, _ = outputs  # Assuming (fine_logits, coarse_logits)
                    else:
                        logits = outputs  # Assume direct logits output

                    # Convert to probabilities
                    probs = torch.softmax(logits, dim=1)  # (batch_size, num_classes, H, W)
                    probs_np = probs.cpu().numpy()

                # Upsample to tile size using bilinear interpolation
                # Decoder output is (batch, num_classes, 16, 16) -> upsample to (batch, num_classes, tile_size, tile_size)
                probs_tensor = torch.from_numpy(probs_np)
                probs_upsampled = torch.nn.functional.interpolate(
                    probs_tensor, size=(tile_size, tile_size), mode='bilinear', align_corners=False
                )
                probs_np = probs_upsampled.numpy()

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
            patch_embeddings = embeddings[:, 1:, :]  # Remove CLS token
            # Reshape to (batch, 1024, 16, 16)
            patch_embeddings = patch_embeddings.reshape(len(batch_array), 1024, 16, 16)

            # Predict
            outputs = decoder_model(patch_embeddings)
            # Handle different output types
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, tuple) and len(outputs) == 2:
                logits, _ = outputs  # Assuming (fine_logits, coarse_logits)
            else:
                logits = outputs  # Assume direct logits output

            # Convert to probabilities
            probs = torch.softmax(logits, dim=1)  # (batch_size, num_classes, H, W)
            probs_np = probs.cpu().numpy()

            # Upsample to tile size using bilinear interpolation
            probs_tensor = torch.from_numpy(probs_np)
            probs_upsampled = torch.nn.functional.interpolate(
                probs_tensor, size=(tile_size, tile_size), mode='bilinear', align_corners=False
            )
            probs_np = probs_upsampled.numpy()

            # Accumulate predictions
            for i, coord in enumerate(batch_coords):
                y_start, x_start, y_end, x_end = coord["y_start"], coord["x_start"], coord["y_end"], coord["x_end"]
                # Simple accumulation (in production, you'd want to handle overlap/blending)
                prediction_sum[:, y_start:y_end, x_start:x_end] += probs_np[i]
                prediction_count[y_start:y_end, x_start:x_end] += 1

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

    # Save additional metadata and visualization if LULC legend is available
    if LULC_LEGEND_AVAILABLE:
        try:
            # Create class prediction map (argmax)
            class_prediction = np.argmax(prediction_avg, axis=0)  # (H, W)

            # Save class prediction as separate file
            class_output_path = output_path.with_suffix('.class.tif')
            class_profile = output_profile.copy()
            class_profile.update({
                "count": 1,
                "dtype": "uint8",
                "compress": "lzw"
            })

            with rasterio.open(class_output_path, "w", **class_profile) as dst:
                dst.write(class_prediction.astype(np.uint8), 1)

            # Create visualization overlay
            try:
                import matplotlib.pyplot as plt
                import matplotlib.patches as mpatches

                # Create color map from class_colors
                from matplotlib.colors import ListedColormap
                cmap = ListedColormap(class_colors)

                # Create figure
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

                # Show class prediction
                im1 = ax1.imshow(class_prediction, cmap=cmap, vmin=0, vmax=19)
                ax1.set_title('LULC Prediction (Decadal LULC India Classes)')

                # Create legend
                legend_elements = []
                for class_id in LULC_VALID_CLASS_IDS:
                    if class_id in LULC_CLASS_MAP:
                        level_i, level_ii = LULC_CLASS_MAP[class_id]
                        legend_elements.append(
                            mpatches.Patch(color=class_colors[class_id],
                                         label=f'{class_id}: {level_ii}')
                        )

                ax1.legend(handles=legend_elements, loc='center left',
                          bbox_to_anchor=(1, 0.5), fontsize='small')

                # Show entropy (uncertainty)
                # Entropy = -sum(p * log(p)) for each pixel
                epsilon = 1e-8
                probs_clipped = np.clip(prediction_avg, epsilon, 1.0)
                entropy = -np.sum(probs_clipped * np.log(probs_clipped), axis=0)
                max_entropy = np.log(args.num_classes)  # Maximum possible entropy
                normalized_entropy = entropy / max_entropy

                im2 = ax2.imshow(normalized_entropy, cmap='hot', vmin=0, vmax=1)
                ax2.set_title('Prediction Uncertainty (Entropy)')
                plt.colorbar(im2, ax=ax2)

                plt.tight_layout()

                # Save visualization
                viz_path = output_path.with_suffix('.viz.png')
                plt.savefig(viz_path, dpi=150, bbox_inches='tight')
                plt.close()

            except Exception as e:
                print(f"Warning: Could not create visualization: {e}")

        except Exception as e:
            print(f"Warning: Could not save additional metadata: {e}")

    print(f"✅ Prediction saved to: {output_path}")
    if LULC_LEGEND_AVAILABLE:
        print(f"✅ Class prediction saved to: {class_output_path}")
        print(f"✅ Visualization saved to: {viz_path}")
    return 0


def _predict_internal(args, device, metadata_path, clay_meta,
                     WAVELENGTHS, GSD, MEANS, STDS, normalize_pixels):
    """Internal Terratorch decoder method - uses Terratorch's built-in prediction."""
    if not TERRATORCH_AVAILABLE:
        raise ImportError("Terratorch is required for internal prediction method")

    print(f"📦 Loading Terratorch model from: {args.decoder_checkpoint}")

    # Load the full Terratorch segmentation model
    try:
        model = SemanticSegmentationTask.load_from_checkpoint(
            checkpoint_path=args.decoder_checkpoint,
            map_location=device
        )
        model.eval()
        model.to(device)
        print("✅ Terratorch model loaded!")
    except Exception as e:
        raise RuntimeError(f"Failed to load Terratorch model: {e}")

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

        # Update profile for output
        output_profile = profile.copy()
        output_profile.update({
            "count": args.num_classes,
            "dtype": "float32",
            "compress": "lzw"
        })

        # Normalize image according to metadata
        image_norm = np.zeros_like(image, dtype=np.float32)
        for i in range(image.shape[0]):
            image_norm[i] = normalize_pixels(image[i:i+1])[0]

        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image_norm).unsqueeze(0).to(device)  # (1, bands, height, width)

        # Run inference with sliding window
        print(f"🔍 Running internal Terratorch prediction with tile_size={args.tile_size}, stride={args.stride}")

        # Initialize output arrays
        prediction_sum = np.zeros((args.num_classes, height, width), dtype=np.float32)
        prediction_count = np.zeros((height, width), dtype=np.int32)

        # Generate tile coordinates
        y_starts = list(range(0, max(0, height - args.tile_size + 1), args.stride))
        x_starts = list(range(0, max(0, width - args.tile_size + 1), args.stride))

        # Process tiles in batches
        batch_images = []
        batch_coords = []

        for y_start in tqdm(y_starts, desc="Processing tile rows"):
            for x_start in x_starts:
                # Extract tile
                y_end = min(y_start + args.tile_size, height)
                x_end = min(x_start + args.tile_size, width)

                if y_end - y_start < args.tile_size or x_end - x_start < args.tile_size:
                    # Skip incomplete tiles at edges
                    continue

                tile_pixels = image_norm[:, y_start:y_end, x_start:x_end]  # (bands, tile_size, tile_size)

                # For simplicity, use center point for metadata (in production, use per-pixel)
                center_y, center_x = y_start + args.tile_size//2, x_start + args.tile_size//2
                lat_val = np.array([float(center_y) / height * 2 - 1])  # Normalized to [-1, 1]
                lon_val = np.array([float(center_x) / width * 2 - 1])
                week_val = np.array([0.0])  # Placeholder
                hour_val = np.array([0.0])  # Placeholder

                batch_images.append(tile_pixels)
                batch_coords.append({
                    "y_start": y_start,
                    "x_start": x_start,
                    "y_end": y_end,
                    "x_end": x_end
                })

                # Process batch when full
                if len(batch_images) >= args.batch_size:
                    # Process batch
                    batch_array = np.stack(batch_images)  # (batch_size, bands, tile_size, tile_size)
                    batch_tensor = torch.from_numpy(batch_array).to(device)  # (batch_size, bands, tile_size, tile_size)

                    # Run inference
                    with torch.no_grad():
                        # Create batch dict for Terratorch model
                        batch_dict = {
                            "image": batch_tensor,
                            # Add metadata if the model expects it
                        }

                        # Run the model
                        outputs = model(batch_dict)

                        # Extract logits/probabilities from outputs
                        if isinstance(outputs, dict):
                            if "logits" in outputs:
                                logits = outputs["logits"]
                            elif "prediction" in outputs:
                                logits = outputs["prediction"]
                            else:
                                # Assume first value is logits
                                logits = list(outputs.values())[0]
                        elif hasattr(outputs, 'logits'):
                            logits = outputs.logits
                        elif isinstance(outputs, torch.Tensor):
                            logits = outputs
                        else:
                            # Try to handle as tuple
                            if isinstance(outputs, tuple) and len(outputs) > 0:
                                logits = outputs[0]
                            else:
                                raise ValueError(f"Unexpected model output format: {type(outputs)}")

                        # Convert to probabilities
                        probs = torch.softmax(logits, dim=1)  # (batch_size, num_classes, H, W)
                        probs_np = probs.cpu().numpy()

                    # Accumulate predictions
                    for i, coord in enumerate(batch_coords):
                        y_start, x_start, y_end, x_end = coord["y_start"], coord["x_start"], coord["y_end"], coord["x_end"]
                        prediction_sum[:, y_start:y_end, x_start:x_end] += probs_np[i]
                        prediction_count[y_start:y_end, x_start:x_end] += 1

                    # Reset batch
                    batch_images = []
                    batch_coords = []

        # Process remaining tiles
        if batch_images:
            batch_array = np.stack(batch_images)  # (batch_size, bands, tile_size, tile_size)
            batch_tensor = torch.from_numpy(batch_array).to(device)

            with torch.no_grad():
                batch_dict = {
                    "image": batch_tensor,
                }

                outputs = model(batch_dict)

                # Extract logits/probabilities from outputs
                if isinstance(outputs, dict):
                    if "logits" in outputs:
                        logits = outputs["logits"]
                    elif "prediction" in outputs:
                        logits = outputs["prediction"]
                    else:
                        logits = list(outputs.values())[0]
                elif hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif isinstance(outputs, torch.Tensor):
                    logits = outputs
                else:
                    if isinstance(outputs, tuple) and len(outputs) > 0:
                        logits = outputs[0]
                    else:
                        raise ValueError(f"Unexpected model output format: {type(outputs)}")

                probs = torch.softmax(logits, dim=1)
                probs_np = probs.cpu().numpy()

                # Accumulate predictions
                for i, coord in enumerate(batch_coords):
                    y_start, x_start, y_end, x_end = coord["y_start"], coord["x_start"], coord["y_end"], coord["x_end"]
                    prediction_sum[:, y_start:y_end, x_start:x_end] += probs_np[i]
                    prediction_count[y_start:y_end, x_start:x_end] += 1

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
    p.add_argument("--prediction-method", choices=["bilinear", "internal"], default="bilinear",
                   help="Prediction method: bilinear (upsample embeddings) or internal (use Terratorch decoder)")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    exit(predict_scene(args))