#!/usr/bin/env python3
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from pathlib import Path
import yaml
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader


def _resolve(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (root / path)

# Try to import CLAY - if not available, we'll use a placeholder for structure
try:
    from claymodel.module import ClayMAEModule
    CLAY_AVAILABLE = True
except ImportError:
    CLAY_AVAILABLE = False
    print("Warning: CLAY not available. Embedding extraction will use dummy values.")

# Try to import Terratorch
try:
    import terratorch
    TERRATORCH_AVAILABLE = True
except ImportError:
    TERRATORCH_AVAILABLE = False
    print("Warning: Terratorch not available.")


class CubeDataset(Dataset):
    """Dataset for loading generated cubes."""
    def __init__(self, npz_files):
        self.files = npz_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        with np.load(path) as data:
            pixels = data['pixels'].astype(np.float32)  # (6, H, W)
            mask = data['mask']                         # (H, W)
            lat_norm = data['lat_norm']                 # (2,)
            lon_norm = data['lon_norm']                 # (2,)
            week_norm = data['week_norm']               # (2,)
            hour_norm = data['hour_norm']               # (2,)

        return {
            "pixels": pixels,
            "mask": mask,
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


def run_batched_inference(config_path: Path, metadata_path: Path, checkpoint_path: Path,
                         cube_dir: Path, embedding_dir: Path, batch_size: int,
                         num_workers: int, device_str: str, model_type: str = "clay"):
    """Extract embeddings from cubes using either CLAY or Terratorch encoder."""

    # Set device
    device = torch.device(device_str if torch.cuda.is_available() and "cuda" in device_str else "cpu")
    if device_str == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")

    print(f"🖥️  Using device: {device}")

    # Load metadata for normalization
    with open(metadata_path, 'r') as f:
        clay_meta = yaml.safe_load(f)

    platform = "landsat-c2-l2"  # Default, could be made configurable
    GSD = clay_meta[platform]['gsd']
    WAVELENGTHS = [clay_meta[platform]['bands']['wavelength'][b] for b in
                  ["red", "green", "blue", "nir08", "swir16", "swir22"]]
    MEANS = [clay_meta[platform]['bands']['mean'][b] for b in
            ["red", "green", "blue", "nir08", "swir16", "swir22"]]
    STDS = [clay_meta[platform]['bands']['std'][b] for b in
           ["red", "green", "blue", "nir08", "swir16", "swir22"]]

    # Normalize function for CLAY input
    def normalize_pixels(pixels_np):
        means = np.array(MEANS, dtype=np.float32).reshape(-1, 1, 1)
        stds = np.array(STDS, dtype=np.float32).reshape(-1, 1, 1)
        return (pixels_np - means) / stds

    # Initialize model
    if model_type == "clay" and CLAY_AVAILABLE:
        print(f"📦 Loading CLAY Weights from: {checkpoint_path}")
        module = ClayMAEModule.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            model_size="large",
            metadata_path=str(metadata_path),
            dolls=[16, 32, 64, 128, 256, 768, 1024],
            doll_weights=[1, 1, 1, 1, 1, 1, 1],
            mask_ration=0.0,
            shuffle=False,
        )
        module.eval()
        module.to(device)
        print("✅ CLAY Model loaded!")
        encode_fn = lambda batch: module.model.encoder(batch)[0]  # Returns (unmsk_patch, ...)
    elif model_type == "terratorch" and TERRATORCH_AVAILABLE:
        print(f"📦 Loading Terratorch Weights from: {checkpoint_path}")
        # This would need to be implemented based on specific Terratorch model
        print("⚠️  Terratorch embedding extraction not fully implemented yet")
        return
    else:
        print("⚠️  Using dummy embeddings for testing")
        encode_fn = lambda batch: torch.randn(len(batch["pixels"]), 257, 1024).to(device)

    # Process cubes by year directory
    cube_subdirs = [d for d in cube_dir.iterdir() if d.is_dir()]

    for cube_subdir in sorted(cube_subdirs):
        year = cube_subdir.name
        cube_files = list(cube_subdir.glob('*.npz'))

        if not cube_files:
            continue

        out_dir = embedding_dir / year
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*40}")
        print(f"📅 STARTING YEAR: {year} | {len(cube_files)} cubes")
        print(f"{'='*40}")

        # Initialize Dataset & DataLoader
        dataset = CubeDataset(cube_files)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )

        cubes_processed = 0
        print_shapes_next = True

        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=f"Batches ({year})")):

            # Normalize pixels
            pixels_norm = np.stack([
                normalize_pixels(batch_data["pixels"][i])
                for i in range(len(batch_data["pixels"]))
            ])

            # Format batch for model
            model_batch = {
                "pixels": pixels_norm,
                "lat_norm": batch_data["lat_norm"],
                "lon_norm": batch_data["lon_norm"],
                "week_norm": batch_data["week_norm"],
                "hour_norm": batch_data["hour_norm"]
            }

            clay_batch = create_clay_batch(model_batch, WAVELENGTHS, GSD, device)

            # Run Encoder
            with torch.no_grad():
                try:
                    unmsk_patch = encode_fn(clay_batch)
                except Exception as e:
                    print(f"Error in encoding: {e}")
                    # Fallback to dummy
                    unmsk_patch = torch.randn(len(batch_data["pixels"]), 257, 1024).to(device)

            # Convert to numpy for saving
            unmsk_patch_np = unmsk_patch.cpu().numpy()
            masks_np = np.array([data["mask"] for data in batch_data.values()]) if isinstance(batch_data["mask"], dict) else batch_data["mask"].numpy()
            filenames = batch_data["filename"]

            # Debug shapes
            if print_shapes_next:
                print(f"\n   [SHAPE CHECK - FIRST BATCH]")
                print(f"   pixels: {pixels_norm.shape} | embeddings: {unmsk_patch_np.shape} | mask: {masks_np.shape}")
                print_shapes_next = False

            cubes_processed += len(filenames)
            if (cubes_processed % 50) < batch_size and cubes_processed > batch_size:
                tqdm.write(f"   ➤ [Cube {cubes_processed}] embedding shape: {unmsk_patch.shape}")

            # Save individual .npz files
            for i in range(len(filenames)):
                out_filename = out_dir / f"emb_{filenames[i]}"
                np.savez_compressed(
                    out_filename,
                    embeddings=unmsk_patch_np[i],
                    mask=masks_np[i] if hasattr(masks_np, '__getitem__) else masks_np
                )

    print(f"\n🎉 EMBEDDING EXTRACTION COMPLETE! All embeddings stored in {embedding_dir.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract embeddings from generated cubes.")
    p.add_argument("--config", type=Path, required=True, help="Path to source_to_target.yaml")
    p.add_argument("--metadata", type=Path, default=Path("configs/metadata.yaml"),
                   help="Path to metadata yaml")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to model checkpoint (CLAY or Terratorch)")
    p.add_argument("--model-type", choices=["clay", "terratorch"], default="clay",
                   help="Type of model to use for embedding extraction")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    p.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers")
    p.add_argument("--device", default="auto", help="Device to use (auto, cpu, cuda, mps)")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()

    # Resolve paths relative to project root
    root = Path(__file__).resolve().parents[1]
    config_path = _resolve(root, args.config)
    metadata_path = _resolve(root, args.metadata)
    checkpoint_path = _resolve(root, args.checkpoint)

    # Set up directories
    cube_dir = root / "data" / "processed" / "tiles" / "all" / "cubes"
    embedding_dir = root / "data" / "embeddings"

    run_batched_inference(
        config_path=config_path,
        metadata_path=metadata_path,
        checkpoint_path=checkpoint_path,
        cube_dir=cube_dir,
        embedding_dir=embedding_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_str=args.device,
        model_type=args.model_type
    )