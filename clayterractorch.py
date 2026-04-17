#!/usr/bin/env python3
"""
ClayTerratorch Unified CLI Tool

A unified command-line interface for the ClayTerratorch LULC prediction pipeline.
Provides subcommands for cube generation, embedding extraction, training, and prediction.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str]) -> int:
    """Run a command and return its exit code."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def cmd_gen_cubes(args):
    """Generate cubes from STAC metadata."""
    script_path = Path(__file__).parent / "scripts" / "02_generate_tiles_from_stac.py"
    cmd = [
        sys.executable, str(script_path),
        "--config", args.config
    ]
    return run_command(cmd)


def cmd_extract_embeddings(args):
    """Extract embeddings from generated cubes."""
    script_path = Path(__file__).parent / "scripts" / "05_extract_embeddings.py"
    cmd = [
        sys.executable, str(script_path),
        "--config", args.config,
        "--metadata", args.metadata,
        "--checkpoint", args.checkpoint,
        "--model-type", args.model_type,
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--device", args.device
    ]
    return run_command(cmd)


def cmd_train(args):
    """Train segmentation head on embeddings."""
    if args.embedding_based:
        # Use embedding-based training
        script_path = Path(__file__).parent / "scripts" / "13_train_embedding_head.py"
        cmd = [
            sys.executable, str(script_path),
            "--embedding-dir", args.embedding_dir,
            "--output-dir", args.output_dir,
            "--in-channels", str(args.in_channels),
            "--num-classes", str(args.num_classes),
            "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers),
            "--max-epochs", str(args.max_epochs),
            "--patience", str(args.patience),
            "--lr", str(args.lr),
            "--weight-decay", str(args.weight_decay)
        ]
    else:
        # Use existing TerraTorch training script
        script_path = Path(__file__).parent / "scripts" / "10_train_terratorch.sh"
        cmd = [
            "bash", str(script_path),
            args.config,
            args.platform or "landsat-c2-l2"
        ]
    return run_command(cmd)


def cmd_predict(args):
    """Predict LULC using trained model."""
    script_path = Path(__file__).parent / "scripts" / "06_predict_lulc.py"
    cmd = [
        sys.executable, str(script_path),
        "--scene-tif", args.scene_tif,
        "--out-tif", args.out_tif,
        "--encoder-checkpoint", args.encoder_checkpoint,
        "--encoder-type", args.encoder_type,
        "--decoder-checkpoint", args.decoder_checkpoint,
        "--metadata", args.metadata,
        "--platform", args.platform,
        "--band-indices", args.band_indices,
        "--tile-size", str(args.tile_size),
        "--stride", str(args.stride),
        "--batch-size", str(args.batch_size),
        "--num-classes", str(args.num_classes),
        "--device", args.device
    ]
    return run_command(cmd)


def cmd_setup(args):
    """Setup environment and check dependencies."""
    print("🔧 Setting up ClayTerratorch environment...")

    # Check if we're in the right directory
    project_root = Path(__file__).parent
    print(f"📁 Project root: {project_root}")

    # Check required directories
    required_dirs = [
        "data/raw",
        "data/processed",
        "data/processed/lulc_masks",
        "data/processed/tiles/all/cubes",
        "data/embeddings",
        "outputs",
        "configs"
    ]

    for dir_path in required_dirs:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Ensuring directory exists: {dir_path}")

    # Check config files
    required_configs = [
        "configs/source_to_target.yaml",
        "configs/metadata.yaml",
        "configs/terratorch_segmentation.yaml"
    ]

    for config_path in required_configs:
        full_path = project_root / config_path
        if full_path.exists():
            print(f"✅ Config found: {config_path}")
        else:
            print(f"⚠️  Config missing: {config_path}")

    print("✅ Setup complete!")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="ClayTerratorch Unified CLI for LULC prediction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup environment
  clayterractorch setup

  # Generate cubes from STAC
  clayterractorch gen-cubes --config configs/source_to_target.yaml

  # Extract embeddings using CLAY
  clayterractorch extract-emb --config configs/source_to_target.yaml --checkpoint ../clay_LULC/models/clay-v1.5.ckpt

  # Train segmentation head
  clayterractorch train --config configs/terratorch_segmentation.yaml

  # Predict LULC on a scene
  clayterractorch predict --scene-tif /path/to/scene.tif --out-tif /path/to/output.tif \\
    --encoder-checkpoint ../clay_LULC/models/clay-v1.5.ckpt \\
    --decoder-checkpoint /path/to/trained_decoder.ckpt
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Setup environment and check dependencies")
    setup_parser.set_defaults(func=cmd_setup)

    # Generate cubes command
    gen_cubes_parser = subparsers.add_parser("gen-cubes", help="Generate image cubes from STAC metadata")
    gen_cubes_parser.add_argument("--config", type=str, required=True,
                                 help="Path to source_to_target.yaml configuration file")
    gen_cubes_parser.set_defaults(func=cmd_gen_cubes)

    # Extract embeddings command
    extract_parser = subparsers.add_parser("extract-emb", help="Extract embeddings from generated cubes")
    extract_parser.add_argument("--config", type=str, required=True,
                               help="Path to source_to_target.yaml configuration file")
    extract_parser.add_argument("--metadata", type=str, default="configs/metadata.yaml",
                               help="Path to metadata yaml file")
    extract_parser.add_argument("--checkpoint", type=str, required=True,
                               help="Path to model checkpoint (CLAY or Terratorch)")
    extract_parser.add_argument("--model-type", choices=["clay", "terratorch"], default="clay",
                               help="Type of model to use for embedding extraction")
    extract_parser.add_argument("--batch-size", type=int, default=32,
                               help="Batch size for inference")
    extract_parser.add_argument("--num-workers", type=int, default=4,
                               help="Number of data loading workers")
    extract_parser.add_argument("--device", default="auto",
                               help="Device to use (auto, cpu, cuda, mps)")
    extract_parser.set_defaults(func=cmd_extract_embeddings)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train segmentation head on embeddings")
    train_parser.add_argument("--config", type=str, required=True,
                             help="Path to Terratorch configuration file (for standard training)")
    train_parser.add_argument("--platform", type=str, default=None,
                             help="Platform for stats synchronization (e.g., landsat-c2-l2)")
    # Embedding-based training options
    train_parser.add_argument("--embedding-based", action="store_true",
                             help="Use embedding-based training instead of standard TerraTorch training")
    train_parser.add_argument("--embedding-dir", type=str, default="data/embeddings",
                             help="Directory containing extracted embeddings (for embedding-based training)")
    train_parser.add_argument("--output-dir", type=str, default="outputs/embedding_training",
                             help="Directory to save checkpoints and logs")
    train_parser.add_argument("--in-channels", type=int, default=1024,
                             help="Number of input channels (embedding dimension)")
    train_parser.add_argument("--num-classes", type=int, default=20,
                             help="Number of LULC classes")
    train_parser.add_argument("--batch-size", type=int, default=16,
                             help="Batch size for training")
    train_parser.add_argument("--num-workers", type=int, default=4,
                             help="Number of data loading workers")
    train_parser.add_argument("--max-epochs", type=int, default=50,
                             help="Maximum number of training epochs")
    train_parser.add_argument("--patience", type=int, default=10,
                             help="Early stopping patience")
    train_parser.add_argument("--lr", type=float, default=1e-3,
                             help="Learning rate")
    train_parser.add_argument("--weight-decay", type=float, default=1e-4,
                             help="Weight decay")
    train_parser.set_defaults(func=cmd_train)

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict LULC using trained model")
    predict_parser.add_argument("--scene-tif", type=str, required=True,
                               help="Input scene TIFF file")
    predict_parser.add_argument("--out-tif", type=str, required=True,
                               help="Output LULC probability TIFF file")
    predict_parser.add_argument("--encoder-checkpoint", type=str, required=True,
                               help="Path to encoder checkpoint (CLAY or Terratorch)")
    predict_parser.add_argument("--encoder-type", choices=["clay", "terratorch"], default="clay",
                               help="Type of encoder model")
    predict_parser.add_argument("--decoder-checkpoint", type=str, required=True,
                               help="Path to trained decoder/head checkpoint")
    predict_parser.add_argument("--metadata", type=str, default="configs/metadata.yaml",
                               help="Path to metadata yaml file")
    predict_parser.add_argument("--platform", type=str, default="landsat-c2-l2",
                               help="Platform for normalization (must match metadata)")
    predict_parser.add_argument("--band-indices", type=str, default="1,2,3,4,5,6",
                               help="Comma-separated list of band indices (1-based)")
    predict_parser.add_argument("--tile-size", type=int, default=256,
                               help="Tile size for prediction")
    predict_parser.add_argument("--stride", type=int, default=128,
                               help="Stride for sliding window")
    predict_parser.add_argument("--batch-size", type=int, default=16,
                               help="Batch size for prediction")
    predict_parser.add_argument("--num-classes", type=int, default=20,
                               help="Number of LULC classes")
    predict_parser.add_argument("--device", default="auto",
                               help="Device to use (auto, cpu, cuda, mps)")
    predict_parser.set_defaults(func=cmd_predict)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())