#!/usr/bin/env python3
"""
ClayTerratorch — Unified CLI for TerraTorch + Clay LULC pipeline.

All settings live in YAML configs. Pass --config, not individual params.

Pipeline order:
  setup → prep → cubes → splits → stats → train → predict
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT    = Path(__file__).parent
SCRIPTS = ROOT / "scripts"


def run(cmd: list[str]) -> int:
    """Print and execute a shell command, return exit code."""
    print(f"\n$ {' '.join(cmd)}\n")
    return subprocess.run(cmd).returncode


# ── data pipeline commands ────────────────────────────────────────────────────

def cmd_setup(args):
    """Create all required project directories."""
    dirs = [
        "data/raw",
        "data/processed/lulc_masks",
        "data/processed/tiles/all/cubes",
        "data/processed/tiles/all/images",
        "data/processed/tiles/all/masks",
        "data/processed/tiles/train/images", "data/processed/tiles/train/masks",
        "data/processed/tiles/val/images",   "data/processed/tiles/val/masks",
        "data/processed/tiles/test/images",  "data/processed/tiles/test/masks",
        "outputs/terratorch_clay/checkpoints",
        "outputs/predictions",
        "configs/stats",
    ]
    print("\n[setup] Creating project directories...\n")
    for d in dirs:
        (ROOT / d).mkdir(parents=True, exist_ok=True)
        print(f"  [ok] {d}")
    print("\n[setup] Done.\n")
    return 0


def cmd_prep(args):
    """Clip + reproject raw LULC GeoTIFFs to AOI. → 01_prepare_lulc_masks.py"""
    return run([sys.executable, str(SCRIPTS / "01_prepare_lulc_masks.py"),
                "--config", args.config])


def cmd_cubes(args):
    """Fetch Landsat tiles from Planetary Computer and save .npz cubes. → 02_generate_tiles_from_stac.py"""
    return run([sys.executable, str(SCRIPTS / "02_generate_tiles_from_stac.py"),
                "--config", args.config])


def cmd_splits(args):
    """Stratified train/val/test split by dominant LULC class. → 03_make_splits.py"""
    return run([sys.executable, str(SCRIPTS / "03_make_splits.py"),
                "--config", args.config])


def cmd_stats(args):
    """Sync per-band means/stds from metadata.yaml → stats txt files + update Clay config. → 04_compute_stats.py"""
    # Always updates the Clay config; metadata path derived from project root
    return run([sys.executable, str(SCRIPTS / "04_compute_stats.py"),
                "--config", str(ROOT / "configs" / "terratorch_segmentation_clay.yaml")])


# ── model commands ────────────────────────────────────────────────────────────

def cmd_train(args):
    """
    Train Clay encoder (frozen) + UNetDecoder via TerraTorch Lightning CLI.
    All hyperparameters, paths, and callbacks are defined in the YAML config.
    """
    return run([sys.executable, "-m", "terratorch", "fit",
                "--config", args.config])


def cmd_predict(args):
    """
    Run sliding-window LULC prediction on a full Landsat scene via TerraTorch.
    Tile size and stride are defined in the YAML config (tiled_inference_parameters).
    """
    cmd = [
        sys.executable, "-m", "terratorch", "predict",
        "--config",   args.config,
        "--ckpt_path", args.ckpt,
        "--trainer.inference_mode", "true",
    ]
    if args.scene:
        cmd += ["--data.predict_data_root", args.scene]
    if args.out:
        cmd += ["--model.output_dir", args.out]
    return run(cmd)


# ── CLI wiring ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ClayTerratorch — TerraTorch + Clay LULC pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline (run in order):
  clayterractorch setup
  clayterractorch prep    --config configs/source_to_target.yaml
  clayterractorch cubes   --config configs/source_to_target.yaml
  clayterractorch splits  --config configs/source_to_target.yaml
  clayterractorch stats   --config configs/source_to_target.yaml
  clayterractorch train   --config configs/terratorch_segmentation_clay.yaml
  clayterractorch predict --config configs/terratorch_segmentation_clay.yaml \\
                          --ckpt   outputs/terratorch_clay/checkpoints/best.ckpt \\
                          --scene  /path/to/scene.tif \\
                          --out    outputs/predictions/lulc.tif
        """
    )

    sub = parser.add_subparsers(dest="command")

    # setup — no config needed
    sub.add_parser("setup", help="Create project directories").set_defaults(func=cmd_setup)

    # data pipeline — all driven by source_to_target.yaml
    for name, fn, default_cfg in [
        ("prep",   cmd_prep,   "configs/source_to_target.yaml"),
        ("cubes",  cmd_cubes,  "configs/source_to_target.yaml"),
        ("splits", cmd_splits, "configs/source_to_target.yaml"),
        ("stats",  cmd_stats,  "configs/source_to_target.yaml"),  # config arg unused by stats (kept for consistency)
        ("train",  cmd_train,  "configs/terratorch_segmentation_clay.yaml"),
    ]:
        p = sub.add_parser(name)
        p.add_argument("--config", default=default_cfg)
        p.set_defaults(func=fn)

    # predict — needs ckpt + optional scene/out
    pred = sub.add_parser("predict", help="Predict LULC on a new scene")
    pred.add_argument("--config", default="configs/terratorch_segmentation_clay.yaml")
    pred.add_argument("--ckpt",   required=True, help="Path to trained .ckpt file")
    pred.add_argument("--scene",  help="Input scene GeoTIFF path")
    pred.add_argument("--out",    help="Output LULC GeoTIFF path")
    pred.set_defaults(func=cmd_predict)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
