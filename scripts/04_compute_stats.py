#!/usr/bin/env python3
"""
04_compute_stats.py
===================
Reads per-band mean/std from configs/metadata.yaml and writes them as
plain-text files (one value per line) that TerraTorch configs reference.
Also patches the YAML config to point at the generated stats files.

Input  : configs/metadata.yaml  +  one or more TerraTorch YAML configs
Output : configs/stats/<config_stem>.<platform>.means.txt
         configs/stats/<config_stem>.<platform>.stds.txt
         (config YAML updated in-place)

Config keys used (metadata.yaml):
  <platform>.bands.mean.<band>
  <platform>.bands.std.<band>
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tqdm import tqdm


# Maps TerraTorch/TorchGeo band tokens → metadata.yaml keys
BAND_ALIASES = {
    "BLUE":       "blue",
    "GREEN":      "green",
    "RED":        "red",
    "NIR":        "nir08",
    "NIR08":      "nir08",
    "NIR_NARROW": "nir08",
    "SWIR1":      "swir16",
    "SWIR_1":     "swir16",
    "SWIR16":     "swir16",
    "SWIR2":      "swir22",
    "SWIR_2":     "swir22",
    "SWIR22":     "swir22",
}


def normalize_band_name(name: str) -> str:
    return BAND_ALIASES.get(str(name).strip().upper(), str(name).strip().lower())


def infer_band_order(cfg: dict[str, Any]) -> list[str]:
    """Extract band list from TerraTorch data config."""
    init_args = cfg.get("data", {}).get("init_args", {})
    bands = init_args.get("dataset_bands") or init_args.get("output_bands")
    if not bands:
        raise ValueError("Cannot infer band order: missing data.init_args.dataset_bands or output_bands.")
    return [normalize_band_name(str(b)) for b in bands]


def _write_txt(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(values, dtype=np.float32), fmt="%.10g")


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


# ── per-config processing ─────────────────────────────────────────────────────

def sync_config(
    config_path: Path,
    metadata: dict[str, Any],
    platform: str,
    stats_dir: Path,
    project_root: Path,
    dry_run: bool,
) -> None:
    cfg        = yaml.safe_load(config_path.read_text())
    init_args  = cfg.setdefault("data", {}).setdefault("init_args", {})
    band_order = infer_band_order(cfg)

    if platform not in metadata:
        raise KeyError(f"Platform '{platform}' not in metadata.yaml.")

    means_map = metadata[platform]["bands"]["mean"]
    stds_map  = metadata[platform]["bands"]["std"]

    missing = [b for b in band_order if b not in means_map or b not in stds_map]
    if missing:
        raise KeyError(f"Bands missing in metadata for '{platform}': {missing}")

    means = [float(means_map[b]) for b in band_order]
    stds  = [float(stds_map[b])  for b in band_order]

    means_file = stats_dir / f"{config_path.stem}.{platform}.means.txt"
    stds_file  = stats_dir / f"{config_path.stem}.{platform}.stds.txt"

    tqdm.write(f"\n  Config   : {config_path.name}")
    tqdm.write(f"  Platform : {platform}")
    tqdm.write(f"  Bands    : {band_order}")
    tqdm.write(f"  Means    : {means}")
    tqdm.write(f"  Stds     : {stds}")

    if not dry_run:
        _write_txt(means_file, means)
        _write_txt(stds_file,  stds)

        init_args["means"] = _rel(means_file, project_root)
        init_args["stds"]  = _rel(stds_file,  project_root)
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        tqdm.write(f"  [ok] stats written + config updated")
    else:
        tqdm.write(f"  [dry-run] no files written")


# ── main ──────────────────────────────────────────────────────────────────────

def run(metadata_yaml: Path, platform: str, configs: list[Path], stats_dir: Path, dry_run: bool) -> None:
    project_root = Path(__file__).resolve().parents[1]

    if not metadata_yaml.exists():
        raise FileNotFoundError(f"metadata.yaml not found: {metadata_yaml}")

    print(f"\n[stats] Metadata : {metadata_yaml}")
    print(f"[stats] Platform : {platform}")
    print(f"[stats] Configs  : {len(configs)}\n")

    metadata = yaml.safe_load(metadata_yaml.read_text())

    # ── iterate over configs with progress bar ────────────────────────────────
    for cfg_path in tqdm(configs, desc="Syncing stats", unit="config"):
        if not cfg_path.exists():
            tqdm.write(f"  [SKIP] config not found: {cfg_path}")
            continue
        sync_config(cfg_path, metadata, platform, stats_dir, project_root, dry_run)

    print(f"\n[stats] Done. Stats files in {stats_dir}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Sync TerraTorch band stats from metadata.yaml.")
    p.add_argument(
        "--metadata-yaml", type=Path,
        default=project_root / "configs" / "metadata.yaml",
    )
    p.add_argument("--platform", type=str, default="landsat-c2-l2")
    p.add_argument(
        "--config", type=Path, action="append", default=[],
        help="TerraTorch config to update (repeatable). Defaults to Clay config.",
    )
    p.add_argument(
        "--stats-dir", type=Path,
        default=project_root / "configs" / "stats",
    )
    p.add_argument("--dry-run", action="store_true")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]

    # Default: update the Clay segmentation config only
    configs = args.config or [project_root / "configs" / "terratorch_segmentation_clay.yaml"]

    run(
        metadata_yaml=args.metadata_yaml,
        platform=args.platform,
        configs=configs,
        stats_dir=args.stats_dir,
        dry_run=args.dry_run,
    )
