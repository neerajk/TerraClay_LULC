#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# Maps common TerraTorch/TorchGeo band tokens to CLAY metadata keys.
BAND_ALIASES = {
    "BLUE": "blue",
    "GREEN": "green",
    "RED": "red",
    "NIR": "nir08",
    "NIR08": "nir08",
    "NIR_NARROW": "nir08",
    "SWIR1": "swir16",
    "SWIR_1": "swir16",
    "SWIR16": "swir16",
    "SWIR2": "swir22",
    "SWIR_2": "swir22",
    "SWIR22": "swir22",
}


def normalize_band_name(name: str) -> str:
    key = str(name).strip()
    return BAND_ALIASES.get(key.upper(), key.lower())


def infer_band_order(cfg: dict[str, Any]) -> list[str]:
    init_args = cfg.get("data", {}).get("init_args", {})
    bands = init_args.get("dataset_bands") or init_args.get("output_bands")
    if not bands:
        raise ValueError("Could not infer band order. Expected data.init_args.dataset_bands or output_bands.")
    return [normalize_band_name(str(b)) for b in bands]


def _to_relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _write_vector_txt(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(values, dtype=np.float32), fmt="%.10g")


def sync_one_config(
    config_path: Path,
    metadata: dict[str, Any],
    platform: str,
    stats_dir: Path,
    project_root: Path,
    dry_run: bool,
) -> None:
    cfg = yaml.safe_load(config_path.read_text())
    init_args = cfg.setdefault("data", {}).setdefault("init_args", {})

    band_order = infer_band_order(cfg)

    if platform not in metadata:
        raise KeyError(f"Platform '{platform}' not found in metadata file.")

    means_map = metadata[platform]["bands"]["mean"]
    stds_map = metadata[platform]["bands"]["std"]

    missing = [b for b in band_order if b not in means_map or b not in stds_map]
    if missing:
        raise KeyError(
            "These bands are missing in metadata for platform "
            f"'{platform}': {missing}. Inferred order={band_order}"
        )

    means = [float(means_map[b]) for b in band_order]
    stds = [float(stds_map[b]) for b in band_order]

    means_file = stats_dir / f"{config_path.stem}.{platform}.means.txt"
    stds_file = stats_dir / f"{config_path.stem}.{platform}.stds.txt"

    means_ref = _to_relative_or_absolute(means_file, project_root)
    stds_ref = _to_relative_or_absolute(stds_file, project_root)

    print(f"\nConfig: {config_path}")
    print(f"  platform: {platform}")
    print(f"  inferred band order: {band_order}")
    print(f"  means values: {means}")
    print(f"  stds values : {stds}")
    print(f"  means file  : {means_ref}")
    print(f"  stds file   : {stds_ref}")

    if not dry_run:
        _write_vector_txt(means_file, means)
        _write_vector_txt(stds_file, stds)

        init_args["means"] = means_ref
        init_args["stds"] = stds_ref
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

        print("  updated config: YES")
        print("  wrote stats files: YES")
    else:
        print("  updated config: NO (dry-run)")
        print("  wrote stats files: NO (dry-run)")


def run(metadata_yaml: Path, platform: str, configs: list[Path], stats_dir: Path, dry_run: bool) -> None:
    project_root = Path(__file__).resolve().parents[1]

    if not metadata_yaml.exists():
        raise FileNotFoundError(f"metadata.yaml not found: {metadata_yaml}")

    metadata = yaml.safe_load(metadata_yaml.read_text())

    for cfg_path in configs:
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")
        sync_one_config(
            config_path=cfg_path,
            metadata=metadata,
            platform=platform,
            stats_dir=stats_dir,
            project_root=project_root,
            dry_run=dry_run,
        )


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(
        description=(
            "Sync TerraTorch means/stds from metadata.yaml: write values to txt files and "
            "set config means/stds to those file paths (no hardcoded arrays)."
        )
    )
    p.add_argument(
        "--metadata-yaml",
        type=Path,
        default=project_root / "configs" / "metadata.yaml",
        help="Path to metadata.yaml",
    )
    p.add_argument("--platform", type=str, default="landsat-c2-l2", help="Metadata platform key")
    p.add_argument(
        "--config",
        type=Path,
        action="append",
        default=[],
        help="Config file to update. If omitted, both default seg configs are updated.",
    )
    p.add_argument(
        "--stats-dir",
        type=Path,
        default=project_root / "configs" / "stats",
        help="Output directory for means/stds txt files",
    )
    p.add_argument("--dry-run", action="store_true", help="Preview values without writing files")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()

    project_root = Path(__file__).resolve().parents[1]
    configs = args.config
    if not configs:
        configs = [
            project_root / "configs" / "terratorch_segmentation.yaml",
            project_root / "configs" / "terratorch_segmentation_clay_template.yaml",
        ]

    run(
        metadata_yaml=args.metadata_yaml,
        platform=args.platform,
        configs=configs,
        stats_dir=args.stats_dir,
        dry_run=args.dry_run,
    )
