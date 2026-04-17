#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def _resolve(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (root / path)


def _run_cmd(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _check_imports(mods: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in mods:
        try:
            importlib.import_module(m)
            out[m] = "OK"
        except Exception as e:  # noqa: BLE001
            out[m] = f"FAIL: {e}"
    return out


def _script_help_check(script_path: Path) -> str:
    rc, _, err = _run_cmd([sys.executable, str(script_path), "--help"])
    if rc == 0:
        return "OK"
    return f"FAIL: exit={rc} err={err[:200]}"


def _normalization_check(config_path: Path, root: Path) -> dict[str, str]:
    cfg = yaml.safe_load(config_path.read_text())
    init = cfg.get("data", {}).get("init_args", {})
    out: dict[str, str] = {}

    for key in ["means", "stds"]:
        val = init.get(key)
        if isinstance(val, list):
            out[key] = "WARN: hardcoded list (prefer metadata-driven file path)"
        elif isinstance(val, str):
            p = _resolve(root, val)
            out[key] = "OK" if p.exists() else f"MISSING_FILE: {p}"
        else:
            out[key] = f"FAIL: unsupported type {type(val)}"
    return out


def _check_terratorch_cli() -> tuple[str, str]:
    # returns (status, mode)
    if shutil.which("terratorch"):
        rc, out, err = _run_cmd(["terratorch", "--help"])
        if rc == 0:
            return "OK", "terratorch"
        return f"FAIL: {(err or out)[:200]}", "terratorch"

    rc, out, err = _run_cmd([sys.executable, "-m", "terratorch", "--help"])
    if rc == 0:
        return "OK", "python -m terratorch"
    return f"FAIL: {(err or out)[:200]}", "python -m terratorch"


def run(config_path: Path) -> int:
    root = Path(__file__).resolve().parents[1]

    report: dict[str, Any] = {
        "project_root": str(root),
        "python": sys.executable,
        "checks": {},
        "warnings": [],
    }

    # 1) Parse config files
    seg_cfg = root / "configs" / "terratorch_segmentation.yaml"
    clay_cfg = root / "configs" / "terratorch_segmentation_clay_template.yaml"
    metadata_cfg = root / "configs" / "metadata.yaml"

    configs = [config_path, seg_cfg, clay_cfg]
    cfg_status = {}
    for p in configs:
        try:
            yaml.safe_load(p.read_text())
            cfg_status[str(p)] = "OK"
        except Exception as e:  # noqa: BLE001
            cfg_status[str(p)] = f"FAIL: {e}"
    report["checks"]["config_parse"] = cfg_status
    report["checks"]["metadata_yaml"] = "OK" if metadata_cfg.exists() else f"MISSING: {metadata_cfg}"

    # 2) Import checks
    mods = [
        "terratorch",
        "torch",
        "lightning",
        "pytorch_lightning",
        "torchmetrics",
        "rasterio",
        "rioxarray",
        "odc.stac",
        "pystac_client",
        "planetary_computer",
        "numpy",
        "pandas",
        "yaml",
    ]
    report["checks"]["imports"] = _check_imports(mods)

    # 3) source_to_target path checks
    cfg = yaml.safe_load(config_path.read_text())
    path_checks: dict[str, str] = {}

    aoi = _resolve(root, cfg["paths"]["aoi_shapefile"])
    path_checks["aoi_shapefile"] = "OK" if aoi.exists() else f"MISSING: {aoi}"

    for year, p in cfg["paths"]["source_lulc"].items():
        pp = _resolve(root, p)
        key = f"source_lulc_{year}"
        path_checks[key] = "OK" if pp.exists() else f"MISSING: {pp}"

    out_dirs = [
        cfg["paths"]["prepared_masks_dir"],
        cfg["paths"]["tiles_all_images_dir"],
        cfg["paths"]["tiles_all_masks_dir"],
    ]
    for od in out_dirs:
        pp = _resolve(root, od)
        pp.mkdir(parents=True, exist_ok=True)
        path_checks[f"output_dir::{od}"] = "OK"

    report["checks"]["source_target_paths"] = path_checks

    # 4) CLI availability checks
    cli_status, cli_mode = _check_terratorch_cli()
    report["checks"]["terratorch_cli"] = cli_status
    report["checks"]["terratorch_cli_mode"] = cli_mode

    # 5) Script help checks (non-destructive)
    script_checks = {}
    for s in [
        "01_prepare_lulc_masks.py",
        "02_generate_tiles_from_stac.py",
        "03_make_splits.py",
        "04_compute_stats.py",
        "12_predict_large_scene.py",
    ]:
        script_checks[s] = _script_help_check(root / "scripts" / s)
    report["checks"]["script_help"] = script_checks

    # 6) Normalization wiring checks
    report["checks"]["normalization::terratorch_segmentation"] = _normalization_check(seg_cfg, root)
    report["checks"]["normalization::terratorch_segmentation_clay_template"] = _normalization_check(clay_cfg, root)

    # 7) Existing data sanity (if already generated)
    data_checks = {}
    all_img = list((root / "data" / "processed" / "tiles" / "all" / "images").glob("*.tif"))
    all_msk = list((root / "data" / "processed" / "tiles" / "all" / "masks").glob("*.tif"))
    data_checks["tiles_all_images_count"] = len(all_img)
    data_checks["tiles_all_masks_count"] = len(all_msk)
    if len(all_img) != len(all_msk):
        report["warnings"].append("Image/mask tile counts differ in data/processed/tiles/all")

    report["checks"]["data_sanity"] = data_checks

    print(json.dumps(report, indent=2))

    # return non-zero if any hard failures
    hard_fail = False
    for grp in report["checks"].values():
        if isinstance(grp, dict):
            for v in grp.values():
                if isinstance(v, str) and (v.startswith("FAIL") or v.startswith("MISSING_FILE")):
                    hard_fail = True
        elif isinstance(grp, str):
            if grp.startswith("FAIL"):
                hard_fail = True

    return 1 if hard_fail else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dry run checks for ClayTerratorch pipeline.")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/source_to_target.yaml"),
        help="Path to source_to_target config",
    )
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    raise SystemExit(run(args.config))
