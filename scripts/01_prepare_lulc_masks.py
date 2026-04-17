#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from osgeo import gdal 


gdal.UseExceptions()


def _resolve(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (root / path)


def run(config_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(config_path.read_text())

    aoi = _resolve(root, cfg["paths"]["aoi_shapefile"])
    source_lulc = cfg["paths"]["source_lulc"]
    out_dir = _resolve(root, cfg["paths"]["prepared_masks_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    target_epsg = cfg["crs"]["target_epsg"]
    res = float(cfg["crs"]["output_res_m"])

    if not aoi.exists():
        raise FileNotFoundError(f"AOI shapefile not found: {aoi}")

    print(f"AOI: {aoi}")
    print(f"Output dir: {out_dir}")

    for year, src in source_lulc.items():
        src_path = _resolve(root, src)
        if not src_path.exists():
            print(f"[SKIP] {year}: source not found -> {src_path}")
            continue

        out_path = out_dir / f"uk_{year}_30m.tif"
        print(f"[RUN ] {year}: {src_path.name} -> {out_path.name}")

        gdal.Warp(
            str(out_path),
            str(src_path),
            options=gdal.WarpOptions(
                cutlineDSName=str(aoi),
                cropToCutline=True,
                dstSRS=target_epsg,
                xRes=res,
                yRes=res,
                dstNodata=0,
                multithread=True,
            ),
        )

    print("Done.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare AOI-clipped LULC masks.")
    p.add_argument("--config", type=Path, required=True, help="Path to source_to_target.yaml")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args.config)
