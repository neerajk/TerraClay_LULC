#!/usr/bin/env python3
"""
01_prepare_lulc_masks.py
========================
Clips raw decadal LULC GeoTIFFs to the Uttarakhand AOI boundary,
reprojects to target CRS, and resamples to target resolution.

Input  : Raw LULC TIFs (1985 / 1995 / 2005) + AOI shapefile
Output : data/processed/lulc_masks/uk_<year>_30m.tif

Config keys used (source_to_target.yaml):
  paths.aoi_shapefile
  paths.source_lulc       — dict of {year: path}
  paths.prepared_masks_dir
  crs.target_epsg
  crs.output_res_m
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from osgeo import gdal
from tqdm import tqdm

gdal.UseExceptions()


def _resolve(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (root / path)


def run(config_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    cfg  = yaml.safe_load(config_path.read_text())

    aoi     = _resolve(root, cfg["paths"]["aoi_shapefile"])
    out_dir = _resolve(root, cfg["paths"]["prepared_masks_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    target_epsg = cfg["crs"]["target_epsg"]
    res         = float(cfg["crs"]["output_res_m"])
    years       = cfg["paths"]["source_lulc"]

    if not aoi.exists():
        raise FileNotFoundError(f"AOI shapefile not found: {aoi}")

    print(f"\n[prep] AOI      : {aoi}")
    print(f"[prep] Output   : {out_dir}")
    print(f"[prep] CRS      : {target_epsg}  @  {res}m\n")

    # ── process each year ─────────────────────────────────────────────────────
    with tqdm(years.items(), desc="Preparing LULC masks", unit="year") as pbar:
        for year, src in pbar:
            src_path = _resolve(root, src)
            out_path = out_dir / f"uk_{year}_30m.tif"
            pbar.set_postfix(year=year, status="processing")

            if not src_path.exists():
                tqdm.write(f"  [SKIP] {year}: source not found → {src_path}")
                pbar.set_postfix(year=year, status="skipped")
                continue

            tqdm.write(f"  [RUN ] {year}: {src_path.name} → {out_path.name}")

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
            pbar.set_postfix(year=year, status="done")

    print(f"\n[prep] Done. Masks saved to {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Clip + reproject LULC masks to AOI.")
    p.add_argument("--config", type=Path, required=True, help="Path to source_to_target.yaml")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args().config)
