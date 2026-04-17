#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
import rioxarray
import yaml
from odc import stac as odc_stac
from tqdm import tqdm


@dataclass
class QualityCfg:
    min_lulc_coverage_pct: float
    max_nodata_pct: float
    max_black_pct: float
    max_blue_mean: float


def _resolve(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (root / path)


def _write_raster(path: Path, arr: np.ndarray, transform, crs, dtype: str, nodata: float | int) -> None:
    height, width = arr.shape[-2:]
    count = 1 if arr.ndim == 2 else arr.shape[0]
    data = arr if arr.ndim == 3 else arr[np.newaxis, ...]

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "lzw",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(dtype))


def _quality_ok(pixels: np.ndarray, q: QualityCfg) -> tuple[bool, dict[str, float]]:
    nan_mask = np.isnan(pixels).any(axis=0)
    nan_pct = float(100.0 * nan_mask.mean())

    black_mask = (pixels == 0).all(axis=0)
    black_pct = float(100.0 * black_mask.mean())

    blue_mean = float(np.nanmean(pixels[2]))

    ok = nan_pct <= q.max_nodata_pct and black_pct <= q.max_black_pct and blue_mean < q.max_blue_mean
    return ok, {"nan_pct": nan_pct, "black_pct": black_pct, "blue_mean": blue_mean}


def _normalize_latlon(lat_vals: np.ndarray, lon_vals: np.ndarray, transform, crs, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Normalize latitude and longitude values to [-1, 1] range."""
    # Get the four corners of the tile
    left = transform.c
    top = transform.f
    right = transform.c + transform.a * width
    bottom = transform.f + transform.e * height

    # Convert to lat/lon if needed
    if crs != "EPSG:4326":
        # Transform corners to WGS84
        try:
            import pyproj
            transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            lon_left, lat_left = transformer.transform(left, top)
            lon_right, lat_right = transformer.transform(right, bottom)
        except ImportError:
            # Fallback to approximate transformation if pyproj not available
            lon_left, lat_left = left, top
            lon_right, lat_right = right, bottom
    else:
        lon_left, lat_left = left, top
        lon_right, lat_right = right, bottom

    # Normalize to [-1, 1]
    lat_norm = 2 * (lat_vals - lat_left) / (lat_right - lat_left) - 1
    lon_norm = 2 * (lon_vals - lon_left) / (lon_right - lon_left) - 1

    # Clip to valid range
    lat_norm = np.clip(lat_norm, -1, 1)
    lon_norm = np.clip(lon_norm, -1, 1)

    return lat_norm, lon_norm


def _get_temporal_norm(dt) -> tuple[float, float]:
    """Extract and normalize week-of-year and hour-of-day from datetime."""
    # Week of year (0-51) normalized to [-1, 1]
    week_of_year = dt.timetuple().tm_yday // 7
    week_norm = 2 * (week_of_year / 51.0) - 1

    # Hour of day (0-23) normalized to [-1, 1]
    hour_of_day = dt.hour + dt.minute / 60.0
    hour_norm = 2 * (hour_of_day / 23.0) - 1

    return week_norm, hour_norm


def run(config_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(config_path.read_text())

    src = cfg["source"]
    qcfg = QualityCfg(**cfg["quality"])
    bands = cfg["bands"]["order"]
    tile_size = int(cfg["tiling"]["tile_size"])
    stride = int(cfg["tiling"]["stride"])

    mask_dir = _resolve(root, cfg["paths"]["prepared_masks_dir"])
    out_cube_dir = _resolve(root, cfg["paths"]["tiles_all_cubes_dir"])
    out_csv = _resolve(root, cfg["paths"]["tiles_metadata_csv"])

    out_cube_dir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    allowed_months = set(src["allowed_months"])
    max_alternates = int(src["max_alternates"])

    catalog = pystac_client.Client.open(src["stac_api"], modifier=planetary_computer.sign_inplace)

    rows: list[dict] = []
    tile_counter = 0

    for year, date_range in src["date_ranges"].items():
        mask_path = mask_dir / f"uk_{year}_30m.tif"
        if not mask_path.exists():
            print(f"[SKIP] mask missing for {year}: {mask_path}")
            continue

        mask = rioxarray.open_rasterio(mask_path)
        ny, nx = mask.sizes["y"], mask.sizes["x"]

        y_starts = list(range(0, max(0, ny - tile_size + 1), stride))
        x_starts = list(range(0, max(0, nx - tile_size + 1), stride))

        print(f"\nYEAR {year}: mask={mask_path.name} tiles={len(y_starts) * len(x_starts)}")

        for i in tqdm(y_starts, desc=f"Tiles {year}"):
            for j in x_starts:
                lbl_tile = mask.isel(y=slice(i, i + tile_size), x=slice(j, j + tile_size))
                label = lbl_tile.values.squeeze().astype(np.uint8)

                valid_pct = float(100.0 * np.mean(label > 0))
                if valid_pct < qcfg.min_lulc_coverage_pct:
                    continue

                bbox = lbl_tile.rio.transform_bounds("EPSG:4326")
                search = catalog.search(
                    collections=[src["collection"]],
                    bbox=bbox,
                    datetime=date_range,
                    query=src.get("query", {}),
                )

                items = [it for it in search.items() if it.datetime and it.datetime.month in allowed_months]
                if not items:
                    continue

                items = sorted(items, key=lambda x: x.properties.get("eo:cloud_cover", 100.0))

                chosen_pixels = None
                chosen_item = None
                chosen_metrics = None

                for item in items[:max_alternates]:
                    try:
                        ds = odc_stac.load(
                            [item],
                            geobox=lbl_tile.odc.geobox,
                            bands=bands,
                            resampling="bilinear",
                        ).squeeze().compute()

                        ds = ds[bands]
                        pixels = ds.to_array().values.astype(np.float32)
                        ok, metrics = _quality_ok(pixels, qcfg)
                        if ok:
                            chosen_pixels = pixels
                            chosen_item = item
                            chosen_metrics = metrics
                            break
                    except Exception:
                        continue
                    finally:
                        gc.collect()

                if chosen_pixels is None or chosen_item is None or chosen_metrics is None:
                    continue

                # Generate cube ID
                tile_id = f"y{year}_r{i}_c{j}_{tile_counter:07d}"
                cube_path = out_cube_dir / f"{tile_id}.npz"

                # Extract metadata for normalization
                transform = lbl_tile.rio.transform()
                crs = lbl_tile.rio.crs

                # Get center coordinates for lat/lon normalization
                center_y, center_x = tile_size // 2, tile_size // 2
                # Create coordinate arrays for the tile center point
                lat_center = np.array([transform * (center_x, center_y)][0][1])  # y,x -> (x,y) for transform
                lon_center = np.array([transform * (center_x, center_y)][0][0])

                # For simplicity, we'll use the center point lat/lon
                # In a more advanced version, we could compute per-pixel or use corner-based normalization
                lat_val = np.array([lat_center])  # Make it an array for consistency
                lon_val = np.array([lon_center])

                # Normalize lat/lon to [-1, 1] using tile bounds
                lat_norm, lon_norm = _normalize_latlon(lat_val, lon_val, transform, crs, 1, 1)

                # Get temporal info
                dt = chosen_item.datetime
                week_norm, hour_norm = _get_temporal_norm(dt)
                week_norm = np.array([week_norm])
                hour_norm = np.array([hour_norm])

                # Save as .npz cube (matching clay_LULC format)
                np.savez_compressed(
                    cube_path,
                    pixels=chosen_pixels,           # (6, H, W)
                    mask=label,                     # (H, W)
                    lat_norm=lat_norm,              # (2,) - will be broadcast or used as center
                    lon_norm=lon_norm,              # (2,)
                    week_norm=week_norm,            # (2,)
                    hour_norm=hour_norm             # (2,)
                )

                rows.append(
                    {
                        "tile_id": tile_id,
                        "year": int(year),
                        "row": i,
                        "col": j,
                        "cube_path": str(cube_path.relative_to(root)),
                        "collection": src["collection"],
                        "scene_id": chosen_item.id,
                        "acq_datetime": chosen_item.datetime.isoformat() if chosen_item.datetime else "",
                        "catalog_cloud_cover": float(chosen_item.properties.get("eo:cloud_cover", np.nan)),
                        "valid_lulc_pct": valid_pct,
                        **chosen_metrics,
                    }
                )
                tile_counter += 1

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
        print(f"\nSaved tiles metadata: {out_csv} ({len(df)} rows)")
    else:
        print("No tiles produced. Check source filters and mask coverage thresholds.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate source->target image/mask cubes from STAC.")
    p.add_argument("--config", type=Path, required=True, help="Path to source_to_target.yaml")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args.config)