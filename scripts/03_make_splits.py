#!/usr/bin/env python3
"""
03_make_splits.py
=================
Stratified train / val / test split of all processed tiles.
Stratification is by dominant LULC class so every class appears in each split.

Input  : data/processed/tiles/all/images/ + masks/
Output : data/processed/tiles/{train,val,test}/{images,masks}/
         data/processed/tiles/split_manifest.csv

Config keys used (source_to_target.yaml):
  paths.tiles_all_cubes_dir   — parent dir; images/ and masks/ sit alongside cubes/
  training_target.ignore_index
"""
from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml
from tqdm import tqdm


# ── helpers ───────────────────────────────────────────────────────────────────

def dominant_class(mask: np.ndarray, ignore_index: int) -> int | None:
    """Return the most common non-ignored class in a mask tile."""
    valid = mask[mask != ignore_index]
    if valid.size == 0:
        return None
    return int(np.argmax(np.bincount(valid.ravel())))


def _clear_dir(p: Path) -> None:
    if p.exists():
        for f in p.glob("*"):
            if f.is_file():
                f.unlink()


def _resolve(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (root / path)


# ── main ──────────────────────────────────────────────────────────────────────

def run(
    root: Path,
    out: Path,
    train: float,
    val: float,
    seed: int,
    ignore_index: int,
    clean: bool,
) -> None:
    rng = np.random.default_rng(seed)

    img_dir = root / "images"
    msk_dir = root / "masks"
    if not img_dir.exists() or not msk_dir.exists():
        raise FileNotFoundError(f"Expected {img_dir} and {msk_dir}")

    image_files = sorted(img_dir.glob("*.tif"))
    mask_files  = {f.stem: f for f in msk_dir.glob("*.tif")}

    print(f"\n[splits] Source images : {len(image_files)}")
    print(f"[splits] Split ratio   : train={train}  val={val}  test={round(1-train-val, 2)}")
    print(f"[splits] Seed          : {seed}\n")

    # ── assign each tile to a dominant-class bucket ───────────────────────────
    by_class: dict[int, list[str]] = defaultdict(list)
    for img in tqdm(image_files, desc="Reading masks for stratification", unit="tile"):
        msk = mask_files.get(img.stem)
        if msk is None:
            continue
        with rasterio.open(msk) as ds:
            cls = dominant_class(ds.read(1), ignore_index)
        if cls is not None:
            by_class[cls].append(img.stem)

    # ── stratified split per class ────────────────────────────────────────────
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    print("\n[splits] Class distribution:")
    for cls in sorted(by_class):
        ids = list(by_class[cls])
        rng.shuffle(ids)
        n = len(ids)

        n_train = max(1, int(np.floor(n * train))) if n >= 3 else n
        n_val   = max(1, int(np.floor(n * val)))   if n >= 3 else 0
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)
        n_test = n - n_train - n_val

        splits["train"].extend(ids[:n_train])
        splits["val"].extend(ids[n_train : n_train + n_val])
        splits["test"].extend(ids[n_train + n_val :])

        print(f"  class {cls:2d}: total={n:4d}  train={n_train:4d}  val={n_val:4d}  test={n_test:4d}")

    # ── create output dirs, optionally clean ──────────────────────────────────
    for split in ["train", "val", "test"]:
        for sub in ["images", "masks"]:
            d = out / split / sub
            d.mkdir(parents=True, exist_ok=True)
            if clean:
                _clear_dir(d)

    # ── copy tiles ────────────────────────────────────────────────────────────
    rows = []
    total = sum(len(v) for v in splits.values())
    with tqdm(total=total, desc="Copying tiles to splits", unit="tile") as pbar:
        for split, ids in splits.items():
            for stem in ids:
                src_img = img_dir / f"{stem}.tif"
                src_msk = msk_dir / f"{stem}.tif"
                shutil.copy2(src_img, out / split / "images" / src_img.name)
                shutil.copy2(src_msk, out / split / "masks"  / src_msk.name)
                rows.append({"tile_id": stem, "split": split})
                pbar.set_postfix(split=split)
                pbar.update(1)

    # ── write manifest ────────────────────────────────────────────────────────
    manifest = out / "split_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)

    print("\n[splits] Final counts:")
    for split in ["train", "val", "test"]:
        n = len(list((out / split / "images").glob("*.tif")))
        print(f"  {split:5s}: {n} tiles")
    print(f"[splits] Manifest: {manifest}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stratified train/val/test tile split.")
    p.add_argument("--config",       type=Path, help="Path to source_to_target.yaml (preferred)")
    p.add_argument("--root",         type=Path, help="tiles/all dir  (if not using --config)")
    p.add_argument("--out",          type=Path, help="tiles root dir (if not using --config)")
    p.add_argument("--train",        type=float, default=0.8)
    p.add_argument("--val",          type=float, default=0.1)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--ignore-index", type=int,   default=0)
    p.add_argument("--clean",        action="store_true", help="Clear split dirs before writing")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()

    if args.config:
        # ── config-driven (used by clayterractorch CLI) ───────────────────────
        project_root = Path(__file__).resolve().parents[1]
        cfg = yaml.safe_load(args.config.read_text())

        all_dir = project_root / cfg["paths"]["tiles_all_cubes_dir"].replace("/cubes", "")
        out_dir = all_dir.parent  # tiles/
        ignore  = cfg.get("training_target", {}).get("ignore_index", args.ignore_index)

        run(all_dir, out_dir, args.train, args.val, args.seed, ignore, args.clean)

    elif args.root and args.out:
        # ── direct args (manual use) ──────────────────────────────────────────
        if args.train + args.val >= 1.0:
            raise ValueError("train + val must be < 1.0")
        run(args.root, args.out, args.train, args.val, args.seed, args.ignore_index, args.clean)

    else:
        build_parser().error("Provide --config OR both --root and --out.")
