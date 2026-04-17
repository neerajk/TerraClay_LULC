#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


def dominant_class(mask: np.ndarray, ignore_index: int) -> int | None:
    valid = mask[mask != ignore_index]
    if valid.size == 0:
        return None
    cnt = np.bincount(valid.ravel())
    return int(np.argmax(cnt))


def _clear_dir(p: Path) -> None:
    if not p.exists():
        return
    for f in p.glob("*"):
        if f.is_file():
            f.unlink()


def run(root: Path, out: Path, train: float, val: float, seed: int, ignore_index: int, clean: bool) -> None:
    rng = np.random.default_rng(seed)

    img_dir = root / "images"
    msk_dir = root / "masks"
    if not img_dir.exists() or not msk_dir.exists():
        raise FileNotFoundError(f"Expected {img_dir} and {msk_dir}")

    image_files = sorted(img_dir.glob("*.tif"))
    mask_files = {f.stem: f for f in msk_dir.glob("*.tif")}

    by_class: dict[int, list[str]] = defaultdict(list)
    for img in image_files:
        msk = mask_files.get(img.stem)
        if msk is None:
            continue
        with rasterio.open(msk) as ds:
            arr = ds.read(1)
        cls = dominant_class(arr, ignore_index)
        if cls is None:
            continue
        by_class[cls].append(img.stem)

    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    for cls, ids in by_class.items():
        ids = list(ids)
        rng.shuffle(ids)
        n = len(ids)

        n_train = int(np.floor(n * train))
        n_val = int(np.floor(n * val))

        if n >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
            if n_train + n_val >= n:
                n_val = max(1, n - n_train - 1)
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0

        splits["train"].extend(ids[:n_train])
        splits["val"].extend(ids[n_train : n_train + n_val])
        splits["test"].extend(ids[n_train + n_val :])

        print(f"class={cls:2d} count={n:4d} -> train={n_train:4d} val={n_val:4d} test={n_test:4d}")

    for split in ["train", "val", "test"]:
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "masks").mkdir(parents=True, exist_ok=True)
        if clean:
            _clear_dir(out / split / "images")
            _clear_dir(out / split / "masks")

    rows = []
    for split, ids in splits.items():
        for stem in ids:
            src_img = img_dir / f"{stem}.tif"
            src_msk = msk_dir / f"{stem}.tif"
            dst_img = out / split / "images" / src_img.name
            dst_msk = out / split / "masks" / src_msk.name
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_msk, dst_msk)
            rows.append({"tile_id": stem, "split": split})

    manifest = out / "split_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)

    print("\nSplit sizes:")
    for split in ["train", "val", "test"]:
        n = len(list((out / split / "images").glob("*.tif")))
        print(f"  {split}: {n}")
    print(f"Manifest: {manifest}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create stratified train/val/test splits from all tiles.")
    p.add_argument("--root", type=Path, required=True, help="Path to tiles/all")
    p.add_argument("--out", type=Path, required=True, help="Path to tiles root (contains train/val/test)")
    p.add_argument("--train", type=float, default=0.8)
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ignore-index", type=int, default=0)
    p.add_argument("--clean", action="store_true", help="Clear existing split dirs before writing")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.train + args.val >= 1.0:
        raise ValueError("train + val must be < 1.0")
    run(args.root, args.out, args.train, args.val, args.seed, args.ignore_index, args.clean)
