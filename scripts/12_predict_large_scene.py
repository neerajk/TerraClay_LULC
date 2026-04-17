#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
import rioxarray
import torch
import yaml
from terratorch.cli_tools import LightningInferenceModel
from terratorch.tasks.tiled_inference import tiled_inference


def _resolve_ref(path_or_value: str, project_root: Path) -> Path:
    p = Path(path_or_value)
    return p if p.is_absolute() else (project_root / p)


def _load_vector(value, project_root: Path, key: str) -> np.ndarray:
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float32)
    if isinstance(value, str):
        p = _resolve_ref(value, project_root)
        if not p.exists():
            raise FileNotFoundError(f"{key} file not found: {p}")
        arr = np.genfromtxt(p)
        arr = np.atleast_1d(arr).astype(np.float32)
        return arr
    raise TypeError(f"Unsupported type for {key}: {type(value)}")


def load_means_stds(config_path: Path) -> tuple[np.ndarray, np.ndarray]:
    project_root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(config_path.read_text())
    init = cfg["data"]["init_args"]

    means = _load_vector(init["means"], project_root, "means")
    stds = _load_vector(init["stds"], project_root, "stds")

    if means.shape != stds.shape:
        raise ValueError(f"means/stds length mismatch: {means.shape} vs {stds.shape}")
    return means, stds


def run(
    config: Path,
    ckpt: Path,
    input_tif: Path,
    out_tif: Path,
    crop: int,
    stride: int,
    batch_size: int,
    device: str,
) -> None:
    means, stds = load_means_stds(config)

    model = LightningInferenceModel.from_config(str(config), str(ckpt))
    model.eval()

    if device == "cuda" and torch.cuda.is_available():
        dev = torch.device("cuda")
    elif device == "mps" and torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")

    model.to(dev)

    xda = rioxarray.open_rasterio(input_tif)
    arr = xda.values.astype(np.float32)
    arr = (arr - means[:, None, None]) / stds[:, None, None]

    x = torch.from_numpy(arr).unsqueeze(0)  # [1,C,H,W]

    def model_forward(inp: torch.Tensor, **kwargs) -> torch.Tensor:
        out = model(inp.to(dev), **kwargs)
        if hasattr(out, "output"):
            return out.output
        return out

    with torch.no_grad():
        logits = tiled_inference(
            model_forward,
            x,
            crop=crop,
            stride=stride,
            batch_size=batch_size,
            verbose=True,
        )

    pred = logits.squeeze(0).argmax(dim=0).cpu().numpy().astype(np.uint8)

    out_tif.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(input_tif) as src:
        profile = src.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=0, compress="lzw")
        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(pred, 1)

    print(f"Saved prediction: {out_tif}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run TerraTorch tiled inference on a large stacked scene.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--input-tif", type=Path, required=True)
    p.add_argument("--out-tif", type=Path, required=True)
    p.add_argument("--crop", type=int, default=224)
    p.add_argument("--stride", type=int, default=192)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    dev = args.device
    if dev == "auto":
        if torch.cuda.is_available():
            dev = "cuda"
        elif torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"

    run(
        config=args.config,
        ckpt=args.ckpt,
        input_tif=args.input_tif,
        out_tif=args.out_tif,
        crop=args.crop,
        stride=args.stride,
        batch_size=args.batch_size,
        device=dev,
    )
