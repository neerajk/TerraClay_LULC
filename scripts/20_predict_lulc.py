#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path
from typing import Any

import yaml


def _resolve(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (root / path)


def _resolve_exec(root: Path, value: str) -> str:
    # Keep bare executables (e.g. "python") untouched for portable configs.
    if "/" not in value and not value.startswith('.'):
        return value
    return str(_resolve(root, value))


def _pick(val: Any, runtime: str):
    if isinstance(val, dict):
        return val.get(runtime)
    return val


def _cmd_str(cmd: list[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def build_legacy_command(root: Path, profile: dict[str, Any], runtime_cfg: dict[str, Any], runtime: str, scene_tif: Path, out_tif: Path, conf_tif: Path | None) -> list[str]:
    py = _resolve_exec(root, profile.get("python", "python"))
    script = str(_resolve(root, profile["script"]))
    clay_ckpt = str(_resolve(root, profile["clay_ckpt"]))
    decoder_ckpt = str(_resolve(root, profile["decoder_ckpt"]))
    metadata_path = str(_resolve(root, profile["metadata_path"]))

    batch_size = int(_pick(profile["batch_size"], runtime))

    cmd = [
        py,
        script,
        "--scene-tif",
        str(scene_tif),
        "--out-tif",
        str(out_tif),
        "--decoder-ckpt",
        decoder_ckpt,
        "--clay-ckpt",
        clay_ckpt,
        "--metadata-path",
        metadata_path,
        "--platform",
        str(profile.get("platform", "landsat-c2-l2")),
        "--band-indices",
        str(profile.get("band_indices", "1,2,3,4,5,6")),
        "--tile-size",
        str(profile.get("tile_size", 256)),
        "--stride",
        str(profile.get("stride", 128)),
        "--batch-size",
        str(batch_size),
    ]

    if conf_tif is not None:
        cmd.extend(["--confidence-tif", str(conf_tif)])

    cpu_only = bool(profile.get("cpu_only_legacy", runtime_cfg.get("cpu_only_legacy", False)))
    if cpu_only:
        cmd.append("--cpu-only")

    return cmd


def build_terratorch_large_scene_command(root: Path, profile: dict[str, Any], runtime_cfg: dict[str, Any], runtime: str, scene_tif: Path, out_tif: Path) -> list[str]:
    py = _resolve_exec(root, profile.get("python", "python"))
    script = str(root / "scripts" / "12_predict_large_scene.py")
    config = str(_resolve(root, profile["config"]))
    ckpt = str(_resolve(root, profile["ckpt"]))

    batch_size = int(_pick(profile["batch_size"], runtime))
    crop = int(profile.get("crop", 224))
    stride = int(profile.get("stride", 192))
    device = str(profile.get("device", runtime_cfg.get("device", "auto")))

    return [
        py,
        script,
        "--config",
        config,
        "--ckpt",
        ckpt,
        "--input-tif",
        str(scene_tif),
        "--out-tif",
        str(out_tif),
        "--crop",
        str(crop),
        "--stride",
        str(stride),
        "--batch-size",
        str(batch_size),
        "--device",
        device,
    ]


def run(args) -> int:
    root = Path(__file__).resolve().parents[1]
    profiles_path = _resolve(root, args.profiles)
    cfg = yaml.safe_load(profiles_path.read_text())

    model_name = args.model or cfg["default_model"]
    runtime = args.runtime or cfg.get("default_runtime", "mac")

    models = cfg["models"]
    if model_name not in models:
        raise KeyError(f"Model '{model_name}' not found in {profiles_path}")

    if runtime not in cfg.get("runtime_profiles", {}):
        raise KeyError(f"Runtime '{runtime}' not found in runtime_profiles")

    profile = models[model_name]
    engine = profile["engine"]
    runtime_cfg = cfg.get("runtime_profiles", {}).get(runtime, {})

    scene_tif = _resolve(root, args.scene_tif)
    if args.out_tif:
        out_tif = _resolve(root, args.out_tif)
    else:
        out_dir = _resolve(root, cfg.get("default_output_dir", "outputs/predictions"))
        out_tif = out_dir / f"{scene_tif.stem}.{model_name}.lulc.tif"

    if not args.dry_run:
        out_tif.parent.mkdir(parents=True, exist_ok=True)

    conf_tif = _resolve(root, args.confidence_tif) if args.confidence_tif else None

    if engine == "legacy_clay_lulc":
        cmd = build_legacy_command(root, profile, runtime_cfg, runtime, scene_tif, out_tif, conf_tif)
    elif engine == "terratorch_large_scene":
        cmd = build_terratorch_large_scene_command(root, profile, runtime_cfg, runtime, scene_tif, out_tif)
    else:
        raise ValueError(f"Unsupported engine: {engine}")

    print(f"model={model_name} | runtime={runtime} | engine={engine}")
    print("command:")
    print(_cmd_str(cmd))

    if args.dry_run:
        return 0

    proc = subprocess.run(cmd)
    return proc.returncode


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="One-command LULC prediction using model profiles config.")
    p.add_argument("--profiles", default="configs/model_profiles.yaml", help="Model profiles yaml")
    p.add_argument("--model", default=None, help="Model key in model_profiles.yaml (default uses default_model)")
    p.add_argument("--runtime", default=None, choices=["mac", "gpu", "cloud"], help="Runtime profile")
    p.add_argument("--scene-tif", required=True, help="Input stacked scene tif")
    p.add_argument("--out-tif", default=None, help="Output LULC tif (default auto path from profiles config)")
    p.add_argument("--confidence-tif", default=None, help="Optional confidence tif (legacy engine only)")
    p.add_argument("--dry-run", action="store_true", help="Print command only")
    return p


if __name__ == "__main__":
    parser = build_parser()
    ns = parser.parse_args()
    raise SystemExit(run(ns))
