#!/usr/bin/env python3
from __future__ import annotations


def main() -> None:
    from terratorch import BACKBONE_REGISTRY

    names = sorted(list(BACKBONE_REGISTRY))
    print(f"Total backbones: {len(names)}")
    print("\nAll:")
    for n in names:
        print(f"  {n}")

    clay = [n for n in names if "clay" in n.lower()]
    print("\nCLAY-like names:")
    for n in clay:
        print(f"  {n}")


if __name__ == "__main__":
    main()
