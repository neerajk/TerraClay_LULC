# Source -> Target Pipeline

```mermaid
flowchart TD
    A[AOI + LULC source masks] --> B[01_prepare_lulc_masks.py]
    B --> C[AOI-clipped LULC mask GeoTIFFs]

    C --> D[02_generate_tiles_from_stac.py]
    D --> E[STAC search with filters\ncollection/date/cloud/month/custom query]
    E --> F[Candidate scenes quality checks\nNaN, black pixels, brightness]
    F --> G[Accepted image tile + label tile]

    G --> H[03_make_splits.py\nstratified split by dominant class]
    H --> I[train/val/test image+mask dirs]

    I --> J[04_compute_stats.py\nsync means/stds from metadata.yaml]
    J --> L[TerraTorch config\nSemanticSegmentationTask + GenericNonGeoSegmentationDataModule]
    L --> M[terratorch fit]
    M --> N[checkpoint.ckpt]

    N --> O[terratorch predict / Python tiled inference]
    O --> P[LULC GeoTIFF outputs + probability/confidence layers]
```

## Key Design Decisions

1. Keep STAC sampling logic explicit and configurable in `source_to_target.yaml`.
2. Move training/inference orchestration into TerraTorch YAMLs to reduce custom-code drift.
3. Keep normalization aligned with CLAY metadata by syncing means/stds from `metadata.yaml`.
4. Use tiled inference with overlap/blending by default to reduce patch seams.
