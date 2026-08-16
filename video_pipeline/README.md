# Video Pipeline — Person 1

This module transforms the official AIC data into a stable local data package consumed through `data_layer/` and the shared schemas.

## Batch 1 build

Install the project with the ML and development extras:

```powershell
pip install -e ".[ml,dev]"
```

Then point the builder at the local Batch 1 root:

```powershell
python -m video_pipeline.build_dataset `
  --data-root .\data\raw `
  --database .\database\aic2026.sqlite `
  --index-root .\indexes
```

The command:

1. scans videos, keyframe images, mapping CSVs, CLIP `.npy` files, media-info JSONs and object JSONs;
2. validates the Batch-1 artifact counts/alignment and writes `indexes/manifest.json`;
3. builds SQLite tables for videos, keyframes, objects and metadata;
4. builds a normalized CLIP `IndexFlatIP` and `frame_mapping.json`;
5. writes index metadata describing the vector metric and dimensionality.

## Correctness rules

- `frame_id` is the original/source frame identifier used for competition output.
- The mapping CSV is authoritative for source `frame_id`.
- A keyframe ordinal is never silently substituted for `frame_id`.
- CLIP row order must exactly match mapping row order.
- Every FAISS internal id maps deterministically to one `(video_id, frame_id)`.
- Missing optional metadata is recorded as missing; it is never fabricated.
- Raw data and generated SQLite/FAISS artifacts remain outside Git.

## Expected local layout

The builder is intentionally tolerant of the exact subdirectory names because the downloaded BTC package may be organized differently. It discovers artifact files recursively and matches per-video artifacts by filename stem.

A typical layout is:

```text
data/raw/
  videos/
    <video_id>.mp4
  keyframes/
    <video_id>/*.jpg
  objects/
    <video_id>/*.json
  clip_features/
    <video_id>.npy
  metadata/
    <video_id>.json
  mappings/
    <video_id>.csv
```

The mapping CSV must expose a source frame identifier (`frame_id`, `source_frame_id`, or an equivalent documented field). If it does not, validation fails rather than guessing.

## Validation

Run the unit tests before ingesting the full dataset:

```powershell
pytest
ruff check .
```

The full Batch-1 build must be run on the machine containing the dataset because CI deliberately has no access to competition data.
