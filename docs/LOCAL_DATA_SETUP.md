# Local AIC2026 Data Package

The official dataset is kept outside git. The Query Engine consumes a generated local package built from the project `data/` directory.

## 1. Audit

```powershell
python tools/dataset_audit.py .\data
```

Expected Batch-1 baseline:

- 873 videos
- 177,321 CLIP vectors
- 177,321 mapping rows
- 177,321 keyframes
- 177,321 object files
- zero integrity errors

## 2. Build SQLite + FAISS

From the repository root:

```powershell
python tools/build_data_package.py
```

Generated locally (and ignored by git):

```text
database/aic2026.sqlite
indexes/clip_vit_b32.faiss
indexes/clip_vit_b32.mapping.json
```

The builder uses the mapping CSV as the canonical order. Each CLIP row is mapped to `(video_id, frame_idx)`. Keyframe/object filenames are matched against either `n` or `frame_idx` without changing source `frame_id` semantics.

## 3. Validate the generated package

```powershell
python scripts/validate_batch1.py `
  --db database/aic2026.sqlite `
  --index indexes/clip_vit_b32.faiss `
  --mapping indexes/clip_vit_b32.mapping.json
```

## 4. Run the real Query Engine

Set the API to `clip` mode. The runtime has project-local defaults for the three generated artifacts, so explicit paths are optional.

```powershell
$env:AIC_ENGINE = "clip"
python -m uvicorn api.main:app --reload
```

The first real query loads the CLIP text encoder. The image-text VLM for QA remains optional until it is benchmarked on the AIC QA set.

## Data ownership boundary

`data/` is the local official-data package. `video_pipeline/` owns data preparation. `data_layer/` owns storage access. `query_engine/` must consume data through `DataStore`; it must not hard-code the external `C:\VideoRetrieval-AIC2026\...` path.
