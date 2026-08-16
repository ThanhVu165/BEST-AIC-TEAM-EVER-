# Batch 1 Dataset Audit

> Status: observational audit only. This document records verified properties of the existing AIC 2026 Batch 1 artifacts. It does **not** authorize architectural changes or claim official retrieval accuracy.

## 1. Source dataset used for the audit

The audit was performed against the existing local AIC 2026 dataset:

```text
C:\VideoRetrieval-AIC2026\VideoRetriaval-AIC2026\data\
```

The dataset contains these top-level directories:

```text
data/
├── clip/
├── keyframes/
├── mapping/
├── media_info/
├── objects/
├── queries/
└── videos/
```

`ground_truth.json` exists in the local dataset, but it is **not to be used as an evaluation source** for the current work.

The official query artifact currently observed is:

```text
queries/DanhSachTruyVanAIC_Chungket.xlsx
```

## 2. Verified CLIP artifact

Example:

```text
clip/clip-features-32/L21_V001.npy
```

Verified properties:

```text
shape = (307, 512)
dtype = float16
ndim  = 2
```

Therefore, for this video, the supplied CLIP artifact contains 307 frame/keyframe-level vectors of dimension 512.

Do not infer a fixed number of keyframes per video. `N` is video-dependent.

## 3. Verified keyframe mapping schema

Example:

```text
mapping/map-keyframes/L21_V001.csv
```

Header:

```text
n,pts_time,fps,frame_idx
```

Example records:

```text
1,0.0,30.0,0
2,3.0,30.0,90
3,8.7,30.0,261
4,11.7333,30.0,351
5,13.7,30.0,411
6,17.7,30.0,531
7,23.7,30.0,711
8,28.6,30.0,858
9,30.7,30.0,921
```

Important observations:

- `n` is the keyframe sequence/index within the video.
- `pts_time` provides the timestamp associated with the sampled keyframe.
- `fps` is provided explicitly.
- `frame_idx` is the source/original frame index.
- Sampling intervals are **not fixed**. Do not assume a constant number of seconds between consecutive keyframes.
- Competition frame output must use the validated source-frame convention, not an invented index convention.

A likely relationship is `CLIP[i] -> mapping row n=i+1`, but this must be verified by an integrity check before being treated as a hard invariant across the dataset.

## 4. Verified media metadata

Example:

```text
media_info/media-info/L21_V001.json
```

Observed fields include:

- `author`
- `channel_id`
- `channel_url`
- `description`
- `keywords`
- `length`
- `publish_date`
- `thumbnail_url`
- `title`
- `watch_url`

This means the supplied metadata can support a textual/metadata retrieval channel in addition to visual retrieval.

However, metadata should not be assumed to uniquely identify the correct frame. Videos from the same source/program may have highly similar titles, descriptions, and keywords. Metadata is therefore best treated as candidate-generation or reranking evidence rather than as a replacement for frame-level visual/temporal evidence.

## 5. Verified object-detection artifact

Example:

```text
objects/objects/L21_V001/001.json
```

Observed fields include:

- `detection_scores`
- `detection_class_names`
- `detection_class_labels`
- `detection_boxes`

The artifact contains many detections for a frame. Example semantic labels observed include `Lantern`, `Skyscraper`, `Poster`, `Tower`, `Building`, `Vehicle`, `Balloon`, `Boat`, `Car`, etc.

The object data should be treated as auxiliary semantic evidence. It is not sufficient by itself to represent complex event semantics or temporal relationships.

The correspondence between `objects/.../001.json` and mapping `n=1` must be explicitly verified before it becomes a system invariant.

## 6. Expected multimodal relationship

The observed artifacts support the following conceptual data flow:

```text
video_id
   │
   ├── CLIP .npy
   │      └── vector[i]
   │
   ├── mapping CSV
   │      └── n -> pts_time -> frame_idx
   │
   ├── keyframes
   │      └── visual frame
   │
   ├── object JSON
   │      └── semantic object evidence
   │
   ├── media-info JSON
   │      └── title / description / keywords / metadata
   │
   └── source video
          └── exact temporal verification
```

The intended query-side concept remains:

```text
Natural-language query
        │
        ├── visual retrieval (CLIP)
        ├── metadata retrieval
        └── optional object/semantic evidence
                │
                ▼
        candidate generation
                │
                ▼
        fine-grained frame retrieval / reranking
                │
                ▼
        temporal localization
                │
                ▼
        semantic keyframe selection
```

This is a data-driven observation, not a mandate to replace the existing architecture.

## 7. Evaluation policy

The current audit must not use the local `ground_truth.json` as official AIC 2026 evaluation ground truth.

Therefore:

- Do not calculate official R@1/R@5/R@20/R@50/R@100 from that file.
- Do not report a retrieval score as an official benchmark unless the evaluation source is explicitly supplied/authorized by BTC.
- Qualitative retrieval inspection is allowed.
- Data-integrity tests are allowed and should be performed independently of ground truth.

Recommended immediate validation before any quality benchmark:

```text
CLIP vector count
        ==
mapping row count
        ==
keyframe count
        ==
object-file count (where object artifacts are expected)
```

and verify for sampled videos:

```text
CLIP index
   -> mapping n
   -> frame_idx / timestamp
   -> keyframe
   -> object artifact
```

## 8. Engineering implications

The following are currently established working assumptions:

1. CLIP features are supplied per video as `.npy` matrices with 512-dimensional vectors in the inspected sample.
2. Mapping CSVs provide explicit timestamp and source-frame information.
3. Keyframe sampling is irregular; temporal code must use the supplied mapping rather than a fixed sampling interval.
4. Metadata is available per video and can be indexed independently from visual features.
5. Object detections are available per frame/keyframe and can be used as auxiliary semantic evidence.
6. Large feature arrays should remain outside SQLite or other relational storage; only indexes/mappings/metadata should be stored in structured stores where appropriate.
7. Query Engine should consume stable data-access interfaces rather than depending on concrete storage implementation details.
8. Missing auxiliary artifacts must be handled explicitly; they must not be silently fabricated.
9. The existing high-level three-part architecture (`video_pipeline`, `query_engine`, `ui`) is not changed by this audit.

## 9. Not yet verified

The following must be checked before being encoded as hard invariants:

- CLIP row `i` exactly corresponds to mapping row `n=i+1` for all videos.
- Number of mapping rows equals the number of CLIP vectors for all videos.
- Number and naming of keyframe files exactly match mapping `n`.
- Object JSON numbering exactly matches mapping `n`.
- Source `frame_idx` can always be resolved back to the corresponding video frame.
- All videos have complete auxiliary artifacts.
- The exact structure of the query workbook and its task-specific fields.
- Any official evaluation interface/data supplied separately by BTC.

These checks belong to the data-integrity/benchmark phase and must not be assumed from a single sample.

## 10. Guidance for AI coding agents

When modifying this repository:

- Treat this document as an audit of observed data, not as permission to redesign the system.
- Do not change shared schemas solely because of an observation in this document.
- Do not invent ground truth or evaluation labels.
- Do not assume fixed keyframe sampling intervals.
- Do not assume object file numbering is aligned with CLIP/mapping until integrity tests prove it.
- Prefer adapters and explicit mappings between raw BTC artifacts and internal contracts.
- Keep the official video as the primary competition data and treat keyframes, objects, CLIP features, and metadata as supporting artifacts.
- Baseline and validate data integrity before adding research optimizations.
