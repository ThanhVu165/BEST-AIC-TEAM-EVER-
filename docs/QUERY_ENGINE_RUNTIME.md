# Query Engine runtime

The Query Engine branch consumes a local Batch 1 data package. The large
video dataset, SQLite database, FAISS index, and model weights stay outside Git.

## Runtime boundary

```text
Natural-language query
        |
        v
CLIP text encoder
        |
        v
FAISS frame retrieval
        |
        +--> KIS: frame hypotheses -> semantic keyframe proxy
        |
        +--> QA: video/frame candidates -> optional VLM answer extractor
        |
        +--> TRAKE: event-wise retrieval -> common-video filtering -> one frame/event
```

The current temporal stage is intentionally conservative. It ranks retrieved
source frames and never invents a frame ID or event boundary. A learned temporal
grounder can replace that stage later.

## Batch 1 integration

The runtime expects three local artifacts:

- SQLite database containing `videos` and `frames` (and optionally `objects`)
- FAISS frame index
- JSON mapping whose array position is the deterministic FAISS internal ID

Every mapping entry must contain `video_id` and the original `frame_id`.

Validate these artifacts before running retrieval:

```powershell
python scripts/validate_batch1.py `
  --db .\database\aic2026.sqlite `
  --index .\indexes\clip\frame.faiss `
  --mapping .\indexes\clip\frame_mapping.json
```

The validator checks FAISS/index alignment and that every mapped source frame
exists in SQLite. It reads local data only and does not upload anything.

## Evaluation

Prepare a JSONL file where each row is a `QueryRequest` payload plus either
`relevant_video_ids` or `relevant_video_id`.

```powershell
python scripts/evaluate_queries.py `
  --queries .\local\queries.jsonl `
  --db .\database\aic2026.sqlite `
  --index .\indexes\clip\frame.faiss `
  --mapping .\indexes\clip\frame_mapping.json
```

The evaluator reports `R@1`, `R@5`, `R@20`, `R@50`, `R@100`, and their mean as a
video-level retrieval `FinalScore`. This is an engineering baseline, not a
claim that it reproduces the official task evaluator.

## API runtime

The FastAPI service defaults to the mock engine for UI/contract tests.
To use the real local retrieval runtime, configure:

```text
AIC_ENGINE=clip
AIC_DB_PATH=<local SQLite path>
AIC_CLIP_INDEX=<local FAISS index>
AIC_CLIP_MAPPING=<local mapping JSON>
AIC_CLIP_MODEL=openai/clip-vit-base-patch32
AIC_DEVICE=auto
```

For optional QA VLM inference, additionally set:

```text
AIC_VLM_MODEL=<benchmarked image-text model>
AIC_VLM_DEVICE=auto
```

No VLM is selected as the competition default until it is benchmarked on the
actual AIC QA queries.

## QA answer extraction

`BaselineQueryEngine` accepts an `AnswerExtractor`. Without a configured model
it returns an explicit unavailable status and never fabricates an answer from
query text alone. `TransformersImageAnswerExtractor` provides a configurable
Transformers adapter for local model benchmarking.
