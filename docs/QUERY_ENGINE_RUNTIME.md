# Query Engine runtime

The Query Engine branch is designed to consume a local Batch 1 data package. The
large dataset, SQLite database, FAISS index, and model weights must stay outside
Git.

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
        +--> QA: video/frame candidates -> optional answer extractor
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

The evaluator reports `R@1`, `R@5`, `R@20`, `R@50`, and `R@100`.

## QA answer extraction

`BaselineQueryEngine` accepts an `AnswerExtractor`. When no answer model is
configured it returns an empty answer with `answer_status=model_unavailable` or
`evidence_unavailable`; it never fabricates an answer from the query text.
