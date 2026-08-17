# Query Engine runtime

The Query Engine branch consumes a local Batch 1 data package. The large
video dataset, SQLite database, FAISS index, and model weights stay outside Git.

## Canonical runtime boundary

```text
Natural-language / structured query
        |
        v
Query Understanding
        |
        v
BTC CLIP candidate retrieval
        |
        v
Multimodal evidence collection
        |
        v
Video-level aggregation / ranking
        |
        v
Temporal localization
        |
        v
Semantic keyframe alignment
        |
        +--> KIS
        +--> QA
        +--> TRAKE
        |
        v
Final candidate ranking
        |
        v
Top-100
```

The current Batch 1 implementation provides the retrieval, evidence and
sequence-alignment baseline. Fine-grained temporal grounding on original video
frames and learned semantic verification remain explicit research stages; the
system must not represent the sparse-keyframe proxy as equivalent to exact BTC
temporal localization.

## Batch 1 integration

The runtime expects:

- SQLite database containing `videos`, `frames`, `objects`, `metadata`, and
  optional `ocr`/`asr_segments`
- FAISS frame index
- JSON mapping whose array position is the deterministic FAISS internal ID
- Every mapping entry to preserve `video_id`, `keyframe_n`, and original
  source `frame_id`

Validate these artifacts before running retrieval. The local data package is
not uploaded by the runtime.

## Retrieval design

BTC-provided CLIP ViT-B/32 remains the primary dense retrieval signal. Auxiliary
signals are collected independently:

- Objects: entity evidence only
- Metadata: video-level textual evidence
- OCR: frame-level textual evidence
- ASR: video-level speech evidence

The weights are explicit runtime parameters so each signal can be ablated and
benchmarked rather than treated as a fixed BTC requirement.

## Temporal design

The current temporal module has two distinct roles:

1. **Proxy selection** — rank already retrieved source-frame hypotheses.
2. **TRAKE sequence alignment** — use dynamic programming to preserve ordered
event frame progression.

This is not yet fine temporal localization. Exact event grounding must later
search/refine the original video around coarse candidate windows, especially
for TRAKE intervals that can be narrower than 10 source frames.

## QA answer extraction

`BaselineQueryEngine` requires a real `AnswerExtractor` for competition-grade
Q&A. Without one, it returns an explicit `model_unavailable` status and never
fabricates an answer. `TransformersImageAnswerExtractor` is a configurable
benchmark adapter; no model is assumed to be the final competition default
until measured on the actual task queries.

## Evaluation

Engineering evaluation should report:

`R@1`, `R@5`, `R@20`, `R@50`, `R@100`, and `Final Score`.

When authorized task-level reference annotations are available, use the
competition-aligned evaluator. Local `ground_truth.json` must not be silently
treated as official BTC evaluation ground truth.
