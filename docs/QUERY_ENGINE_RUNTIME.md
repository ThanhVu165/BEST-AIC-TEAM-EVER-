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

The current Batch 1 implementation provides retrieval, inspectable multimodal
evidence, deterministic ranking, source-frame temporal refinement and TRAKE
sequence alignment. Learned temporal grounding and learned semantic
verification remain explicit research stages; the CLIP source-frame proxy must
not be represented as equivalent to a learned BTC temporal localizer.

## Batch 1 integration

The runtime expects:

- SQLite database containing `videos`, `frames`, `objects`, `metadata`, and
  optional `ocr`/`asr_segments`
- FAISS frame index
- JSON mapping whose array position is the deterministic FAISS internal ID
- Every mapping entry to preserve `video_id`, `keyframe_n`, and original
  source `frame_id`
- Original videos reachable through `VideoRecord.path` for fine source-frame
  refinement

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

The temporal module has three roles:

1. **Sparse proxy selection** — rank retrieved keyframe/source-frame hypotheses.
2. **Source-frame refinement** — for a small number of top anchors, decode a
   configurable neighborhood from the original video and compare each frame
   with the query using the same CLIP embedding space. This can return a
   non-keyframe `frame_id`.
3. **TRAKE sequence alignment** — use dynamic programming to preserve strict
   ordered event progression across the selected event frames.

The source-frame stage is a strong deterministic CLIP proxy, not a learned
fine temporal grounder. Its `radius` and anchor count are runtime parameters
and must be benchmarked on the actual AIC queries. This distinction matters
for TRAKE intervals that can be narrower than 10 source frames.

## QA answer extraction

`BaselineQueryEngine` requires a real `AnswerExtractor` for competition-grade
Q&A. Without one, it returns an explicit `model_unavailable` status and never
fabricates an answer. `TransformersImageAnswerExtractor` can consume either a
keyframe path or a decoded source-video frame; no model is assumed to be the
final competition default until measured on the actual task queries.

## Evaluation

Engineering evaluation should report:

`R@1`, `R@5`, `R@20`, `R@50`, `R@100`, and `Final Score`.

When authorized task-level reference annotations are available, use the
competition-aligned evaluator. Local `ground_truth.json` must not be silently
treated as official BTC evaluation ground truth.
