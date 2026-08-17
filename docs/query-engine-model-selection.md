# Query Engine Model Selection

The Query Engine is deliberately model-pluggable. BTC-provided features are official input/evidence, not a technology restriction.

## Cascade

```text
Natural-language query
        |
        v
High-recall CLIP/FAISS retrieval
        |
        v
Candidate union + evidence
        |
        v
SigLIP2 image-text semantic reranking
        |
        v
Source-frame temporal refinement
        |
        v
Bounded VideoWindow
        |
        +------------------------------+
        |                              |
        v                              v
Qwen3-VL action/relation        InternVideo3 video/event
verification                    verification
        |                              |
        +--------------+---------------+
                       v
                task-specific solver
                  /      |      \
                KIS      QA     TRAKE
                       |
                       v
                 final ranking
                       |
                     Top-100
```

## Model policy

1. **Candidate generation:** optimize recall and latency. Keep the BTC CLIP/FAISS index as the first-stage anchor because its embeddings and index are already available.
2. **Semantic reranking:** use a stronger image-text model such as SigLIP2 when it improves hard-negative discrimination. The full natural-language query is preserved so action/relation wording is not discarded.
3. **Video verification:** use a video-language model only on a small candidate set. `VideoWindow` makes the temporal interval explicit instead of passing an entire source video to an expensive verifier.
4. **Action/relation verification:** use a capable VLM when object co-occurrence is insufficient to distinguish relations such as `riding`, `repairing`, `pushing`, or `standing near`.
5. **Temporal localization:** a model upgrade is accepted only if it improves event/frame localization on the target task; the current source-frame CLIP refinement remains the deterministic fallback.
6. **Final ranking:** model scores are evidence. They must be fused once and their weights must be benchmarked against the official R@K objective.

## Benchmark gate

No new model becomes the production default solely because it is newer or stronger on a public benchmark. Compare:

- R@1 / R@5 / R@20 / R@50 / R@100
- hard-negative pair accuracy
- positive-vs-negative score margin
- temporal hit/localization accuracy where labels are available
- latency, VRAM and preprocessing cost

The preferred system is the best end-to-end trade-off, not the largest model.

## Current implementation status

- CLIP/FAISS candidate retrieval: implemented.
- SigLIP2 adapter and late-stage frame reranking: implemented, opt-in until benchmarked.
- Source-video frame access: implemented in `DataStore`.
- Bounded `VideoWindow`: implemented.
- Qwen3-VL adapter: implemented, opt-in.
- InternVideo3 adapter: implemented, opt-in.
- Production VLM/video-model fusion: intentionally gated on benchmark data and runtime validation.
