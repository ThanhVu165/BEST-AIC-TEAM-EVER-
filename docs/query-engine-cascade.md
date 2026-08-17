# Query Engine Model Cascade

## Objective

Use the strongest practical model for each bottleneck without forcing an expensive model to scan the full corpus.

## Canonical cascade

```text
Natural-language query
        |
        v
Query understanding / semantic decomposition
        |
        v
High-recall candidate generation
  - BTC CLIP embeddings
  - FAISS
  - object / metadata / OCR / ASR evidence
        |
        v
Image-text semantic reranking
  - SigLIP2
        |
        v
Fine temporal localization on original video
        |
        v
Bounded VideoWindow
        |
        v
Late video verification
  - InternVideo3
  - Qwen3-VL
        |
        v
Task-specific solver
  - KIS: frame ranking
  - QA: evidence + answer extraction
  - TRAKE: ordered event alignment
        |
        v
Final ranking / Top-k
```

## Model selection policy

A model is not promoted to the default production path merely because it is newer or larger. Every expensive backend must have:

1. a bounded candidate stage;
2. an explicit adapter contract;
3. unit/integration tests with mocked inference;
4. a benchmark comparing it with the current backend;
5. an observed accuracy/latency/memory trade-off.

## Ranking contract

The final frame score has independent late-stage evidence channels:

- retrieval;
- object;
- metadata;
- OCR;
- ASR;
- temporal localization;
- semantic image-text score;
- video-window verification score.

Semantic and video-verification weights are explicit. The remaining canonical evidence is renormalized so adding a backend does not silently double-count evidence.

## Compute policy

Do not run large VLM/video-language models over the corpus. The intended compute pattern is:

```text
large corpus
  -> cheap high-recall retrieval
  -> small semantic rerank pool
  -> small temporal pool
  -> very small video-verification pool
```

The exact candidate limits and weights are configuration/benchmark parameters, not permanent assumptions.

## Current status

- CLIP/FAISS candidate generation: implemented.
- SigLIP2 adapter and candidate-frame scoring: implemented.
- Original-video temporal refinement: implemented as a CLIP-based proxy.
- VideoWindow abstraction: implemented.
- InternVideo3 verifier adapter: implemented.
- Qwen3-VL verifier adapter: implemented.
- Late verification helper: implemented.
- Direct production wiring of expensive video verification into `BaselineQueryEngine`: intentionally gated until its frame/window-to-ranking contract is benchmarked.
- Learned temporal grounding: still an experiment target.
- Learning-to-rank: still an experiment target.
