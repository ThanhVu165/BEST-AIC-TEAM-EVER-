# Query Engine — Person 2

This module owns the AI/retrieval core. The design is **model-pluggable**: BTC-provided data/features are inputs, not technology constraints. We may add stronger pretrained models, libraries, or ranking algorithms when they improve the competition objective.

## Canonical pipeline

```text
Natural Language Query
    -> Query normalization / task detection
    -> Semantic query understanding
    -> High-recall candidate generation
         |-- BTC CLIP + FAISS
         |-- optional additional encoders
         |-- metadata / objects / OCR / ASR evidence
    -> Candidate union
    -> Model-based semantic reranking
         |-- SigLIP2 first backend
         |-- VLM / action / relation verifier for Top-K when justified
    -> Temporal localization on original video
    -> Semantic keyframe selection
    -> KIS / Q&A / TRAKE solver
    -> Final ranking / diversification (<=100)
```

## Model selection policy

Do not hard-code a single model as the solution. Each expensive model must be evaluated against the current baseline on the bottleneck it is intended to solve, using retrieval/ranking metrics and inference cost.

| Stage | Baseline | Candidate upgrades |
|---|---|---|
| Candidate generation | BTC CLIP ViT-B/32 + FAISS | SigLIP2 / other dense encoders / multi-index union |
| Semantic reranking | inspectable evidence baseline | **SigLIP2**, then VLM/action/relation models |
| Temporal localization | CLIP source-frame refinement | temporal grounding / moment retrieval models |
| TRAKE alignment | constrained DP | stronger sequence scoring / beam/Viterbi variants |
| Q&A | answer extractor interface | VLMs with frame/window evidence |
| Final ranking | weighted fusion | learned-to-rank / calibrated fusion |

## SigLIP2 backend

`query_engine/semantic_reranker.py` provides a lazy `SigLIP2ImageTextScorer` using:

```text
google/siglip2-base-patch16-256
```

The checkpoint is loaded only when the backend is enabled and first used. SigLIP2 is intended for image-text retrieval and semantic understanding, and its Transformers interface supports retrieval-style text/image scoring.

The full natural-language query is passed to the visual encoder so relations such as `person riding motorcycle` are not reduced to independent object tokens. Structured `entity/action/relation` fields remain inspectable auxiliary information.

### Runtime principle

Do **not** run a heavyweight semantic model over the entire corpus when the BTC CLIP/FAISS index can cheaply produce high-recall candidates. The intended cascade is:

```text
all indexed keyframes
    -> CLIP/FAISS high recall
    -> hundreds of candidates
    -> SigLIP2 semantic rerank
    -> tens of candidates
    -> temporal / VLM verification
    -> final Top-K
```

The exact candidate counts and weights must be benchmarked on the official/local dataset before being frozen.

## First milestone

Build a valid ranked result for every task through the shared `DataStore` interface. Optional model backends must remain replaceable and must not make baseline/CI imports depend on downloading checkpoints.

## Constraints

- Do not read SQLite/FAISS files directly inside retrieval modules.
- Do not change shared schemas without team agreement.
- Keep model implementations behind adapters so they can be replaced.
- Preserve multiple candidates because BTC scores R@1/R@5/R@20/R@50/R@100.
- Keep submission formatting separate from retrieval logic.
- Treat external/reference query datasets as non-official unless BTC later supplies official queries.
