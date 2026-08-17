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
         |-- SigLIP2 image-text alignment
         |-- optional VLM action/relation verification
         |-- optional video-text model verification
    -> Temporal localization on original video
    -> Semantic keyframe selection
    -> KIS / Q&A / TRAKE solver
    -> Final ranking / diversification (<=100)
```

## Model selection policy

Do not hard-code a single model as the solution. Each expensive model must be evaluated against the current baseline on the bottleneck it is intended to solve, using retrieval/ranking metrics and inference cost.

| Stage | Baseline | Candidate upgrades |
|---|---|---|
| Query understanding | structured parser | sentence encoders / LLM decomposition / semantic paraphrases |
| Candidate generation | BTC CLIP ViT-B/32 + FAISS | SigLIP2 / other dense encoders / multi-index union |
| Semantic reranking | inspectable evidence baseline | **SigLIP2**, larger SigLIP2 variants |
| Action/relation verification | object evidence | **Qwen2.5-VL-7B-Instruct**, other VLM/action models |
| Video-text reranking | frame-level scoring | **InternVideo2** 1B retrieval models |
| Temporal localization | CLIP source-frame refinement | temporal grounding / moment retrieval models |
| TRAKE alignment | constrained DP | stronger sequence scoring / beam/Viterbi variants |
| Q&A | answer extractor interface | VLMs with frame/window evidence |
| Final ranking | weighted fusion | learned-to-rank / calibrated fusion |

## SigLIP2 backend

`query_engine/semantic_reranker.py` provides a lazy `SigLIP2ImageTextScorer` using:

```text
google/siglip2-base-patch16-256
```

SigLIP2 is intended for image-text retrieval and semantic understanding. The full natural-language query is passed to the visual encoder so relations such as `person riding motorcycle` are not reduced to independent object tokens. Structured `entity/action/relation` fields remain inspectable auxiliary information.

A larger checkpoint can be selected through configuration after benchmarking. SigLIP2 exposes multiple model sizes, so the implementation deliberately does not hard-code the smallest checkpoint as the permanent solution.

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

## VLM action/relation verification

`query_engine/vlm_verifier.py` provides a lazy `Qwen25VLVerifier` backend using:

```text
Qwen/Qwen2.5-VL-7B-Instruct
```

It is intentionally a **late verifier**, not a corpus-wide retriever. The prompt explicitly asks the VLM to focus on actions and relations rather than object presence. This is designed for hard negatives such as:

```text
person riding motorcycle
    vs.
person standing beside motorcycle

person repairing bicycle
    vs.
person riding bicycle
```

The verifier must only be applied to a small candidate set because a 7B VLM is materially more expensive than a dense encoder. Its output is normalized to `[0, 1]` and can be introduced into final fusion after benchmark validation.

Qwen2.5-VL also supports video inputs and temporal reasoning, so it remains a candidate for the later Q&A / temporal-verification stage rather than being restricted to single-frame verification.

## Video-level model research

InternVideo2 is tracked as a separate video-text candidate because its official multi-modality implementation provides retrieval modes and 1B checkpoints. It should be evaluated as a **video/window reranker** rather than replacing the BTC FAISS index. Its English-only `1B-s2` branch and multilingual `1B-clip` branch have different text-encoder behavior; the choice must follow the query language distribution and benchmark results.

The project does not vendor the InternVideo2 codebase. Keep it behind an adapter and make it an optional dependency only if an experiment demonstrates a measurable retrieval gain.

## Hard-negative evaluation

`tools/semantic_hard_negative_benchmark.py` defines a model-agnostic benchmark protocol:

```text
positive image-text pair
        vs.
hard-negative image-text pair
```

Primary metrics:

- pairwise accuracy
- mean positive-minus-negative score margin

Example hard negatives should specifically target action/relation confusion, not only object mismatch. The benchmark is intended to compare CLIP, SigLIP2 variants, and future semantic backends before changing production weights.

## Model governance

Every new model follows this sequence:

```text
research candidate
    -> adapter
    -> CI-safe lazy loading
    -> hard-negative benchmark
    -> R@1/R@5/R@20/R@50/R@100 evaluation
    -> latency / VRAM measurement
    -> decision
```

A model is **not** considered production-ready merely because its paper reports strong benchmark numbers.

## First milestone

Build a valid ranked result for every task through the shared `DataStore` interface. Optional model backends must remain replaceable and must not make baseline/CI imports depend on downloading checkpoints.

## Constraints

- Do not read SQLite/FAISS files directly inside retrieval modules.
- Do not change shared schemas without team agreement.
- Keep model implementations behind adapters so they can be replaced.
- Preserve multiple candidates because BTC scores R@1/R@5/R@20/R@50/R@100.
- Keep submission formatting separate from retrieval logic.
- Treat external/reference query datasets as non-official unless BTC later supplies official queries.
