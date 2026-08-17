# AIC 2026 — Canonical System Pipeline

**Status:** Canonical engineering direction for this repository.

This document is the single pipeline reference for AI-assisted development. Existing implementations are components of this pipeline; they are not alternative pipelines. Do not introduce a second end-to-end architecture unless the team explicitly replaces this document.

## 1. Authority hierarchy

1. **BTC official preliminary-round document** — authoritative for task semantics, scoring and submission constraints.
2. **Team-verified dataset audit** — authoritative for what is actually present in the local Batch 1 package.
3. **This document** — canonical engineering architecture derived from the two sources above.
4. Research papers, AIC2025 reports, baselines and experiments — references/hypotheses only. They may motivate an implementation, but they do not redefine BTC requirements.

In particular, the supplementary baseline report mentioning SigLIP, BEiT3, Qdrant, Elasticsearch, Whisper, Gemini OCR, BLIP-2, etc. is **reference material**, not a list of mandatory BTC components. A model or data source is adopted only after verification and benchmarking.

## 2. Non-negotiable system objective

The system is a **ranked video retrieval and temporal understanding system**, not a classifier and not a single-best predictor.

```text
Natural-language query / structured query
                |
                v
        Query Understanding
                |
                v
      Candidate Video Retrieval
                |
                v
       Video-level Ranking
                |
                v
      Coarse Temporal Search
                |
                v
      Fine Temporal Localization
                |
                v
   Semantic Keyframe Alignment
                |
          +-----+------+
          |            |
         KIS          Q&A / TRAKE
          |            |
     video+frame   frame(s)+answer/alignment
          \            /
           \          /
            v        v
          Final Candidate Ranking
                |
                v
             Top-100
                |
                v
         Official Submission
```

The internal representation must retain ranked alternatives through the pipeline because the evaluation uses R@1/R@5/R@20/R@50/R@100.

## 3. Offline data pipeline

Video is the **official competition source of truth**. BTC Keyframes, Objects, CLIP features and Metadata are supporting data.

```text
Official Videos ------------------------------+
                                               |
BTC Keyframes --> frame mapping ---------------+
BTC CLIP features ------------------------------+--> Data validation
BTC Objects -----------------------------------+
BTC Metadata ----------------------------------+
                                               |
                                               v
                                      Unified Data Layer
                                      SQLite + FAISS
                                      + raw feature files
```

The offline layer must preserve exact `video_id`, source `frame_id`, timestamp/frame correspondence and BTC-provided mappings. Never silently renumber source frames.

Current Batch 1 audit known to the team: 873 videos, 177,321 keyframes, 177,321 object records, 873 CLIP feature files, 873 mapping CSVs and 873 metadata/media-info records. OCR and ASR are currently absent from the checked SQLite database and are optional enrichment layers, not baseline prerequisites.

## 4. Online retrieval stages

### Stage A — Query understanding

Normalize the natural-language query and identify task semantics. Decompose multi-event queries when required. Query expansion/translation/LLM assistance is optional and must be benchmarked.

### Stage B — Candidate generation

Primary baseline: use the **BTC-provided CLIP ViT-B/32 frame features** through FAISS to retrieve visual candidates. The current validated index is 512-D with 177,321 vectors and a one-to-one mapping of 177,321 entries.

The retrieval layer may later add sparse metadata/OCR/ASR or another embedding model, but these are additional signals, not replacements for the verified BTC CLIP baseline without evidence.

### Stage C — Video aggregation/ranking

Aggregate frame evidence into video hypotheses. Preserve multiple candidate videos. Do not let one auxiliary object/metadata match override strong visual retrieval evidence without benchmarking.

### Stage D — Temporal localization

Use coarse-to-fine localization on the **original video** for final temporal evidence:

```text
candidate video
  -> coarse temporal window
  -> fine frame/window search
  -> event score
  -> exact source frame_id
```

This stage is critical because TRAKE event intervals can be narrower than 10 frames. Retrieval from sparse BTC keyframes is evidence for candidate generation, not a substitute for precise temporal localization when the target frame is between/around sampled keyframes.

### Stage E — Semantic keyframe selection

The selected frame must represent the queried event semantically. Do not assume that the middle frame, nearest keyframe, or maximum CLIP similarity is automatically the semantic keyframe.

For TRAKE, event order must be preserved when the query defines an ordered sequence:

```text
Event 1 -> frame 1
Event 2 -> frame 2
...
Event N -> frame N
```

with temporal consistency where applicable (`t1 < t2 < ... < tN`). Independent per-event top-k hypotheses may be retained internally, but final alignment must be sequence-aware.

### Stage F — Multimodal reranking

Reranking may combine:

- BTC CLIP retrieval score
- temporal/event score
- object evidence
- metadata evidence
- OCR/ASR evidence when available
- VLM evidence when affordable

Score fusion must remain inspectable and tunable. No fixed weight is a BTC requirement. Every new signal must be validated against Recall@K/Final Score and error analysis.

### Stage G — Task solvers

**KIS:** correct video + frame inside the ground-truth event interval.

**Q&A:** correct video + frame inside the ground-truth interval + semantically correct answer. Answer extraction/generation is a distinct module; a placeholder/unavailable extractor is not a finished Q&A system.

**TRAKE:** correct video + one semantic keyframe for each queried event, with sequence-consistent alignment. Wrong video is fatal to the R-Score.

### Stage H — Final ranking/submission

Return at most 100 ranked candidates. Keep submission formatting separate from retrieval/inference logic.

## 5. Role boundaries

### Person 1 — Video Processing / Data Layer

Owns video ingestion, timeline/frame mapping, BTC feature validation/indexing, object/metadata normalization, SQLite, FAISS, exact frame access and optional OCR/ASR preprocessing.

### Person 2 — Query Engine

Owns query understanding, candidate retrieval, temporal localization, reranking, KIS/Q&A/TRAKE solving, candidate generation and ranking.

### Person 3 — UI

Owns Streamlit and operator/debug presentation. UI communicates through FastAPI and does not access SQLite/FAISS/query-engine internals directly.

## 6. Implementation order

```text
0. Dataset integrity
1. Video ingestion + timeline/frame access
2. BTC CLIP retrieval
3. Video-level aggregation/ranking
4. Coarse-to-fine temporal localization
5. Semantic keyframe alignment
6. Object + metadata fusion
7. OCR + ASR enrichment (only if useful/available)
8. VLM / Q&A answer extraction
9. TRAKE sequence reasoning
10. Learning-to-rank / score optimization
11. Official-like evaluation + submission validation
```

At every stage:

```text
Implement -> Benchmark -> Error analysis -> Keep only if useful
```

Primary metrics: R@1, R@5, R@20, R@50, R@100 and Final Score. Do not optimize by visual inspection alone.

## 7. Rules for adding technology

Before adding a model/index/database/agent:

1. Identify which pipeline stage it serves.
2. Identify the evidence it adds.
3. Verify that the input data actually exists.
4. Benchmark against the current stage baseline.
5. Keep it only if it improves the relevant metric or provides a required capability.
6. Record the decision here or in the relevant module documentation.

Do not add SigLIP, BEiT3, Qdrant, Elasticsearch, Whisper, OCR, BLIP-2, VLMs, LLM query decomposition, etc. merely because a reference baseline mentions them.

## 8. Known current gaps

The repository's current direction already contains retrieval, temporal and ranking scaffolding, but the following must be treated as work items rather than assumed-complete capabilities:

- Validate that all runtime retrieval uses the BTC CLIP index rather than a parallel/custom embedding path.
- Complete video-level aggregation and ranking evaluation.
- Strengthen fine temporal localization against sub-10-frame events.
- Make semantic keyframe selection event-aware rather than similarity-only.
- Add true multimodal fusion only after validating object/metadata usefulness.
- OCR/ASR are not present in the current database audit; add them only as optional enrichment.
- Q&A requires a real answer extractor/generator and evaluation; an unavailable extractor is not completion.
- TRAKE needs sequence-consistent global alignment, not only independent event ranking.
- Establish official-query/submission validation only when BTC supplies the official schema.

## 9. Anti-duplication rule

There is one end-to-end pipeline. Individual modules, experimental scripts and model adapters are implementation variants inside a stage.

Do not create a second `pipeline_v2`, `new_pipeline`, `baseline2`, or parallel end-to-end orchestrator. Replace or extend the relevant stage behind the existing contracts.
