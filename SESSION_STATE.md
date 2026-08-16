# AIC2026 Video Retrieval — Session Checkpoint

Last updated: 2026-08-17

## 1. Current objective

Build a complete AIC2026 Video Retrieval system covering:

- KIS: video + semantic keyframe retrieval
- QA: video + relevant frame + answer
- TRAKE: video + ordered semantic keyframes for events

Current priority is validating and improving the retrieval/ranking pipeline before larger architectural changes.

## 2. Repository / local state

Repository: `ThanhVu165/BEST-AIC-TEAM-EVER-`
Default branch: `main`

Important local artifacts confirmed by the user:

- `database/aic2026.sqlite` — ~2.5 GB
- `indexes/clip_vit_b32.faiss` — 177,321 vectors, 512 dimensions
- `indexes/clip_vit_b32.mapping.json` — 177,321 mappings
- `scripts/setup_local_data.ps1`
- `scripts/manual_review.py`
- `scripts/validate_batch1.py`
- `scripts/smoke_query.py`

Local database counts:

- videos: 873
- frames: 177,321
- objects: 17,732,100

SQLite tables confirmed:

- `videos`
- `frames`
- `objects`
- `metadata`
- `ocr`
- `asr_segments`

FAISS checks confirmed:

- `ntotal = 177321`
- `dimension = 512`
- metric type = 0 (inner product)
- first 1000 reconstructed vectors have norm ~= 1.0
- all 177,321 mapping entries match the corresponding SQLite `frame_id` (full mapping check returned `bad=0`)

Therefore the current CLIP index/mapping/database package is internally consistent based on the checks performed.

## 3. Current retrieval implementation

Main files inspected:

- `data_layer/faiss_store.py`
- `data_layer/datastore.py`
- `query_engine/retrieval.py`
- `query_engine/temporal.py`
- `query_engine/ranking.py`
- `query_engine/engine.py`

Current architecture is approximately:

`text query -> CLIP embedding -> FAISS frame retrieval -> object evidence fusion -> temporal/keyframe selection -> task-specific output`

Current retrieval defaults:

- `frame_top_k = 1000`
- `video_top_k = 100`
- `object_weight = 0.10`

Current object reranking is lexical/token based. It detects whether an object label exactly covers query tokens and uses detection confidence as auxiliary evidence.

Current temporal code mostly selects/ranks already retrieved frames. TRAKE has a dynamic-programming ordered-frame selector, but it does not yet perform fine-grained temporal grounding from raw video.

Current `ranking.py` defines a separate fusion formula (`0.92 retrieval + 0.06 object + 0.02 metadata`), but the main `BaselineQueryEngine` path currently uses the retriever's fused score rather than a full metadata-aware reranking stage.

Current QA answer extraction defaults to `UnavailableAnswerExtractor` unless another extractor is supplied.

## 4. Important architectural findings

The current system is a valid baseline/data-access foundation, but it is NOT yet a competition-grade multimodal retrieval system.

Main gaps identified:

1. **Action semantics** are weak.
   - CLIP + object evidence can confuse `riding a bicycle` with `repairing a bicycle`.

2. **Compositional relations** are weak.
   - `person sitting at a table` can retrieve a frame containing only a table or only a sitting person.
   - Current object scoring checks object existence but not relations/actions between entities.

3. **OCR and ASR are stored but not yet used by the main retrieval path.**

4. **Metadata is stored but not yet meaningfully integrated into ranking.**

5. **Temporal localization is still shallow.**
   - Current temporal functions select from retrieved keyframes rather than searching a fine-grained frame window in the original video.
   - This is especially important for TRAKE where the correct semantic window may be <10 frames.

6. **No ground-truth labels are available from BTC.**
   - Do not assume a ground-truth file exists.
   - Evaluation must initially rely on manual review and later on any official evaluation mechanism/data that becomes available.

7. **Do not blindly increase object weight.**
   - Manual tests show that high-confidence object presence can push semantically wrong action frames upward.

## 5. Manual evaluation already completed

### Query 1
`a man riding a bicycle`

Top-20 manual labels:

`1A 2A 3B 4A 5A 6A 7A 8A 9A 10A 11A 12A 13A 14B 15A 16A 17A 18A 19A 20A`

Result: 18/20 relevant, P@20 = 0.90.

False positives:

- rank 3: bicycle + man, but man is repairing the bicycle rather than riding it
- rank 14: man is still repairing the bicycle

Failure type: **action semantic confusion**.

### Query 2
`a person sitting at a table`

Manual labels:

- rank 3 = B: table visible, but no person sitting
- rank 7 = B: person sitting, but no table
- all other ranks 1–20 = A

Result: 18/20 relevant, P@20 = 0.90.

Failure type: **compositional retrieval / missing relation constraint**.

## 6. Current interpretation

The baseline has reasonable coarse semantic retrieval, but the errors are not primarily database/index integrity problems. The next bottleneck is **semantic reranking**:

- query decomposition into entities / attributes / actions / relations
- multimodal evidence fusion
- action-aware verification
- compositional verification
- OCR/ASR/metadata integration
- fine temporal refinement around candidate frames

The target architecture should evolve toward:

`Natural Language Query`
`-> Query Understanding`
`-> coarse multimodal retrieval`
`-> candidate video/frame generation`
`-> compositional/action-aware reranking`
`-> temporal refinement`
`-> semantic keyframe selection`
`-> task-specific output`

## 7. Manual evaluation protocol for next session

Continue with representative queries and label only Top-20 as A/B, including a short reason for every B.

Recommended next query:

`a person holding a cup`

Command:

```powershell
python scripts/manual_review.py `
  --query "a person holding a cup" `
  --task KIS `
  --top-k 20
```

Then test categories:

- object: `a red car`, `a dog`, `a bicycle`
- action: `a person running`, `a person cooking`, `a person drinking`
- composition: `a person holding a cup`, `a man standing next to a car`
- scene: `people sitting in a restaurant`, `a person walking on a street`
- temporal/action transition: `a person opening a door`, `a person picking up a cup`

After roughly 10–15 queries, summarize P@1/P@5/P@20 and false-positive types before changing ranking weights.

## 8. Next engineering priorities

Priority order for the next session:

1. Finish a small manual benchmark (10–15 queries).
2. Inspect/query the actual OCR, ASR and metadata coverage.
3. Inspect the query schemas and official task/output contracts.
4. Design query decomposition for entities/actions/relations.
5. Implement a clean reranking interface so multiple evidence sources can be ablated independently.
6. Add OCR/ASR/metadata evidence where appropriate.
7. Add a fine-grained temporal refinement stage using the original videos.
8. Add VLM/cross-encoder verification only after candidate generation is stable.
9. Evaluate ranking at Top-1/5/20/50/100, not only Top-1.

## 9. Important caution

Do NOT treat the current 90% P@20 manual result as official performance. It is only a small human-reviewed sample without ground truth.

Do NOT modify the core retrieval weights solely from these two queries. Collect more evidence first.

## 10. Session restart instruction

At the beginning of the next session, read this file first. Then inspect the current repository state and continue from Section 7/8. The immediate task is to continue manual evaluation and then improve the retrieval architecture based on observed failure modes, while preserving the validated SQLite/FAISS data package.
