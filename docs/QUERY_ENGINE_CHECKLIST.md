# Query Engine completion checklist

## Implemented on `feature/query-engine`

- [x] CLIP text embedding adapter
- [x] FAISS frame retrieval contract
- [x] expanded frame candidate pool (default 5000)
- [x] deterministic frame/video candidate ranking
- [x] optional object evidence fusion
- [x] optional OCR/ASR/metadata data-access adapters
- [x] KIS candidate generation
- [x] TRAKE common-video candidate generation
- [x] deterministic semantic-keyframe proxy
- [x] TRAKE score reflects aligned event predictions
- [x] QA evidence boundary and no-fabrication fallback
- [x] Recall@1/5/20/50/100 evaluator for supplied reference ground truth
- [x] manual retrieval review report
- [x] API source-frame lookup uses original `frame_id`
- [x] CI for main-targeted PRs

## Verified locally by the team

- [x] SQLite contains `videos`, `frames`, `objects`, `metadata`, `ocr`, `asr_segments`
- [x] 873 videos
- [x] 177,321 frames
- [x] 17,732,100 object rows
- [x] FAISS has 177,321 vectors
- [x] FAISS dimension is 512
- [x] FAISS metric is inner product
- [x] first 1000 vector norms are approximately 1
- [x] all 177,321 mapping rows resolve to the matching SQLite frame identity

## Cannot be measured officially yet

- [ ] official preliminary-round query file (not supplied by BTC at the time of development)
- [ ] official ground-truth query annotations (not supplied by BTC at the time of development)
- [ ] official Final Score before submission to BTC

## Manual validation workflow

1. Build/validate the local data package.
2. Run `scripts/manual_review.py` on a query.
3. Inspect the generated HTML report and verify video/frame relevance manually.
4. Record a small internal annotation set for representative KIS, QA and TRAKE queries.
5. Compare retrieval changes on the same internal set before accepting a ranking change.

The manual set is an internal engineering benchmark. It must never be represented as official AIC 2026 ground truth.

## Research/model work still required

- [ ] coarse-to-fine temporal localization on original video frames
- [ ] semantic keyframe selection beyond the CLIP-score proxy
- [ ] production VLM answer extractor and answer normalization
- [ ] stronger multimodal reranker using auxiliary evidence
- [ ] TRAKE sequence optimization with fine-grained event localization
- [ ] official submission-format adapter once BTC's exact submission interface is fixed
