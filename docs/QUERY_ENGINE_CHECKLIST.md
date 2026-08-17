# Query Engine completion checklist

## Canonical pipeline status

### Stage A — Query understanding
- [x] Normalize `QueryRequest` into a stable `QuerySpec`
- [x] Explicit KIS / QA / TRAKE task handling
- [x] Preserve ordered TRAKE event descriptions
- [ ] Learned/LLM semantic decomposition into entity/action/relation/attribute slots

### Stage B — Candidate generation
- [x] BTC CLIP ViT-B/32 text encoder adapter
- [x] FAISS frame retrieval contract
- [x] Expanded frame candidate pool (default 5000)
- [x] Preserve frame alternatives before video aggregation
- [x] Collect optional object/metadata/OCR/ASR evidence
- [ ] Independent multimodal retrieval channels with measured ablations

### Stage C — Video-level ranking
- [x] Aggregate frame evidence into video hypotheses
- [x] Preserve multiple candidate videos
- [ ] Calibrated video-level reranker trained/validated on internal annotations

### Stage D — Temporal localization
- [x] Deterministic source-frame temporal candidate handling
- [x] Ordered TRAKE dynamic-programming alignment
- [x] Temporal window grouping utilities
- [x] Original-video local source-frame refinement around sparse anchors
- [x] Fine frame-level CLIP localization proxy on original videos
- [ ] Learned temporal grounder / sub-10-frame benchmark

### Stage E — Semantic keyframe alignment
- [x] Explicit proxy selector
- [x] Source-frame ID preservation
- [x] Original-video frame refinement proxy
- [ ] Event-aware semantic keyframe scoring beyond CLIP similarity

### Stage F — Multimodal reranking
- [x] Central inspectable score fusion
- [x] Object / metadata / OCR / ASR evidence fields
- [x] Deterministic deduplication
- [x] Candidate diversification utility
- [ ] Action-aware semantic verification
- [ ] Relation/compositional verification
- [ ] Learned/cross-modal reranker benchmark

### Stage G — Task solvers
- [x] KIS candidate generation
- [x] KIS source-frame fine temporal proxy
- [x] QA evidence boundary
- [ ] Production QA answer model
- [ ] Answer normalization / semantic equivalence benchmark
- [x] TRAKE common-video candidate generation
- [x] TRAKE ordered alignment
- [x] TRAKE source-frame fine temporal proxy
- [ ] TRAKE sequence-level learned scoring

### Stage H — Final ranking / submission
- [x] Top-100 output boundary
- [x] Competition-style Final Score utility for supplied reference annotations
- [x] Candidate diversification utility
- [ ] Official submission adapter once BTC schema is fixed

## Data/runtime validation

- [x] SQLite contains `videos`, `frames`, `objects`, `metadata`, `ocr`, `asr_segments`
- [x] 873 videos
- [x] 177,321 frames
- [x] 17,732,100 object rows
- [x] FAISS has 177,321 vectors
- [x] FAISS dimension is 512
- [x] FAISS metric is inner product
- [x] first 1000 vector norms are approximately 1
- [x] all 177,321 mapping rows resolve to matching SQLite frame identity

## Evaluation policy

Official BTC evaluation must not be inferred from local `ground_truth.json` unless BTC explicitly authorizes it as evaluation ground truth. Internal manual/reference annotations are engineering-only.

The required competition-oriented metrics remain:

`R@1`, `R@5`, `R@20`, `R@50`, `R@100`, `Final Score`.
