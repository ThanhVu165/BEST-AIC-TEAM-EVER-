# Query Engine completion checklist

## Done on `feature/query-engine`

- [x] CLIP text embedding adapter
- [x] FAISS frame retrieval contract
- [x] deterministic frame/video candidate ranking
- [x] optional object evidence fusion
- [x] KIS candidate generation
- [x] TRAKE common-video candidate generation
- [x] deterministic semantic-keyframe proxy
- [x] QA evidence boundary and no-fabrication fallback
- [x] Recall@1/5/20/50/100 evaluator
- [x] CI for main-targeted PRs

## Requires Batch 1 local validation

- [ ] verify actual SQLite schema and paths
- [ ] verify FAISS dimension and mapping against all Batch 1 rows
- [ ] measure Recall@K on the real query set
- [ ] inspect top-100 misses and duplicate-video concentration
- [ ] tune object-fusion weight from measured validation data

## Research/model work still required

- [ ] learned temporal grounding for sub-10-frame events
- [ ] semantic keyframe model beyond CLIP-score proxy
- [ ] production VLM answer extractor
- [ ] stronger multimodal reranker using all available auxiliary evidence
- [ ] official-format submission/evaluation adapter once the exact ground-truth schema is available
