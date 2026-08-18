# KIS stage-by-stage evaluation

`query_engine` is a ranked-candidate system. Before changing retrieval/ranking logic, evaluate where the correct moment is lost.

## Benchmark input

Use JSONL. Each row contains a normal `QueryRequest` payload plus a local `ground_truth` list:

```json
{"query_id":"q001","task":"KIS","description":"...","ground_truth":[{"video_id":"L21_V001","start_frame":1200,"end_frame":1210}]}
```

`start_frame` and `end_frame` are inclusive. Multiple GT intervals are allowed.

This file is an **internal/stress-test format**. It is not the official AIC 2026 query/GT format. The repository context states that BTC has not yet supplied an official preliminary-round query file, so these measurements must not be reported as official competition scores.

## Run

```powershell
python scripts/evaluate_kis.py `
  --queries path\to\kis_eval.jsonl `
  --db database\aic2026.sqlite `
  --index indexes\clip_vit_b32.faiss `
  --mapping indexes\clip_vit_b32.mapping.json `
  --device auto `
  --output reports\kis_eval.json
```

## What it measures

- `retrieval_frame`: frame-level recall from the raw CLIP retrieval candidates.
- `video_candidates`: recall after collapsing retrieval results to unique videos.
- `fine_localization`: recall after source-video frame decoding and local temporal refinement.
- `final_top100`: recall after canonical KIS reranking/diversification.
- `decoder`: number of source-frame reads, partial/failed reads and latency.

Each stage reports `R@1`, `R@5`, `R@20`, `R@50`, `R@100` plus a mean `FinalScore` over those five values.

## Interpretation

- Retrieval low -> improve candidate generation/query representation.
- Video candidates lower than frame retrieval -> video aggregation is losing the GT video.
- Fine localization lower than video/retrieval -> temporal localization or source-frame decoding is the bottleneck.
- Final Top-100 lower than fine localization -> ranking/diversification is the bottleneck.

Do not tune weights until this stage attribution is measured.
