# Query Engine — Person 2

Own this module. This is the AI/retrieval core.

## Target pipeline

```text
Natural Language Query
    -> Query normalization
    -> Task detection / structured task input
    -> Query understanding
    -> Candidate retrieval
    -> Temporal localization
    -> Fine-grained keyframe selection
    -> Multimodal reranking
    -> KIS / Q&A / TRAKE solver
    -> Candidate generation + ranking (<=100)
```

## First milestone

Build a baseline that can consume the shared `DataStore` interface and return a valid ranked result for every task without OCR/ASR/VLM dependencies.

## Constraints

- Do not read SQLite/FAISS files directly inside retrieval modules.
- Do not change shared schemas without team agreement.
- Keep model implementations behind adapters so they can be replaced.
- Preserve multiple candidates because BTC scores R@1/R@5/R@20/R@50/R@100.
- Keep submission formatting separate from retrieval logic.
- Treat external/reference query datasets as non-official unless BTC later supplies official queries.
