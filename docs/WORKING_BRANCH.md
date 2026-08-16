# Working Branch Policy

For the AIC 2026 Query/Video Retrieval implementation, the active development branch is `feature/query-engine`.

All query-engine work should be committed to this branch until the module is ready to merge into `main`.

The query engine must depend on shared contracts (`schemas/`, `data_layer/`) rather than on a concrete video-pipeline implementation. Large datasets, model weights, SQLite databases, FAISS indexes, and generated artifacts stay outside Git.
