# Video Pipeline — Person 1

Own this module. The goal is to transform official AIC data into a stable local data package consumed through the interfaces in `query_engine/interfaces.py` and the schemas in `schemas/`.

## First milestone

1. Scan the raw video/keyframe/objects/CLIP/metadata directories.
2. Build a manifest for all 873 Batch-1 videos.
3. Validate the 177,321 keyframe records and frame mappings.
4. Validate that every CLIP feature file is aligned with its mapping file.
5. Validate the 177,321 object JSON files.
6. Build SQLite metadata tables.
7. Build the first CLIP FAISS index plus deterministic index→`(video_id, frame_id)` mapping.
8. Add a small mock `DataStore`/real `DataStore` implementation that the Query Engine can consume.

## Important constraints

- Do not alter `frame_id` semantics.
- Do not commit raw data, generated indexes, SQLite databases or model artifacts.
- Missing optional metadata must be represented explicitly; do not invent data.
- OCR/ASR are optional and must not block the baseline.
