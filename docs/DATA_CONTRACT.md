# Data Contract v1

This file defines the stable boundary between the Video Processing/Data Layer and the Query Engine.

## 1. Design rules

- `video_id` is the canonical video identifier.
- `frame_id` is the original/source frame identifier used for competition output.
- `timestamp` is derived from source video timing and is not a replacement for `frame_id`.
- Every mapping from an index-internal ID to `(video_id, frame_id)` must be deterministic.
- Large binary arrays remain outside SQLite.
- The contract must not encode a specific embedding/model implementation.

## 2. VideoRecord

```json
{
  "video_id": "L01_V001",
  "path": "data/raw/videos/L01_V001.mp4",
  "fps": 25.0,
  "width": 1920,
  "height": 1080,
  "duration": 132.4,
  "total_frames": 3310,
  "batch_id": "batch1",
  "metadata_available": true
}
```

Required fields:

- `video_id`
- `path`
- `fps`
- `width`
- `height`
- `duration`
- `total_frames`

## 3. FrameRecord

```json
{
  "video_id": "L01_V001",
  "frame_id": 1532,
  "timestamp": 61.28,
  "path": "data/raw/keyframes/L01_V001/....jpg",
  "is_keyframe": true
}
```

Required fields:

- `video_id`
- `frame_id`
- `timestamp`
- `path`
- `is_keyframe`

A `frame_id` used for submission must map back to the source video frame convention validated by the video pipeline.

## 4. ObjectRecord

```json
{
  "video_id": "L01_V001",
  "frame_id": 1532,
  "objects": [
    {
      "label": "person",
      "confidence": 0.94,
      "bbox": [100, 120, 300, 500]
    }
  ]
}
```

`bbox` uses `[x1, y1, x2, y2]` in pixel coordinates unless the source format establishes another convention. The video pipeline must document any coordinate conversion.

## 5. Optional OCRRecord

```json
{
  "video_id": "L01_V001",
  "frame_id": 1532,
  "text": "...",
  "confidence": 0.91
}
```

OCR is optional and must not be required for baseline execution.

## 6. Optional ASRSegment

```json
{
  "video_id": "L01_V001",
  "start_time": 61.0,
  "end_time": 64.3,
  "text": "..."
}
```

ASR is optional and must not be required for baseline execution.

## 7. QueryRequest

The query contract supports both structured and natural-language input.

### KIS

```json
{
  "query_id": "q001",
  "task": "KIS",
  "description": "..."
}
```

### Q&A

```json
{
  "query_id": "q002",
  "task": "QA",
  "description": "...",
  "question": "..."
}
```

### TRAKE

```json
{
  "query_id": "q003",
  "task": "TRAKE",
  "events": [
    {"event_id": "E1", "description": "..."},
    {"event_id": "E2", "description": "..."}
  ]
}
```

The `task` field may be omitted at a future API boundary if natural-language task detection is enabled. Internally, the normalized representation must still identify the task.

## 8. Candidate

```json
{
  "rank": 1,
  "video_id": "L01_V001",
  "frame_id": 1532,
  "score": 0.9231,
  "retrieval_score": 0.91,
  "temporal_score": 0.88,
  "rerank_score": 0.94,
  "evidence": {
    "sources": ["clip", "object"]
  }
}
```

For debugging, component scores may be present. `score` is the final ranking score for the candidate.

## 9. KISResult

```json
{
  "query_id": "q001",
  "task": "KIS",
  "candidates": [
    {
      "rank": 1,
      "video_id": "L01_V001",
      "frame_id": 1532,
      "score": 0.94
    }
  ]
}
```

## 10. QAResult

```json
{
  "query_id": "q002",
  "task": "QA",
  "candidates": [
    {
      "rank": 1,
      "video_id": "L01_V001",
      "frame_id": 1532,
      "answer": "màu đỏ",
      "score": 0.91
    }
  ]
}
```

## 11. TRAKEEvent

```json
{
  "event_id": "E1",
  "frame_id": 103,
  "score": 0.91
}
```

## 12. TRAKECandidate

```json
{
  "rank": 1,
  "video_id": "L01_V001",
  "events": [
    {"event_id": "E1", "frame_id": 103, "score": 0.91},
    {"event_id": "E2", "frame_id": 151, "score": 0.88}
  ],
  "score": 0.90
}
```

## 13. Data-store operations

The Query Engine should depend on an abstract data access layer with operations conceptually equivalent to:

```text
get_video(video_id)
get_frame(video_id, frame_id)
get_frames(video_id)
get_frames_in_range(video_id, start_frame, end_frame)
get_objects(video_id, frame_id)
get_metadata(video_id)
get_ocr(video_id, frame_id)          # optional
get_asr(video_id, start_time, end_time)  # optional
search_vector(index_name, vector, top_k)
get_embedding(index_name, video_id, frame_id)
```

The physical implementation may use SQLite, FAISS, NumPy files, or future storage systems. Query Engine code must not require those concrete implementations.

## 14. Vector index mapping

Every FAISS/internal vector identifier must map to a stable record:

```text
internal_index_id -> video_id + frame_id
```

A mapping file must be versioned together with the index.

## 15. Dataset versioning

Each generated data package/index set must record:

- `contract_version`
- `dataset_version`
- `batch_id`
- source paths/version information where applicable
- generation timestamp

## 16. Invalid-data policy

If an auxiliary artifact is missing for a video/frame, do not silently invent it. Return a missing/nullable result and allow Query Engine to continue using other modalities.
