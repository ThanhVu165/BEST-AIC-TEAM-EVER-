# API Contract v1

UI talks to the system through FastAPI only. Internal query-engine implementation details are not public API.

Base path:

```text
/api/v1
```

## POST /search

Runs a query and returns ranked candidates.

### KIS request

```json
{
  "query_id": "q001",
  "task": "KIS",
  "description": "Tìm ..."
}
```

### QA request

```json
{
  "query_id": "q002",
  "task": "QA",
  "description": "...",
  "question": "..."
}
```

### TRAKE request

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

### Response envelope

```json
{
  "query_id": "q001",
  "task": "KIS",
  "status": "completed",
  "candidates": [],
  "error": null
}
```

`status` may be `queued`, `running`, `completed`, or `failed` if asynchronous execution is added later.

## GET /result/{query_id}

Returns the most recent stored result for the query.

## GET /video/{video_id}

Returns public metadata required by UI rendering. It must not expose internal storage secrets.

Example:

```json
{
  "video_id": "L01_V001",
  "fps": 25.0,
  "duration": 132.4,
  "width": 1920,
  "height": 1080,
  "total_frames": 3310
}
```

## GET /video/{video_id}/frame/{frame_id}

Returns the exact frame image associated with the canonical `(video_id, frame_id)` pair.

The UI must not need to know the physical keyframe/video path.

## POST /submission

Converts the latest ranked results into the configured submission format.

Request:

```json
{
  "query_ids": ["q001", "q002", "q003"]
}
```

Response:

```json
{
  "status": "completed",
  "file_name": "submission_YYYYMMDD_HHMMSS.csv"
}
```

Official BTC output formatting is isolated inside the submission formatter. Do not put CSV/JSON column assumptions into retrieval code.

## API principles

1. Stable versioned path: `/api/v1`.
2. JSON schemas come from `schemas/`.
3. UI never imports ML implementation modules.
4. API responses contain enough information for UI but no raw internal model objects.
5. Long-running inference may later become asynchronous without changing the logical result schema.
6. Errors must be explicit; never return an empty candidate list as a substitute for an inference failure.
