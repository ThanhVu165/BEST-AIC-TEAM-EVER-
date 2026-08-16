PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS videos (
    video_id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    fps REAL NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    duration REAL NOT NULL,
    total_frames INTEGER NOT NULL,
    batch_id TEXT,
    metadata_available INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS frames (
    video_id TEXT NOT NULL,
    frame_id INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    path TEXT NOT NULL,
    is_keyframe INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (video_id, frame_id),
    FOREIGN KEY (video_id) REFERENCES videos(video_id)
);

CREATE INDEX IF NOT EXISTS idx_frames_video_timestamp
    ON frames(video_id, timestamp);

CREATE TABLE IF NOT EXISTS objects (
    video_id TEXT NOT NULL,
    frame_id INTEGER NOT NULL,
    label TEXT NOT NULL,
    confidence REAL NOT NULL,
    x1 REAL NOT NULL,
    y1 REAL NOT NULL,
    x2 REAL NOT NULL,
    y2 REAL NOT NULL,
    FOREIGN KEY (video_id, frame_id) REFERENCES frames(video_id, frame_id)
);

CREATE INDEX IF NOT EXISTS idx_objects_video_frame
    ON objects(video_id, frame_id);

CREATE INDEX IF NOT EXISTS idx_objects_label
    ON objects(label);

CREATE TABLE IF NOT EXISTS metadata (
    video_id TEXT PRIMARY KEY,
    raw_json TEXT NOT NULL,
    FOREIGN KEY (video_id) REFERENCES videos(video_id)
);

CREATE TABLE IF NOT EXISTS ocr (
    video_id TEXT NOT NULL,
    frame_id INTEGER NOT NULL,
    text TEXT NOT NULL,
    confidence REAL,
    FOREIGN KEY (video_id, frame_id) REFERENCES frames(video_id, frame_id)
);

CREATE TABLE IF NOT EXISTS asr_segments (
    video_id TEXT NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    text TEXT NOT NULL,
    FOREIGN KEY (video_id) REFERENCES videos(video_id)
);
