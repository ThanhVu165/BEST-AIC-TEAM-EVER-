from __future__ import annotations

import numpy as np

from query_engine.temporal import FrameEvidence, TemporalCandidate, fine_localize_source_frames


class FakeReader:
    def read_source_frame(self, video_id: str, frame_id: int):
        return np.full((2, 2, 3), frame_id, dtype=np.uint8)


class FakeEncoder:
    def encode(self, text: str):
        return np.array([1.0, 0.0], dtype=np.float32)

    def encode_images(self, images, *, batch_size: int = 16):
        rows = []
        for image in images:
            value = float(image[0, 0, 0])
            rows.append([value, 1.0])
        arr = np.asarray(rows, dtype=np.float32)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-12)
        return arr


def test_fine_localize_accepts_temporal_candidates_without_schema_mismatch() -> None:
    anchor = TemporalCandidate(
        video_id="V1",
        frame_id=10,
        keyframe_n=3,
        timestamp=1.0,
        score=0.8,
        rank=1,
        anchor_frame_id=10,
        anchor_retrieval_score=0.8,
    )
    result = fine_localize_source_frames(
        [anchor],
        query_text="person riding a motorcycle",
        reader=FakeReader(),
        image_encoder=FakeEncoder(),
        radius=1,
        max_candidates=1,
    )

    assert len(result) == 1
    assert result[0].video_id == "V1"
    assert result[0].retrieval_score == 0.8
    assert result[0].anchor_frame_id == 10
    assert result[0].anchor_retrieval_score == 0.8


def test_fine_localize_accepts_frame_evidence() -> None:
    anchor = FrameEvidence(
        video_id="V1",
        frame_id=10,
        keyframe_n=3,
        timestamp=1.0,
        retrieval_score=0.7,
    )
    result = fine_localize_source_frames(
        [anchor],
        query_text="person riding a motorcycle",
        reader=FakeReader(),
        image_encoder=FakeEncoder(),
        radius=0,
        max_candidates=1,
    )

    assert len(result) == 1
    assert result[0].retrieval_score == 0.7
    assert result[0].anchor_frame_id == 10
