from __future__ import annotations

import numpy as np

from query_engine.temporal import FrameEvidence, fine_localize_source_frames


class FakeReader:
    def read_source_frame(self, video_id: str, frame_id: int):
        # The frame value itself is enough for the fake image encoder.
        return np.array([[frame_id]], dtype=np.float32)

    def read_source_frames(self, video_id: str, frame_ids: list[int]):
        return {frame_id: self.read_source_frame(video_id, frame_id) for frame_id in frame_ids}


class FakeImageEncoder:
    def encode(self, text: str):
        assert text == "person riding bicycle"
        return np.array([1.0, 0.0], dtype=np.float32)

    def encode_images(self, images, *, batch_size: int = 16):
        # Frame 12 is the only exact semantic match in the local source window.
        vectors = []
        for image in images:
            frame_id = int(image[0, 0])
            vectors.append([1.0, 0.0] if frame_id == 12 else [0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


def test_fine_localization_can_return_non_keyframe_source_frame() -> None:
    anchors = [
        FrameEvidence(
            video_id="V1",
            frame_id=10,
            keyframe_n=1,
            timestamp=1.0,
            retrieval_score=0.8,
        )
    ]

    result = fine_localize_source_frames(
        anchors,
        query_text="person riding bicycle",
        reader=FakeReader(),
        image_encoder=FakeImageEncoder(),
        radius=3,
        max_candidates=5,
    )

    assert len(result) == 1
    assert result[0].video_id == "V1"
    assert result[0].frame_id == 12
    assert result[0].keyframe_n is None
    assert result[0].score == 1.0
