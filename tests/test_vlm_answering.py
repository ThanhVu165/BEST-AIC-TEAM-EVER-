from pathlib import Path

from query_engine.answering import AnswerEvidence
from query_engine.vlm_answering import TransformersImageAnswerExtractor


def test_vlm_returns_evidence_unavailable_for_missing_frame(tmp_path: Path) -> None:
    extractor = TransformersImageAnswerExtractor("test/model")
    result = extractor.answer(
        AnswerEvidence(
            video_id="V1",
            frame_id=1,
            frame_path=str(tmp_path / "missing.jpg"),
            question="What is visible?",
        )
    )
    assert result.status == "evidence_unavailable"
    assert result.answer == ""
