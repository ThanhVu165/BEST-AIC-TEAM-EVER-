from query_engine.answering import AnswerEvidence, UnavailableAnswerExtractor


def test_unavailable_answer_extractor_is_explicit() -> None:
    result = UnavailableAnswerExtractor().answer(
        AnswerEvidence(
            video_id="V1",
            frame_id=12,
            frame_path="frame.jpg",
            question="What is shown?",
        )
    )
    assert result.answer == ""
    assert result.confidence is None
    assert result.status == "model_unavailable"
