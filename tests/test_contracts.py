from schemas import (
    KISResult,
    QACandidate,
    QAResult,
    QueryRequest,
    TRAKECandidate,
    TRAKEEventPrediction,
    TRAKEResult,
)


def test_kis_request_and_result_limit():
    request = QueryRequest(
        query_id="q1",
        task="KIS",
        description="test",
    )
    assert request.task == "KIS"

    result = KISResult(
        query_id="q1",
        task="KIS",
        candidates=[
            {"rank": 1, "video_id": "V1", "frame_id": 10, "score": 0.8}
        ],
    )
    assert len(result.candidates) == 1


def test_qa_candidate():
    result = QAResult(
        query_id="q2",
        task="QA",
        candidates=[
            QACandidate(
                rank=1,
                video_id="V1",
                frame_id=20,
                score=0.9,
                answer="five",
            )
        ],
    )
    assert result.candidates[0].answer == "five"


def test_trake_candidate():
    result = TRAKEResult(
        query_id="q3",
        task="TRAKE",
        candidates=[
            TRAKECandidate(
                rank=1,
                video_id="V1",
                score=0.75,
                events=[
                    TRAKEEventPrediction(event_id="E1", frame_id=100, score=0.8),
                    TRAKEEventPrediction(event_id="E2", frame_id=200, score=0.7),
                ],
            )
        ],
    )
    assert len(result.candidates[0].events) == 2
