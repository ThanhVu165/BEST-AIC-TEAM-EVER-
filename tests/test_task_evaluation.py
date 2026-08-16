from query_engine.task_evaluation import evaluate_ranked, final_score, kis_r_score, trake_r_score


def test_kis_requires_video_and_frame_interval() -> None:
    gt = {"video_id": "V1", "frame_start": 100, "frame_end": 110}
    assert kis_r_score({"video_id": "V1", "frame_id": 105}, gt) == 1.0
    assert kis_r_score({"video_id": "V1", "frame_id": 111}, gt) == 0.0
    assert kis_r_score({"video_id": "V2", "frame_id": 105}, gt) == 0.0


def test_trake_is_fractional_over_events() -> None:
    gt = {
        "video_id": "V1",
        "events": [
            {"event_id": "e1", "frame_start": 100, "frame_end": 110},
            {"event_id": "e2", "frame_start": 200, "frame_end": 210},
        ],
    }
    candidate = {
        "video_id": "V1",
        "events": [
            {"event_id": "e1", "frame_id": 105},
            {"event_id": "e2", "frame_id": 999},
        ],
    }
    assert trake_r_score(candidate, gt) == 0.5


def test_final_score_uses_max_r_score_within_each_cutoff() -> None:
    report = final_score([0.5, 0.0, 0.8, 0.0, 0.0, 0.6])
    assert report["R@1"] == 0.5
    assert report["R@5"] == 0.8
    assert report["R@20"] == 0.8
    assert report["R@100"] == 0.8
    assert report["FinalScore"] == 0.74


def test_ranked_kis_evaluation() -> None:
    report = evaluate_ranked(
        "KIS",
        [
            {"video_id": "V2", "frame_id": 10},
            {"video_id": "V1", "frame_id": 105},
        ],
        {"video_id": "V1", "frame_start": 100, "frame_end": 110},
    )
    assert report["R@1"] == 0.0
    assert report["R@5"] == 1.0
    assert report["FinalScore"] == 0.8
