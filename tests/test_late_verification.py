from types import SimpleNamespace

from query_engine.late_verification import LateVerificationConfig, verify_candidate_windows


class FakeVerifier:
    def __init__(self):
        self.calls = []

    def verify(self, video_path, query, *, fps=2.0):
        self.calls.append((video_path, query, fps))
        return 0.8


class FakeStore:
    def __init__(self, video_path):
        self.video_path = video_path

    def get_video(self, video_id):
        return SimpleNamespace(path=self.video_path, fps=25.0)


def test_disabled_late_verification_is_noop(tmp_path):
    verifier = FakeVerifier()
    video_path = tmp_path / "video.mp4"
    video_path.touch()
    scores = verify_candidate_windows(
        [SimpleNamespace(video_id="v1", frame_id=10)],
        datastore=FakeStore(str(video_path)),
        query="person riding motorcycle",
        verifier=verifier,
        config=LateVerificationConfig(enabled=False),
    )
    assert scores == {}
    assert verifier.calls == []


def test_candidate_limit_and_score_are_enforced(monkeypatch, tmp_path):
    verifier = FakeVerifier()
    video_path = tmp_path / "video.mp4"
    video_path.touch()
    monkeypatch.setattr(
        "query_engine.late_verification.materialize_window",
        lambda window: "/tmp/window.mp4",
    )
    monkeypatch.setattr("os.unlink", lambda path: None)
    candidates = [
        SimpleNamespace(video_id="v1", frame_id=10),
        SimpleNamespace(video_id="v1", frame_id=20),
        SimpleNamespace(video_id="v1", frame_id=30),
    ]
    scores = verify_candidate_windows(
        candidates,
        datastore=FakeStore(str(video_path)),
        query="person riding motorcycle",
        verifier=verifier,
        config=LateVerificationConfig(enabled=True, candidate_limit=2),
    )
    assert len(scores) == 2
    assert all(0.0 <= value <= 1.0 for value in scores.values())
    assert len(verifier.calls) == 2
