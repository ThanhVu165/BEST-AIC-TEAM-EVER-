from pathlib import Path

from data_layer.datastore import LocalDataStore


def test_local_datastore_resolves_repository_relative_video_path(tmp_path: Path):
    project_root = tmp_path / "repo"
    video = project_root / "data" / "raw" / "videos" / "L01_V001.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"placeholder")

    db = project_root / "database" / "aic2026.sqlite"
    db.parent.mkdir(parents=True)
    store = LocalDataStore(db, project_root=project_root)

    assert store._resolve_repo_path("data/raw/videos/L01_V001.mp4") == video.resolve()


def test_local_datastore_default_root_is_repository_for_database_artifact(tmp_path: Path):
    project_root = tmp_path / "repo"
    db = project_root / "database" / "aic2026.sqlite"
    db.parent.mkdir(parents=True)

    store = LocalDataStore(db)

    assert store.project_root == project_root.resolve()


def test_local_datastore_reads_compact_windows_sequentially(monkeypatch, tmp_path: Path):
    class FakeCapture:
        def __init__(self):
            self.current = 0
            self.seek_calls = []

        def isOpened(self):
            return True

        def set(self, prop, value):
            self.seek_calls.append((prop, value))
            self.current = int(value)
            return True

        def read(self):
            frame = self.current
            self.current += 1
            return True, frame

        def release(self):
            pass

    fake_capture = FakeCapture()

    class FakeCV2:
        CAP_PROP_POS_FRAMES = 1

        @staticmethod
        def VideoCapture(_path):
            return fake_capture

        @staticmethod
        def cvtColor(frame, _code):
            return frame

        COLOR_BGR2RGB = 2

    monkeypatch.setitem(__import__("sys").modules, "cv2", FakeCV2)

    project_root = tmp_path / "repo"
    db = project_root / "database" / "aic2026.sqlite"
    db.parent.mkdir(parents=True)
    video = project_root / "data" / "raw" / "videos" / "L01_V001.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"placeholder")

    store = LocalDataStore(db)
    monkeypatch.setattr(store, "_video_path", lambda _video_id: video)

    result = store.read_source_frames("L01_V001", [101, 102, 103])

    assert result == {101: 101, 102: 102, 103: 103}
    assert fake_capture.seek_calls == [(FakeCV2.CAP_PROP_POS_FRAMES, 101)]
