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
