from pathlib import Path

from tools.siglip2_smoke_test import _resolve_images, main


def test_siglip2_smoke_module_imports():
    assert callable(main)


def test_resolve_images_samples_evenly_when_limited(tmp_path: Path):
    paths = []
    for index in range(10):
        path = tmp_path / f"frame_{index:02d}.jpg"
        path.touch()
        paths.append(path)

    resolved = _resolve_images([str(tmp_path / "*.jpg")], limit=3)

    assert len(resolved) == 3
    assert resolved == [paths[0].resolve(), paths[4].resolve(), paths[9].resolve()]
