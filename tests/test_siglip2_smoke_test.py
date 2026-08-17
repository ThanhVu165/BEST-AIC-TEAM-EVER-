from tools.siglip2_smoke_test import main


def test_siglip2_smoke_module_imports():
    assert callable(main)
