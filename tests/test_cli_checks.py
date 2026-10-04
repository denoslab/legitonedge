from legit_edge.cli.checks import (
    Check,
    check_python,
    check_uv,
    check_package,
    check_configs,
    check_datasets,
    check_ollama,
    check_ina260,
)


def test_check_python_ok():
    c = check_python()
    assert c.status == "ok"
    assert "3.1" in c.detail


def test_check_uv_ok():
    c = check_uv()
    assert c.status in ("ok", "fail")  # depends on PATH, but never crashes


def test_check_package_ok():
    c = check_package()
    assert c.status == "ok"
    assert "legit_edge" in c.detail


def test_check_configs_ok():
    c = check_configs()
    assert c.status == "ok"


def test_check_datasets_ok():
    c = check_datasets()
    # may be warn if pinning hasn't run yet on this machine; just assert it doesn't crash
    assert c.status in ("ok", "warn", "fail")


def test_check_ollama_warn_when_unreachable():
    c = check_ollama("http://localhost:1")   # unused port
    assert c.status == "warn"
    assert "unreachable" in c.detail.lower() or "ollama" in c.detail.lower()


def test_check_ina260_warn_when_not_installed():
    c = check_ina260()
    assert c.status in ("ok", "warn")
