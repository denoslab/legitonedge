import os
from pathlib import Path
import pytest

from legit_edge.paths import find_project_root, PROJECT_ROOT, DATA_DIR, RESULTS_DIR, CONFIGS_DIR, LOGS_DIR


def test_project_root_is_a_path_with_pyproject_and_src():
    # Real repo: PROJECT_ROOT must contain both
    assert (PROJECT_ROOT / "pyproject.toml").exists()
    assert (PROJECT_ROOT / "src" / "legit_edge").exists()


def test_dir_constants_compose_off_root():
    assert DATA_DIR == PROJECT_ROOT / "data" / "workloads"
    assert RESULTS_DIR == PROJECT_ROOT / "results"
    assert CONFIGS_DIR == PROJECT_ROOT / "configs"
    assert LOGS_DIR == PROJECT_ROOT / "logs"


def test_env_var_override(tmp_path, monkeypatch):
    fake = tmp_path / "fake_project"
    (fake / "src" / "legit_edge").mkdir(parents=True)
    (fake / "pyproject.toml").write_text("[project]\nname='x'\n")
    monkeypatch.setenv("LEGIT_EDGE_ROOT", str(fake))
    assert find_project_root() == fake.resolve()


def test_env_var_pointing_to_non_project_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("LEGIT_EDGE_ROOT", str(tmp_path))
    with pytest.raises(RuntimeError, match="does not point to a legit-edge project"):
        find_project_root()
