"""Preflight: `legit-edge run` must abort with a clear error if datasets are missing.

A fresh clone has gitignored `data/workloads/` — without preflight, `run` (and `demo`,
which calls `run`) crash deep inside `workload._load_jsonl` with FileNotFoundError.
This test renames `data/workloads/` to `data/workloads.bak` so the CLI sees the
"missing" state, invokes the CLI in-process via CliRunner the same way the other
tests in this directory do, and asserts on exit code + a fix-hint pointing at
`legit-edge pin datasets`. The fixture restores the rename on teardown, even on
failure, so the rest of the test suite keeps working.
"""
from __future__ import annotations
import shutil

import pytest
from typer.testing import CliRunner

from legit_edge.cli.app import app
from legit_edge.paths import DATA_DIR


@pytest.fixture
def datasets_temporarily_renamed():
    if not DATA_DIR.exists():
        pytest.skip("no datasets present — run `uv run legit-edge pin datasets` first")
    backup = DATA_DIR.parent / "workloads.bak"
    if backup.exists():
        pytest.fail(f"stale backup at {backup}; remove manually before retrying")
    DATA_DIR.rename(backup)
    try:
        yield
    finally:
        if DATA_DIR.exists():
            shutil.rmtree(DATA_DIR)
        backup.rename(DATA_DIR)


def test_run_mock_aborts_when_datasets_missing(datasets_temporarily_renamed, tmp_path):
    r = CliRunner().invoke(app, [
        "run", "mock", "--mode", "megaquick", "--out", str(tmp_path / "out"),
    ])
    assert r.exit_code != 0, f"expected non-zero; stdout: {r.stdout!r}"
    assert "pin datasets" in r.stdout.lower(), (
        f"expected 'pin datasets' fix hint; stdout: {r.stdout!r}"
    )
