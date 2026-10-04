"""Shared pytest config. Tests marked ``requires_datasets`` exercise the end-to-end
mock pipeline and need the pinned workload JSONLs; they are skipped on a fresh clone
until ``legit-edge pin datasets`` has been run."""
import pytest


def pytest_collection_modifyitems(config, items):
    from legit_edge.cli.checks import check_datasets

    if check_datasets().status == "ok":
        return
    skip = pytest.mark.skip(reason="workload datasets not pinned; run `legit-edge pin datasets`")
    for item in items:
        if "requires_datasets" in item.keywords:
            item.add_marker(skip)
