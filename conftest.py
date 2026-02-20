"""Root conftest: path setup and --runslow marker."""

import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="Run slow tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with 'pytest -m \"not slow\"')")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="Need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
