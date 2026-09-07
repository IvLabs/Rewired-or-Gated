import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (require real model download/inference, deselect with -m 'not slow')",
    )
