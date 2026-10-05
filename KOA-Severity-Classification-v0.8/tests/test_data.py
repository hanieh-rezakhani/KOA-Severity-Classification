from pathlib import Path

import pytest

from koa_severity.data import get_dataset_path


def test_dataset_path_requires_configuration(monkeypatch):
    monkeypatch.delenv("KOA_DATASET_PATH", raising=False)
    with pytest.raises(ValueError):
        get_dataset_path()


def test_dataset_path_resolves_existing_directory(tmp_path):
    assert get_dataset_path(tmp_path) == Path(tmp_path).resolve()
