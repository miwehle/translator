from __future__ import annotations

import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import pytest
from nmt_lab_shared.logging import close_logger

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYTEST_RUNS_DIR = _REPO_ROOT / ".local_tmp" / "pytest-fixture-runs"


def _create_pytest_run_dir() -> Path:
    run_name = f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}-pid{os.getpid()}"
    run_dir = _PYTEST_RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _cleanup_old_pytest_run_dirs(current_run_dir: Path) -> None:
    if not _PYTEST_RUNS_DIR.is_dir():
        return
    for run_dir in _PYTEST_RUNS_DIR.iterdir():
        if run_dir == current_run_dir or not run_dir.is_dir():
            continue
        shutil.rmtree(run_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def _translator_pytest_run_dir() -> Path:
    run_dir = _create_pytest_run_dir()
    _cleanup_old_pytest_run_dirs(run_dir)
    yield run_dir
    shutil.rmtree(run_dir, ignore_errors=True)


@pytest.fixture
def tmp_path(_translator_pytest_run_dir: Path) -> Path:
    path = _translator_pytest_run_dir / f"tmp_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(autouse=True)
def _close_translator_logging_after_test():
    yield
    close_logger("translator")
