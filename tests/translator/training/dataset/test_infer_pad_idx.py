from __future__ import annotations

import os
from pathlib import Path

import pytest

from translator.training.dataset.arrow_dataset import infer_pad_idx


def _find_preprocessed_dataset() -> Path:
    configured = os.getenv("TRANSLATOR2_TESTDATA_PREPROCESSED")
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if candidate.is_dir():
            return candidate
        pytest.skip(
            "TRANSLATOR2_TESTDATA_PREPROCESSED is set but does not point to a directory: "
            f"{candidate}"
        )

    testdata_dir = Path(__file__).resolve().parents[3] / "testdata"
    if not testdata_dir.exists():
        pytest.skip(f"Missing test data directory: {testdata_dir}")

    candidates = sorted(
        p for p in testdata_dir.rglob("europarl_de-en_train_10000") if p.is_dir()
    )
    if not candidates:
        pytest.skip(
            "No preprocessed Arrow dataset found under tests/testdata. "
            "Expected a directory like '.../europarl_de-en_train_10000'."
        )
    return candidates[0]


def test_infer_pad_idx_returns_next_token_id() -> None:
    dataset_path = _find_preprocessed_dataset()

    src_pad_idx = infer_pad_idx(dataset_path, "src_ids")
    tgt_pad_idx = infer_pad_idx(dataset_path, "tgt_ids")

    assert src_pad_idx > 0
    assert tgt_pad_idx > 0
