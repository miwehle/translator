from __future__ import annotations

from pathlib import Path

from translator.data_prod import infer_pad_idx

from tests.translator.train_prod.test_train_prod_loss_progress import (
    _find_mapped_dataset,
)


def test_infer_pad_idx_returns_next_token_id() -> None:
    dataset_path = _find_mapped_dataset()

    src_pad_idx = infer_pad_idx(dataset_path, "src_ids")
    tgt_pad_idx = infer_pad_idx(dataset_path, "tgt_ids")

    assert src_pad_idx > 0
    assert tgt_pad_idx > 0
