from collections.abc import Iterable
from pathlib import Path
from typing import cast

from datasets import Dataset

from translator.training import Example, check_dataset
from translator.training.dataset import load_arrow_records


def _write_dataset(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    dataset_path = tmp_path / "dataset.mapped"
    Dataset.from_list(rows).save_to_disk(str(dataset_path))
    return dataset_path


def test_check_dataset_collects_basic_stats(tmp_path: Path):
    dataset_path = _write_dataset(tmp_path, [
        {"id": 1, "src_ids": [10, 11], "tgt_ids": [2, 20, 3]},
        {"id": 2, "src_ids": [12, 13, 14], "tgt_ids": [2, 21, 22, 3]},
    ])
    ds = cast(Iterable[Example], load_arrow_records(dataset_path))

    out = check_dataset(ds, src_pad_idx=98, tgt_pad_idx=99)

    assert out["num_examples"] == 2
    assert out["min_src_len"] == 2
    assert out["max_src_len"] == 3
    assert out["min_tgt_len"] == 3
    assert out["max_tgt_len"] == 4
    assert out["max_src_token_id"] == 14
    assert out["max_tgt_token_id"] == 22
    assert out["inferred_tgt_bos_id"] == 2
    assert out["inferred_tgt_eos_id"] == 3
    assert out["bos_consistency"] == 1.0
    assert out["eos_consistency"] == 1.0
    assert out["src_pad_idx"] == 98
    assert out["tgt_pad_idx"] == 99


def test_check_dataset_rejects_duplicate_ids(tmp_path: Path):
    dataset_path = _write_dataset(tmp_path, [
        {"id": 1, "src_ids": [1, 2], "tgt_ids": [2, 3]},
        {"id": 1, "src_ids": [4, 5], "tgt_ids": [2, 3]},
    ])
    ds = cast(Iterable[Example], load_arrow_records(dataset_path))

    try:
        check_dataset(ds)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "Duplicate id" in str(exc)


def test_check_dataset_rejects_inconsistent_tgt_bos(tmp_path: Path):
    dataset_path = _write_dataset(tmp_path, [
        {"id": 1, "src_ids": [10, 11], "tgt_ids": [2, 20, 3]},
        {"id": 2, "src_ids": [12, 13], "tgt_ids": [7, 21, 3]},
    ])
    ds = cast(Iterable[Example], load_arrow_records(dataset_path))

    try:
        check_dataset(ds, src_pad_idx=98, tgt_pad_idx=99)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "BOS token is not consistent" in str(exc)
        assert "id" in str(exc)


def test_check_dataset_rejects_inconsistent_tgt_eos(tmp_path: Path):
    dataset_path = _write_dataset(tmp_path, [
        {"id": 1, "src_ids": [10, 11], "tgt_ids": [2, 20, 3]},
        {"id": 2, "src_ids": [12, 13], "tgt_ids": [2, 21, 8]},
    ])
    ds = cast(Iterable[Example], load_arrow_records(dataset_path))

    try:
        check_dataset(ds, src_pad_idx=98, tgt_pad_idx=99)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "EOS token is not consistent" in str(exc)
        assert "id" in str(exc)


def test_check_dataset_rejects_equal_tgt_bos_and_eos(tmp_path: Path):
    dataset_path = _write_dataset(tmp_path, [
        {"id": 1, "src_ids": [10, 11], "tgt_ids": [2, 20, 2]},
        {"id": 2, "src_ids": [12, 13], "tgt_ids": [2, 21, 2]},
    ])
    ds = cast(Iterable[Example], load_arrow_records(dataset_path))

    try:
        check_dataset(ds, src_pad_idx=98, tgt_pad_idx=99)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "BOS and EOS token must differ" in str(exc)


def test_check_dataset_rejects_pad_token_in_raw_tgt_sequences(tmp_path: Path):
    dataset_path = _write_dataset(tmp_path, [
        {"id": 1, "src_ids": [10, 11], "tgt_ids": [2, 99, 3]},
        {"id": 2, "src_ids": [12, 13], "tgt_ids": [2, 21, 3]},
    ])
    ds = cast(Iterable[Example], load_arrow_records(dataset_path))

    try:
        check_dataset(ds, src_pad_idx=98, tgt_pad_idx=99)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "pad token must not appear" in str(exc)
        assert "id" in str(exc)


def test_check_dataset_rejects_pad_token_in_raw_src_sequences(tmp_path: Path):
    dataset_path = _write_dataset(tmp_path, [
        {"id": 1, "src_ids": [10, 98], "tgt_ids": [2, 20, 3]},
        {"id": 2, "src_ids": [12, 13], "tgt_ids": [2, 21, 3]},
    ])
    ds = cast(Iterable[Example], load_arrow_records(dataset_path))

    try:
        check_dataset(ds, src_pad_idx=98, tgt_pad_idx=99)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "Source pad token must not appear" in str(exc)
        assert "id" in str(exc)
