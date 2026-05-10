from __future__ import annotations

import csv
from pathlib import Path

from translator.registers import append_checkpoint_register


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def test_append_checkpoint_register_writes_checkpoint_parent_columns(tmp_path: Path) -> None:
    append_checkpoint_register(
        tmp_path,
        checkpoint="de-en-translator/r2",
        validation_loss=1.25,
        dataset="europarl/train",
        parent="de-en-translator/r1",
        git_commit="test-commit-abcdef",
    )

    rows = _read_rows(tmp_path / "checkpoint_register.csv")

    assert list(rows[0]) == ["timestamp", "checkpoint", "validation_loss", "dataset", "parent", "git_commit"]
    assert rows == [
        {
            "timestamp": rows[0]["timestamp"],
            "checkpoint": "de-en-translator/r2",
            "validation_loss": "1,25",
            "dataset": "europarl/train",
            "parent": "de-en-translator/r1",
            "git_commit": "test-commit-abcdef",
        }
    ]
