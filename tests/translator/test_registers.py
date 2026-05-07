from __future__ import annotations

import csv
from pathlib import Path

from translator.registers import append_experiment_register


def test_append_experiment_register_writes_experiment_column(tmp_path: Path) -> None:
    append_experiment_register(tmp_path, experiment="de-en-translator")

    with tmp_path.joinpath("experiment_register.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))

    assert rows == [{"experiment": "de-en-translator", "timestamp": rows[0]["timestamp"], "notes": ""}]
