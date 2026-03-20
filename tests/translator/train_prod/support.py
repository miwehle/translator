from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Mapping, cast

from datasets import Dataset
from translator.train_prod.training import TrainConfig

_INITIALIZED_LOG_PATHS: set[Path] = set()


def create_valid_mapped_dataset(dataset_dir: Path) -> Path:
    rows = {
        "id": list(range(512)),
        "src_ids": [[10, 100 + (i % 17), 11] for i in range(512)],
        "tgt_ids": [[2, 20 + (i % 9), 3] for i in range(512)],
    }
    Dataset.from_dict(rows).save_to_disk(str(dataset_dir))
    return dataset_dir


def pad_index_from_records(dataset_path: Path, field: str) -> int:
    from translator.data_prod.arrow_dataset import load_arrow_records

    records = load_arrow_records(dataset_path)
    max_token = -1
    for row in records:
        row_map = cast(Mapping[str, object], row)
        values_obj = row_map.get(field)
        if not isinstance(values_obj, list):
            raise ValueError(f"Expected list[int] in field '{field}', got {type(values_obj).__name__}.")
        values = [int(x) for x in values_obj]
        max_token = max(max_token, max(values))
    return max_token + 1


def train_config_for_test(run_root: str, **overrides: object) -> TrainConfig:
    config = {
        "runs_dir": run_root,
        "run_name": "run1",
        **overrides,
    }
    return TrainConfig(**config)


def log(
    *,
    module_file: Path,
    test_name: str,
    body: str,
) -> Path:
    """Append a test section to the module log in the caller's package-local `.log` directory.

    The log file is written to `<module package>/.log/<module name>.log`.
    For example, calls from `tests/translator/train_prod/test_train_prod_loss_progress.py`
    write to `tests/translator/train_prod/.log/test_train_prod_loss_progress.log`.

    The first write to a given module log file during the current pytest process truncates
    the file; subsequent writes append a blank line and then a new section headed by a
    timestamp and the test function name.
    """
    package_dir = module_file.resolve().parent
    log_dir = package_dir / ".log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{module_file.stem}.log"

    mode = "a" if log_path in _INITIALIZED_LOG_PATHS else "w"
    _INITIALIZED_LOG_PATHS.add(log_path)

    with log_path.open(mode, encoding="utf-8") as f:
        if mode == "a":
            f.write("\n")
        f.write(
            f"=== {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
            f"{test_name} ===\n"
        )
        f.write(body)
        if not body.endswith("\n"):
            f.write("\n")

    return log_path
