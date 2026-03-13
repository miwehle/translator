from __future__ import annotations

import csv
from pathlib import Path

FIELDNAMES = [
    "timestamp",
    "input_ckpt",
    "dataset_path",
    "build_commit",
    "output_ckpt",
]


def insert(
    *,
    register_path: str | Path,
    timestamp: str,
    input_ckpt: str,
    dataset_path: str,
    build_commit: str,
    output_ckpt: str,
) -> Path:
    register_file = Path(register_path)
    register_file.parent.mkdir(parents=True, exist_ok=True)
    write_header = not register_file.exists()

    with register_file.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": timestamp,
                "input_ckpt": input_ckpt,
                "dataset_path": dataset_path,
                "build_commit": build_commit,
                "output_ckpt": output_ckpt,
            }
        )

    return register_file
