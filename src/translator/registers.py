from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


def append_checkpoint_register(
    register_dir: Path,
    *,
    checkpoint: str,
    dataset_path: str,
    git_commit: str,
    output_ckpt: str,
    validation_loss: float | None,
) -> None:
    register_path = register_dir / "checkpoint_register.csv"
    write_header = not register_path.exists()
    with register_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "input_ckpt",
                "dataset_path",
                "git_commit",
                "output_ckpt",
                "validation_loss",
            ],
            delimiter=";",
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "input_ckpt": checkpoint,
                "dataset_path": dataset_path,
                "git_commit": git_commit[:20],
                "output_ckpt": output_ckpt,
                "validation_loss": (
                    "" if validation_loss is None else _format_decimal_for_sheet(validation_loss)
                ),
            }
        )


def append_experiment_register(register_dir: Path, *, experiment_id: str) -> None:
    register_path = register_dir / "experiment_register.csv"
    write_header = not register_path.exists()
    with register_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment_id", "timestamp", "notes"], delimiter=";")
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "experiment_id": experiment_id,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "notes": "",
            }
        )


def append_comet_score_register(
    register_dir: Path, *, checkpoint: str, eval_dataset: str, comet_model: str, comet_score: float
) -> None:
    register_path = register_dir / "comet_score_register.csv"
    write_header = not register_path.exists()
    with register_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["timestamp", "checkpoint", "eval_dataset", "comet_model", "comet_score"],
            delimiter=";",
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "checkpoint": checkpoint,
                "eval_dataset": eval_dataset,
                "comet_model": comet_model,
                "comet_score": _format_decimal_for_sheet(comet_score),
            }
        )


def _format_decimal_for_sheet(value: float) -> str:
    return str(value).replace(".", ",")
