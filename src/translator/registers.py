from __future__ import annotations

from datetime import datetime
from pathlib import Path

from lab_infrastructure.register_csv import insert_row


def register_checkpoint(
    register_dir: Path,
    *,
    checkpoint: str,
    dataset: str,
    parent: str,
    git_commit: str,
    validation_loss: float | None,
) -> None:
    insert_row(
        register_dir / "checkpoint_register.csv",
        ["checkpoint", "validation_loss", "dataset", "parent", "timestamp", "git_commit"],
        {
            "checkpoint": checkpoint,
            "validation_loss": "" if validation_loss is None else _format_decimal_for_sheet(validation_loss),
            "dataset": dataset,
            "parent": parent,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "git_commit": git_commit[:20],
        },
        artifact_key="checkpoint",
    )


def register_comet_score(
    register_dir: Path, *, checkpoint: str, eval_dataset: str, comet_model: str, comet_score: float
) -> None:
    insert_row(
        register_dir / "comet_score_register.csv",
        ["timestamp", "checkpoint", "eval_dataset", "comet_model", "comet_score"],
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "checkpoint": checkpoint,
            "eval_dataset": eval_dataset,
            "comet_model": comet_model,
            "comet_score": _format_decimal_for_sheet(comet_score),
        },
        artifact_key="checkpoint",
    )


def _format_decimal_for_sheet(value: float) -> str:
    return str(value).replace(".", ",")
