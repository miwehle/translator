"""Public Translator2 API."""

from __future__ import annotations

import csv
import json
import subprocess
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from .data_prod import DatasetMetadata, load_arrow_records
from .train_prod import (
    DataLoaderConfig,
    ModelConfig,
    TrainConfig,
    check_dataset,
)
from .train_prod.factory import Factory
from .train_prod.training import Trainer
from .types import Example


def _git_head(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _write_run_config(
    run_dir: Path,
    payload: dict[str, object],
    *,
    build_commit: str,
) -> None:
    (run_dir / "config.json").write_text(
        json.dumps(
            {**payload, "build_commit": build_commit},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _append_checkpoint_register(
    register_path: Path,
    *,
    timestamp: str,
    dataset_path: str,
    git_commit: str,
    output_ckpt: str,
) -> None:
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
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": timestamp,
                "input_ckpt": "",
                "dataset_path": dataset_path,
                "git_commit": git_commit,
                "output_ckpt": output_ckpt,
            }
        )


def train(
    *,
    dataset_path: str | Path,
    train_config: TrainConfig,
    model_config: ModelConfig = ModelConfig(),
    data_loader_config: DataLoaderConfig = DataLoaderConfig(),
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    
    def prepare_training() -> tuple[Sequence[Example], DatasetMetadata, str]:
        dataset_dir = Path(dataset_path)
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_dir}")

        run_dir = Path(train_config.runs_dir) / train_config.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        resolved_repo_root = (
            Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
        )
        git_commit = _git_head(resolved_repo_root)
        _write_run_config(
            run_dir,
            {
                "dataset_path": str(dataset_dir),
                "model_config": asdict(model_config),
                "train_config": asdict(train_config),
                "data_loader_config": asdict(data_loader_config),
            },
            build_commit=git_commit,
        )

        examples = cast(Sequence[Example], load_arrow_records(dataset_dir))
        metadata = DatasetMetadata.from_file(dataset_dir / "dataset_manifest.yaml")

        return examples, metadata, git_commit


    examples, metadata, git_commit = prepare_training()

    summary = Trainer(Factory(metadata)).train(
        examples,
        train_config=train_config,
        model_config=model_config,
        data_loader_config=data_loader_config,
    )

    _append_checkpoint_register(
        Path(train_config.runs_dir) / "checkpoint_register.csv",
        timestamp=datetime.now().isoformat(timespec="seconds"),
        dataset_path=str(Path(dataset_path)),
        git_commit=git_commit,
        output_ckpt=summary["checkpoint_path"],
    )
    return summary
