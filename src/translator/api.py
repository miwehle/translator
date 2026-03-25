"""Public Translator2 API."""

from __future__ import annotations

import csv
import logging
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import cast

import yaml

from translator.training.dataset import DatasetMetadata, load_arrow_records

from .shared import Example, configure_translator_logging, detect_hardware_type
from .training import (
    DataLoaderConfig,
    Factory,
    ModelConfig,
    TrainConfig,
    Trainer,
    TrainingSummary,
    preflight,
)

logger = logging.getLogger(__name__)


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
    (run_dir / "train_config.yaml").write_text(
        yaml.safe_dump(
            {**payload, "build_commit": build_commit},
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _next_available_run_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        return base_dir

    i = 1
    while True:
        candidate = base_dir.with_name(f"{base_dir.name} ({i})")
        if not candidate.exists():
            return candidate
        i += 1


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


def _load_dataset(dataset_path: str | Path) -> tuple[Path, Sequence[Example], DatasetMetadata]:
    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")

    logger.info("Load arrow dataset from %s", dataset_dir)
    examples = cast(Sequence[Example], load_arrow_records(dataset_dir))
    metadata = DatasetMetadata.from_file(dataset_dir / "dataset_manifest.yaml")
    return dataset_dir, examples, metadata


def check_dataset(
    *,
    dataset_path: str | Path,
    require_unique_ids: bool,
    min_seq_len: int,
) -> dict[str, object]:
    _, examples, metadata = _load_dataset(dataset_path)

    return preflight.check_dataset(
        examples,
        id_field=metadata.id_field,
        src_field=metadata.src_field,
        tgt_field=metadata.tgt_field,
        src_pad_idx=metadata.src_pad_id,
        tgt_pad_idx=metadata.tgt_pad_id,
        require_unique_ids=require_unique_ids,
        min_seq_len=min_seq_len,
    )


def train(
    *,
    dataset_path: str | Path,
    train_config: TrainConfig,
    model_config: ModelConfig = ModelConfig(),
    data_loader_config: DataLoaderConfig = DataLoaderConfig(),
    repo_root: str | Path | None = None,
) -> TrainingSummary:
    
    def write_summary_yaml(summary_path: Path, summary: TrainingSummary) -> None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            yaml.safe_dump(asdict(summary), sort_keys=True),
            encoding="utf-8",
        )

    def prepare_training() -> tuple[Sequence[Example], DatasetMetadata, str, TrainConfig]:
        logger.info("Prepare training")
        dataset_dir, examples, metadata = _load_dataset(dataset_path)

        # Avoid overwriting earlier runs by picking the next free run directory.
        run_dir = _next_available_run_dir(Path(train_config.runs_dir) / train_config.run_name)
        run_dir.mkdir(parents=True, exist_ok=False)
        resolved_train_config = replace(train_config, run_name=run_dir.name)
        configure_translator_logging(log_path=run_dir / "training.log")
        resolved_repo_root = (
            Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
        )
        git_commit = _git_head(resolved_repo_root)
        _write_run_config(
            run_dir,
            {
                "dataset_path": str(dataset_dir),
                "model_config": asdict(model_config),
                "train_config": asdict(resolved_train_config),
                "data_loader_config": asdict(data_loader_config),
            },
            build_commit=git_commit,
        )

        return examples, metadata, git_commit, resolved_train_config

    def log_training_start(resolved_train_config: TrainConfig) -> None:
        resolved_device = (
            resolved_train_config.device if resolved_train_config.device is not None else "auto"
        )
        logger.info(
            "Start training hardware=%s run_dir=%s epochs=%s batch_size=%s device=%s",
            detect_hardware_type(),
            Path(resolved_train_config.runs_dir) / resolved_train_config.run_name,
            resolved_train_config.epochs,
            data_loader_config.batch_size,
            resolved_device,
        )

    def log_training_finish(summary: TrainingSummary, summary_path: Path) -> None:
        logger.info(
            "Finished training final_loss=%s checkpoint_path=%s summary_path=%s",
            summary.final_loss,
            summary.checkpoint_path,
            summary_path,
        )
        logger.info("Registered checkpoint output_ckpt=%s", summary.checkpoint_path)

    examples, metadata, git_commit, resolved_train_config = prepare_training()
    log_training_start(resolved_train_config)

    summary = Trainer(Factory(metadata)).train(
        examples,
        train_config=resolved_train_config,
        model_config=model_config,
        data_loader_config=data_loader_config,
    )

    summary_path = Path(resolved_train_config.runs_dir) / resolved_train_config.run_name / "summary.yaml"
    write_summary_yaml(summary_path, summary)
    _append_checkpoint_register(
        Path(train_config.runs_dir) / "checkpoint_register.csv",
        timestamp=datetime.now().isoformat(timespec="seconds"),
        dataset_path=str(Path(dataset_path)),
        git_commit=git_commit,
        output_ckpt=summary.checkpoint_path,
    )
    log_training_finish(summary, summary_path)

    return summary
