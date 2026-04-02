"""Public Translator API."""

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
    (run_dir / "training_config.yaml").write_text(
        yaml.safe_dump(
            {**payload, "build_commit": build_commit},
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_training_summary(
    summary_path: Path,
    summary: TrainingSummary,
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        yaml.safe_dump(asdict(summary), sort_keys=True),
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


def _load_dataset(dataset_path: str | Path
) -> tuple[Path, Sequence[Example], DatasetMetadata]:
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
    train_config: TrainConfig,
    data_loader_config: DataLoaderConfig,
    repo_root: str | Path,
    *,
    model_config: ModelConfig | None = None,
    resume_run: str | None = None,
) -> TrainingSummary:
    """Train the translator on `train_config.dataset`.

    Use `model_config` to start a new run from scratch. Use `resume_run` to
    continue a previous run from its checkpoint.
    """

    def append_checkpoint_register(
        output_run: str,
        validation_loss: float | None,
    ) -> None:
        register_path = train_config.training_runs_dir / "checkpoint_register.csv"
        write_header = not register_path.exists()
        with register_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "timestamp", "input_ckpt", "dataset_path",
                    "git_commit", "output_ckpt", "validation_loss",
                ],
            )
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "input_ckpt": resume_run or "",
                    "dataset_path": Path(train_config.dataset).name,
                    "git_commit": git_commit[:20],
                    "output_ckpt": output_run,
                    "validation_loss": (
                        "" if validation_loss is None else validation_loss
                    ),
                }
            )

    def prepare_training():
        logger.info("Prepare training")
        dataset_path = train_config.datasets_dir / train_config.dataset
        _, examples, metadata = _load_dataset(dataset_path)

        # Avoid overwriting earlier runs by picking the next free run directory.
        run_dir = _next_available_run_dir(
            train_config.training_runs_dir / train_config.run_name
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        resolved_train_config = replace(train_config, run_name=run_dir.name)
        configure_translator_logging(log_path=run_dir / "training.log")
        resolved_repo_root = Path(repo_root)
        git_commit = _git_head(resolved_repo_root)
        _write_run_config(
            run_dir,
            {
                "model_config": (
                    asdict(model_config) if model_config is not None else None
                ),
                "resume_run": resume_run,
                "train_config": asdict(resolved_train_config),
                "data_loader_config": asdict(data_loader_config),
            },
            build_commit=git_commit,
        )

        return examples, metadata, git_commit, resolved_train_config

    def load_validation_dataset(
        resolved_train_config: TrainConfig,
        training_metadata: DatasetMetadata,
    ) -> Sequence[Example]:
        validation_dataset = resolved_train_config.validation_dataset
        if validation_dataset is None:
            raise ValueError("validation_dataset is not configured.")
        validation_path = resolved_train_config.datasets_dir / validation_dataset
        _, validation_examples, validation_metadata = _load_dataset(validation_path)
        if (
            replace(validation_metadata, num_examples=training_metadata.num_examples)
            != training_metadata
        ):
            raise ValueError("Validation dataset metadata mismatch.")
        return validation_examples

    def log_training_start(resolved_train_config: TrainConfig) -> None:
        resolved_device = (
            resolved_train_config.device
            if resolved_train_config.device is not None
            else "auto"
        )
        logger.info(
            "Start training hardware=%s run_dir=%s epochs=%s batch_size=%s device=%s",
            detect_hardware_type(),
            resolved_train_config.training_runs_dir / resolved_train_config.run_name,
            resolved_train_config.epochs,
            data_loader_config.batch_size,
            resolved_device,
        )

    def log_training_finish(summary: TrainingSummary) -> None:
        if summary.validation_loss is not None:
            logger.info(
                "Finished training final_loss=%s validation_loss=%s",
                summary.final_loss,
                summary.validation_loss,
            )
            return
        logger.info("Finished training final_loss=%s", summary.final_loss)

    # main flow
    examples, metadata, git_commit, resolved_train_config = prepare_training()
    log_training_start(resolved_train_config)

    validation_examples = None
    if resolved_train_config.validation_dataset is not None:
        validation_examples = load_validation_dataset(resolved_train_config, metadata)

    # core
    trainer = Trainer(
        Factory(metadata), resolved_train_config, data_loader_config,
        model_config=model_config, resume_run=resume_run,
    )
    summary = trainer.train(examples)

    if validation_examples is not None:
        validation_loss = trainer.evaluate(validation_examples)
        summary = replace(summary, validation_loss=validation_loss)

    summary_path = (
        resolved_train_config.training_runs_dir
        / resolved_train_config.run_name
        / "training_summary.yaml"
    )
    _write_training_summary(summary_path, summary)
    append_checkpoint_register(resolved_train_config.run_name, summary.validation_loss)
    log_training_finish(summary)

    return summary
