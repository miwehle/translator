from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

from lab_infrastructure.compute_metrics import detect_compute_hardware
from lab_infrastructure.logging import get_logger
from lab_infrastructure.run_config import git_head_commit, write_run_config

from ...registers import append_experiment_register
from ...shared import Example
from ..config import TrainConfig, TrainRunConfig
from ..dataset import DatasetMetadata, load_arrow_records

logger = logging.getLogger(__name__)
_RUN_ID_RE = re.compile(r"^r(?P<run_id>\d+)$")
_EXPERIMENT_ID_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")


def _load_dataset(dataset_path: str | Path) -> tuple[Sequence[Example], DatasetMetadata]:
    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")

    logger.info("Load arrow dataset from %s", dataset_dir)
    examples = cast(Sequence[Example], load_arrow_records(dataset_dir))
    metadata = DatasetMetadata.from_file(dataset_dir / "dataset_manifest.yaml")
    return examples, metadata


def preprocess(
    config: TrainRunConfig,
) -> tuple[Sequence[Example], DatasetMetadata, str, TrainConfig, str | None, Sequence[Example] | None]:
    """Prepare this training run.

    Determine whether training starts from scratch with a new model or resumes from a checkpoint,
    load the dataset, create the output run directory, and record the effective run configuration.
    """

    def experiment_scope() -> str | None:
        experiment_id = config.train_config.experiment_id
        if experiment_id is None:
            return None
        if _EXPERIMENT_ID_RE.fullmatch(experiment_id) is None:
            raise ValueError(f"experiment_id must be a lowercase slug, got: {experiment_id}")
        return experiment_id

    def existing_run_ids(experiment_dir: Path) -> list[int]:
        if not experiment_dir.exists():
            return []
        return [
            int(match.group("run_id"))
            for child in experiment_dir.iterdir()
            if child.is_dir() and (match := _RUN_ID_RE.fullmatch(child.name)) is not None
        ]

    def run_scope_dir(scope_name: str | None) -> Path:
        return (
            config.train_config.training_runs_dir / scope_name
            if scope_name is not None
            else config.train_config.training_runs_dir
        )

    def format_run_ref(scope_name: str | None, run_id: int) -> str:
        return f"{scope_name}/r{run_id}" if scope_name is not None else f"r{run_id}"

    def create_next_run_dir() -> tuple[Path, str]:
        scope_name = experiment_scope()
        scope_dir = run_scope_dir(scope_name)
        scope_is_new = not scope_dir.exists()
        scope_dir.mkdir(parents=True, exist_ok=True)
        if scope_name is not None and scope_is_new:
            append_experiment_register(config.train_config.training_runs_dir, experiment_id=scope_name)

        run_id = max(existing_run_ids(scope_dir), default=0) + 1
        while True:
            ref = format_run_ref(scope_name, run_id)
            run_dir = config.train_config.training_runs_dir / ref
            try:
                run_dir.mkdir(exist_ok=False)
                return run_dir, ref
            except FileExistsError:
                run_id += 1

    def determine_resume_run() -> str | None:
        def resolve_run_ref(run_ref: str) -> None:
            parts = run_ref.split("/")
            if len(parts) > 2 or _RUN_ID_RE.fullmatch(parts[-1]) is None:
                raise ValueError(f"resume_run must use experiment-id/rN or rN format, got: {run_ref}")
            if len(parts) == 2 and _EXPERIMENT_ID_RE.fullmatch(parts[0]) is None:
                raise ValueError(f"resume_run must use experiment-id/rN or rN format, got: {run_ref}")
            if not (config.train_config.training_runs_dir / run_ref).is_dir():
                raise FileNotFoundError(f"Resume run not found: {run_ref}")

        def latest_run_ref() -> str:
            scope_name = experiment_scope()
            scope_dir = run_scope_dir(scope_name)
            run_id = max(existing_run_ids(scope_dir), default=0)
            if run_id == 0:
                scope_label = scope_name or config.train_config.training_runs_dir.name
                raise FileNotFoundError(f"Cannot resume latest run from {scope_label}: no runs found.")
            return format_run_ref(scope_name, run_id)

        resume_run = config.resume_run
        if config.model_config is None and resume_run is None:
            resume_run = latest_run_ref()
        if resume_run is not None:
            resolve_run_ref(resume_run)
        return resume_run

    def load_validation_dataset(
        train_config: TrainConfig, training_metadata: DatasetMetadata
    ) -> Sequence[Example]:
        validation_dataset = train_config.validation_dataset
        assert validation_dataset is not None
        validation_examples, validation_metadata = _load_dataset(
            train_config.datasets_dir / validation_dataset
        )
        if replace(validation_metadata, num_examples=training_metadata.num_examples) != training_metadata:
            raise ValueError("Validation dataset metadata mismatch.")
        return validation_examples

    def log_training_start(train_config: TrainConfig) -> None:
        resolved_device = train_config.device if train_config.device is not None else "auto"
        logger.info(
            "Start training hardware=%s run_dir=%s epochs=%s batch_size=%s device=%s",
            detect_compute_hardware(),
            train_config.training_runs_dir / train_config.run_name,
            train_config.epochs,
            config.data_loader_config.batch_size,
            resolved_device,
        )

    def write_training_config(run_dir: Path, run_ref: str, resume_run: str | None) -> TrainConfig:
        train_config = replace(config.train_config, run_name=run_ref)
        write_run_config(
            run_dir / "training_config.yaml",
            {
                "model_config": (asdict(config.model_config) if config.model_config is not None else None),
                "resume_run": resume_run,
                "train_config": asdict(train_config),
                "data_loader_config": asdict(config.data_loader_config),
            },
            repo_root=Path(__file__).resolve().parents[3],
            git_key_prefix="translator",
        )
        return train_config

    # check input parameter
    if config.train_config.validate_every is not None and config.train_config.validation_dataset is None:
        raise ValueError("validate_every requires validation_dataset.")
    if config.model_config is not None and config.resume_run is not None:
        raise ValueError("resume_run requires model_config to be omitted.")

    resume_run = determine_resume_run()

    examples, dataset_metadata = _load_dataset(config.train_config.datasets_dir / config.train_config.dataset)

    run_dir, run_ref = create_next_run_dir()
    get_logger("translator", log_path=run_dir / "training.log", stream=False)

    train_config = write_training_config(run_dir, run_ref, resume_run)

    validation_examples = (
        load_validation_dataset(train_config, dataset_metadata)
        if train_config.validation_dataset is not None
        else None
    )

    log_training_start(train_config)
    git_commit = str(git_head_commit(Path(__file__).resolve().parents[3]) or "")
    return (
        examples,
        dataset_metadata,
        git_commit,
        train_config,
        resume_run,
        validation_examples,
    )
