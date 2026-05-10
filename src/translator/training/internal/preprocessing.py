from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

from lab_infrastructure.artifact_paths import artifact_ref, next_numbered_path
from lab_infrastructure.compute_metrics import detect_compute_hardware
from lab_infrastructure.logging import get_logger
from lab_infrastructure.run_config import git_head_commit, write_run_config

from ...shared import Example
from ..config import TrainRunConfig
from ..dataset import DatasetMetadata, load_arrow_records

logger = logging.getLogger(__name__)
_RUN_ID_RE = re.compile(r"^r(?P<run_id>\d+)$")
_WORK_DIR_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")


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
) -> tuple[Sequence[Example], DatasetMetadata, str, TrainRunConfig, str | None, Sequence[Example] | None]:
    """Prepare this training run.

    Determine whether training starts from scratch with a new model or resumes from a parent
    checkpoint, load the dataset, create the output run directory, and record the effective run
    configuration.
    """

    def work_dir() -> str | None:
        configured_work_dir = config.work_dir
        if configured_work_dir is None:
            return None
        if _WORK_DIR_RE.fullmatch(configured_work_dir) is None:
            raise ValueError(f"work_dir must be a lowercase slug, got: {configured_work_dir}")
        return configured_work_dir

    def existing_run_ids(work_dir_path: Path) -> list[int]:
        if not work_dir_path.exists():
            return []
        return [
            int(match.group("run_id"))
            for child in work_dir_path.iterdir()
            if child.is_dir() and (match := _RUN_ID_RE.fullmatch(child.name)) is not None
        ]

    def run_scope_dir(work_dir_name: str | None) -> Path:
        return (
            config.training_runs_dir / work_dir_name
            if work_dir_name is not None
            else config.training_runs_dir
        )

    def create_next_run_dir() -> tuple[Path, str]:
        work_dir_name = work_dir()
        scope_dir = run_scope_dir(work_dir_name)
        scope_dir.mkdir(parents=True, exist_ok=True)

        while True:
            run_dir = next_numbered_path(scope_dir, "r")
            try:
                run_dir.mkdir(exist_ok=False)
                return run_dir, artifact_ref(config.training_runs_dir, run_dir)
            except FileExistsError:
                pass

    def determine_parent_checkpoint() -> str | None:
        def resolve_parent_checkpoint_ref(parent_checkpoint_ref: str) -> None:
            parts = parent_checkpoint_ref.split("/")
            if len(parts) > 2 or _RUN_ID_RE.fullmatch(parts[-1]) is None:
                raise ValueError(
                    f"parent_checkpoint must use work_dir/rN or rN format, got: {parent_checkpoint_ref}"
                )
            if len(parts) == 2 and _WORK_DIR_RE.fullmatch(parts[0]) is None:
                raise ValueError(
                    f"parent_checkpoint must use work_dir/rN or rN format, got: {parent_checkpoint_ref}"
                )
            if not (config.training_runs_dir / parent_checkpoint_ref).is_dir():
                raise FileNotFoundError(f"Parent checkpoint not found: {parent_checkpoint_ref}")

        def latest_run_ref() -> str:
            work_dir_name = work_dir()
            scope_dir = run_scope_dir(work_dir_name)
            run_id = max(existing_run_ids(scope_dir), default=0)
            if run_id == 0:
                scope_label = work_dir_name or config.training_runs_dir.name
                raise FileNotFoundError(f"Cannot resume latest run from {scope_label}: no runs found.")
            return artifact_ref(config.training_runs_dir, scope_dir / f"r{run_id}")

        parent_checkpoint = config.parent_checkpoint
        if config.model_config is None and parent_checkpoint is None:
            parent_checkpoint = latest_run_ref()
        if parent_checkpoint is not None:
            resolve_parent_checkpoint_ref(parent_checkpoint)
        return parent_checkpoint

    def load_validation_dataset(
        train_config: TrainRunConfig, training_metadata: DatasetMetadata
    ) -> Sequence[Example]:
        validation_dataset = train_config.validation_dataset
        assert validation_dataset is not None
        validation_examples, validation_metadata = _load_dataset(
            train_config.datasets_dir / validation_dataset
        )
        if replace(validation_metadata, num_examples=training_metadata.num_examples) != training_metadata:
            raise ValueError("Validation dataset metadata mismatch.")
        return validation_examples

    def log_training_start(train_config: TrainRunConfig) -> None:
        resolved_device = train_config.device if train_config.device is not None else "auto"
        logger.info(
            "Start training hardware=%s run_dir=%s epochs=%s batch_size=%s device=%s",
            detect_compute_hardware(),
            train_config.training_runs_dir / train_config.run_name,
            train_config.epochs,
            config.data_loader_config.batch_size,
            resolved_device,
        )

    def write_training_config(run_dir: Path, run_ref: str, parent_checkpoint: str | None) -> TrainRunConfig:
        train_config = replace(config, run_name=run_ref, parent_checkpoint=parent_checkpoint)
        write_run_config(
            run_dir / "training_config.yaml",
            asdict(train_config),
            repo_root=Path(__file__).resolve().parents[3],
            git_key_prefix="translator",
        )
        return train_config

    # check input parameter
    if config.validate_every is not None and config.validation_dataset is None:
        raise ValueError("validate_every requires validation_dataset.")
    if config.model_config is not None and config.parent_checkpoint is not None:
        raise ValueError("parent_checkpoint requires model_config to be omitted.")

    parent_checkpoint = determine_parent_checkpoint()

    examples, dataset_metadata = _load_dataset(config.datasets_dir / config.dataset)

    run_dir, run_ref = create_next_run_dir()
    get_logger("translator", log_path=run_dir / "training.log", stream=False)

    train_config = write_training_config(run_dir, run_ref, parent_checkpoint)

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
        parent_checkpoint,
        validation_examples,
    )
