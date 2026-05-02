"""Public Translator API."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

import yaml
from lab_infrastructure.compute_metrics import detect_compute_hardware
from lab_infrastructure.logging import get_logger
from lab_infrastructure.run_config import git_head_commit, write_run_config

from .evaluation.config import CometScoreRunConfig
from .registers import append_checkpoint_register, append_comet_score_register, append_experiment_register
from .shared import Example
from .training import Factory, PreflightConfig, Trainer, TrainingSummary, TrainRunConfig, preflight
from .training.config import TrainConfig
from .training.dataset import DatasetMetadata, load_arrow_records

logger = logging.getLogger(__name__)
_RUN_ID_RE = re.compile(r"^R(?P<run_id>\d{3})$")
_RUN_REF_RE = re.compile(r"^E(?P<experiment_id>\d{3})/R(?P<run_id>\d{3})$")


def _load_dataset(dataset_path: str | Path) -> tuple[Path, Sequence[Example], DatasetMetadata]:
    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")

    logger.info("Load arrow dataset from %s", dataset_dir)
    examples = cast(Sequence[Example], load_arrow_records(dataset_dir))
    metadata = DatasetMetadata.from_file(dataset_dir / "dataset_manifest.yaml")
    return dataset_dir, examples, metadata


def check_dataset(config: PreflightConfig) -> dict[str, object]:
    _, examples, metadata = _load_dataset(config.dataset_path)

    return preflight.check_dataset(
        examples,
        id_field=metadata.id_field,
        src_field=metadata.src_field,
        tgt_field=metadata.tgt_field,
        src_pad_idx=metadata.src_pad_id,
        tgt_pad_idx=metadata.tgt_pad_id,
        require_unique_ids=config.require_unique_ids,
        min_seq_len=config.min_seq_len,
    )


def comet_score(
    config: CometScoreRunConfig,
) -> float:
    from .evaluation import CometScorer

    scorer = CometScorer(
        comet_model=config.model,
        test_dataset=config.dataset_config,
        mapping=config.mapping_config,
        output_path=config.output_path,
    )
    score = scorer.score_checkpoint(config.checkpoint_file)
    summary_path = config.checkpoint_file.parent / "comet_score.yaml"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        yaml.safe_dump({"score": score, "config": asdict(config)}, sort_keys=False), encoding="utf-8"
    )
    append_comet_score_register(
        config.checkpoint_file.parent.parent,
        checkpoint=config.checkpoint,
        eval_dataset=config.dataset_config.data_file or config.dataset_config.path,
        comet_model=config.model,
        comet_score=score,
    )
    return score


def train(config: TrainRunConfig) -> TrainingSummary:
    """Train the translator on `config.train_config.dataset`.

    Use `config.model_config` to start a new run from scratch. Without `model_config`,
    resume either `config.resume_run` or the latest run in `config.train_config.experiment_id`.
    """
    def preprocess():
        """Prepare this training run.

        Determine whether training starts from scratch with a new model or resumes from a checkpoint,
        load the dataset, create the output run directory, and record the effective run configuration.
        """
        def experiment_name() -> str:
            experiment_id = config.train_config.experiment_id
            if not 1 <= experiment_id <= 999:
                raise ValueError("experiment_id must be between 1 and 999.")
            return f"E{experiment_id:03d}"

        def existing_run_ids(experiment_dir: Path) -> list[int]:
            if not experiment_dir.exists():
                return []
            return [
                int(match.group("run_id"))
                for child in experiment_dir.iterdir()
                if child.is_dir() and (match := _RUN_ID_RE.fullmatch(child.name)) is not None
            ]

        def create_next_run_dir() -> tuple[Path, str]:
            experiment_dir = config.train_config.training_runs_dir / experiment_name()
            experiment_is_new = not experiment_dir.exists()
            experiment_dir.mkdir(parents=True, exist_ok=True)
            if experiment_is_new:
                append_experiment_register(
                    config.train_config.training_runs_dir, experiment_id=config.train_config.experiment_id
                )

            run_id = max(existing_run_ids(experiment_dir), default=0) + 1
            while True:
                run_ref = f"{experiment_dir.name}/R{run_id:03d}"
                run_dir = config.train_config.training_runs_dir / run_ref
                try:
                    run_dir.mkdir(exist_ok=False)
                    return run_dir, run_ref
                except FileExistsError:
                    run_id += 1

        def resolve_run_ref(run_ref: str) -> None:
            if _RUN_REF_RE.fullmatch(run_ref) is None:
                raise ValueError(f"resume_run must use E###/R### format, got: {run_ref}")
            if not (config.train_config.training_runs_dir / run_ref).is_dir():
                raise FileNotFoundError(f"Resume run not found: {run_ref}")

        def latest_run_ref() -> str:
            experiment_dir = config.train_config.training_runs_dir / experiment_name()
            run_id = max(existing_run_ids(experiment_dir), default=0)
            if run_id == 0:
                raise FileNotFoundError(f"Cannot resume latest run from {experiment_name()}: no runs found.")
            return f"{experiment_dir.name}/R{run_id:03d}"

        def load_validation_dataset(
            train_config: TrainConfig, training_metadata: DatasetMetadata
        ) -> Sequence[Example]:
            validation_dataset = train_config.validation_dataset
            validation_path = train_config.datasets_dir / validation_dataset
            _, validation_examples, validation_metadata = _load_dataset(validation_path)
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

        # preprocess flow
        logger.info("Prepare training")
        if config.train_config.validate_every is not None and config.train_config.validation_dataset is None:
            raise ValueError("validate_every requires validation_dataset.")
        if config.model_config is not None and config.resume_run is not None:
            raise ValueError("resume_run requires model_config to be omitted.")

        resume_run = config.resume_run
        if config.model_config is None and resume_run is None:
            resume_run = latest_run_ref()
        if resume_run is not None:
            resolve_run_ref(resume_run)

        dataset_path = config.train_config.datasets_dir / config.train_config.dataset
        _, examples, dataset_metadata = _load_dataset(dataset_path)

        run_dir, run_ref = create_next_run_dir()
        train_config = replace(config.train_config, run_name=run_ref)
        get_logger("translator", log_path=run_dir / "training.log", stream=False)
        resolved_repo_root = Path(__file__).resolve().parents[2]
        write_run_config(
            run_dir / "training_config.yaml",
            {
                "model_config": (asdict(config.model_config) if config.model_config is not None else None),
                "resume_run": resume_run,
                "train_config": asdict(train_config),
                "data_loader_config": asdict(config.data_loader_config),
            },
            repo_root=resolved_repo_root,
            git_key_prefix="translator",
        )

        git_commit = str(git_head_commit(resolved_repo_root) or "")
        validation_examples = (
            load_validation_dataset(train_config, dataset_metadata)
            if train_config.validation_dataset is not None
            else None
        )
        log_training_start(train_config)
        return (
            examples,
            dataset_metadata,
            git_commit,
            train_config,
            resume_run,
            validation_examples,
        )

    def postprocess(
        summary: TrainingSummary,
        trainer: Trainer,
        validation_examples: Sequence[Example] | None,
        train_config: TrainConfig,
        git_commit: str,
        resume_run: str | None,
    ) -> TrainingSummary:
        """Postprocess this training run.

        Run final validation if configured, write the training summary, update the
        checkpoint register, and log completion.
        """
        def log_training_finish(summary: TrainingSummary) -> None:
            logger.info(
                "Finished training final_loss=%s validation_loss=%s",
                summary.final_loss,
                summary.validation_loss,
            )

        # postprocess flow
        if validation_examples is not None:
            validation_loss = trainer.validate(validation_examples)
            summary = replace(summary, validation_loss=validation_loss)

        summary_path = (
            train_config.training_runs_dir / train_config.run_name / "training_summary.yaml"
        )
        summary_path.write_text(yaml.safe_dump(asdict(summary), sort_keys=True), encoding="utf-8")
        append_checkpoint_register(
            config.train_config.training_runs_dir,
            checkpoint=resume_run or "",
            dataset_path=config.train_config.dataset,
            git_commit=git_commit,
            output_ckpt=train_config.run_name,
            validation_loss=summary.validation_loss,
        )
        log_training_finish(summary)
        return summary

    # main flow
    examples, dataset_metadata, git_commit, train_config, resume_run, validation_examples = preprocess()
    
    trainer = Trainer(
        Factory(dataset_metadata),
        train_config,
        config.data_loader_config,
        model_config=config.model_config,
        resume_run=resume_run,
    )
    summary = trainer.train(examples, validation_examples)
    
    return postprocess(summary, trainer, validation_examples, train_config, git_commit, resume_run)
