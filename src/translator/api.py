"""Public Translator API."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

import yaml
from lab_infrastructure.compute_metrics import detect_compute_hardware
from lab_infrastructure.logging import get_logger
from lab_infrastructure.run_config import git_head_commit, write_run_config

from .evaluation.config import CometScoreConfig
from .registers import append_checkpoint_register, append_comet_score_register
from .shared import Example
from .training import DataLoaderConfig, Factory, ModelConfig, TrainConfig, Trainer, TrainingSummary, preflight
from .training.dataset import DatasetMetadata, load_arrow_records

logger = logging.getLogger(__name__)


def _write_training_summary(summary_path: Path, summary: TrainingSummary) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(yaml.safe_dump(asdict(summary), sort_keys=True), encoding="utf-8")


def _write_comet_score(summary_path: Path, score: float, config: CometScoreConfig) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(yaml.safe_dump({"score": score, "config": asdict(config)}, sort_keys=False), encoding="utf-8")


def _next_available_run_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        return base_dir

    i = 1
    while True:
        candidate = base_dir.with_name(f"{base_dir.name} ({i})")
        if not candidate.exists():
            return candidate
        i += 1


def _load_dataset(dataset_path: str | Path) -> tuple[Path, Sequence[Example], DatasetMetadata]:
    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")

    logger.info("Load arrow dataset from %s", dataset_dir)
    examples = cast(Sequence[Example], load_arrow_records(dataset_dir))
    metadata = DatasetMetadata.from_file(dataset_dir / "dataset_manifest.yaml")
    return dataset_dir, examples, metadata


def check_dataset(
    *, dataset_path: str | Path, require_unique_ids: bool, min_seq_len: int
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


def comet_score(
    config: CometScoreConfig,
) -> float:
    from .evaluation import CometScorer

    scorer = CometScorer(
        comet_model=config.model,
        test_dataset=config.dataset_config,
        mapping=config.mapping_config,
        output_path=config.output_path,
    )
    score = scorer.score_checkpoint(config.checkpoint_file)
    _write_comet_score(config.checkpoint_file.parent / "comet_score.yaml", score, config)
    append_comet_score_register(
        config.checkpoint_file.parent.parent,
        checkpoint=config.checkpoint,
        eval_dataset=config.dataset_config.path,
        comet_model=config.model,
        comet_score=score,
    )
    return score


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

    def prepare_training():
        logger.info("Prepare training")
        dataset_path = train_config.datasets_dir / train_config.dataset
        _, examples, metadata = _load_dataset(dataset_path)

        # Avoid overwriting earlier runs by picking the next free run directory.
        run_dir = _next_available_run_dir(train_config.training_runs_dir / train_config.run_name)
        run_dir.mkdir(parents=True, exist_ok=False)
        resolved_train_config = replace(train_config, run_name=run_dir.name)
        get_logger("translator", log_path=run_dir / "training.log", stream=False)
        resolved_repo_root = Path(repo_root)
        write_run_config(
            run_dir / "training_config.yaml",
            {
                "model_config": (asdict(model_config) if model_config is not None else None),
                "resume_run": resume_run,
                "train_config": asdict(resolved_train_config),
                "data_loader_config": asdict(data_loader_config),
            },
            repo_root=resolved_repo_root,
            git_key_prefix="translator",
        )

        return examples, metadata, str(git_head_commit(resolved_repo_root) or ""), resolved_train_config

    def load_validation_dataset(
        resolved_train_config: TrainConfig, training_metadata: DatasetMetadata
    ) -> Sequence[Example]:
        validation_dataset = resolved_train_config.validation_dataset
        validation_path = resolved_train_config.datasets_dir / validation_dataset
        _, validation_examples, validation_metadata = _load_dataset(validation_path)
        if replace(validation_metadata, num_examples=training_metadata.num_examples) != training_metadata:
            raise ValueError("Validation dataset metadata mismatch.")
        return validation_examples

    def log_training_start(resolved_train_config: TrainConfig) -> None:
        resolved_device = resolved_train_config.device if resolved_train_config.device is not None else "auto"
        logger.info(
            "Start training hardware=%s run_dir=%s epochs=%s batch_size=%s device=%s",
            detect_compute_hardware(),
            resolved_train_config.training_runs_dir / resolved_train_config.run_name,
            resolved_train_config.epochs,
            data_loader_config.batch_size,
            resolved_device,
        )

    def log_training_finish(summary: TrainingSummary) -> None:
        logger.info(
            "Finished training final_loss=%s validation_loss=%s", summary.final_loss, summary.validation_loss
        )

    # main flow
    examples, metadata, git_commit, resolved_train_config = prepare_training()
    validation_examples = None
    if resolved_train_config.validate_every is not None and resolved_train_config.validation_dataset is None:
        raise ValueError("validate_every requires validation_dataset.")
    if resolved_train_config.validation_dataset is not None:
        validation_examples = load_validation_dataset(resolved_train_config, metadata)
    log_training_start(resolved_train_config)

    # core: train
    trainer = Trainer(
        Factory(metadata),
        resolved_train_config,
        data_loader_config,
        model_config=model_config,
        resume_run=resume_run,
    )
    summary = trainer.train(examples, validation_examples)

    # evaluate
    if validation_examples is not None:
        validation_loss = trainer.validate(validation_examples)
        summary = replace(summary, validation_loss=validation_loss)

    summary_path = (
        resolved_train_config.training_runs_dir / resolved_train_config.run_name / "training_summary.yaml"
    )
    _write_training_summary(summary_path, summary)
    append_checkpoint_register(
        train_config.training_runs_dir,
        checkpoint=resume_run or "",
        dataset_path=Path(train_config.dataset).name,
        git_commit=git_commit,
        output_ckpt=resolved_train_config.run_name,
        validation_loss=summary.validation_loss,
    )
    log_training_finish(summary)

    return summary
