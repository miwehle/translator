"""Public Translator API."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict, replace

import yaml

from .evaluation.config import CometScoreRunConfig
from .registers import append_checkpoint_register, append_comet_score_register
from .shared import Example
from .training import Factory, PreflightCheckRunConfig, Trainer, TrainingSummary, TrainRunConfig, preflight
from .training.internal.preprocessing import _load_dataset, preprocess

logger = logging.getLogger(__name__)


def preflight_check(config: PreflightCheckRunConfig) -> dict[str, object]:
    examples, metadata = _load_dataset(config.dataset_path)

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


def comet_score(config: CometScoreRunConfig) -> float:
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
    """Train the translator on `config.dataset`.

    Use `config.model_config` to start a new run from scratch. Without `model_config`,
    resume either `config.parent_checkpoint` or the latest run in the configured run scope.
    """
    def postprocess(
        summary: TrainingSummary,
        trainer: Trainer,
        validation_examples: Sequence[Example] | None,
        train_config: TrainRunConfig,
        git_commit: str,
        parent_checkpoint: str | None,
    ) -> TrainingSummary:
        """Postprocess this training run.

        Run final validation if configured, write the training summary, update the
        checkpoint register, and log completion.
        """
        def log_training_finish(summary: TrainingSummary) -> None:
            logger.info(
                "Finished training loss=%s val_loss=%s", summary.final_loss, summary.validation_loss
            )

        if validation_examples is not None:
            validation_loss = trainer.validate(validation_examples)
            summary = replace(summary, validation_loss=validation_loss)

        summary_path = train_config.training_runs_dir / train_config.run_name / "training_summary.yaml"
        summary_path.write_text(yaml.safe_dump(asdict(summary), sort_keys=True), encoding="utf-8")
        append_checkpoint_register(
            train_config.training_runs_dir,
            checkpoint=train_config.run_name,
            dataset=train_config.dataset,
            parent=parent_checkpoint or "",
            git_commit=git_commit,
            validation_loss=summary.validation_loss,
        )
        log_training_finish(summary)
        return summary

    # main flow
    preprocessed = preprocess(config)
    examples, metadata, git_commit, train_config, parent_checkpoint, validation_examples = preprocessed

    trainer = Trainer(Factory(metadata), train_config)
    summary = trainer.train(examples, validation_examples)

    return postprocess(summary, trainer, validation_examples, train_config, git_commit, parent_checkpoint)
