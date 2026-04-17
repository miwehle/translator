"""Training orchestration.

Keeps the core train loop focused while adding:
- easy configuration of model, data loading and training via simple dataclasses,
- checkpointing via save and resume, and
- logging via an observer-style helper.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..inference import Translator
from ..inference.tokenizer import create_tokenizer
from ..model import Seq2Seq
from ..shared import Example
from .config import DataLoaderConfig, ModelConfig, TrainConfig
from .internal.checkpointing import load as load_checkpoint
from .internal.checkpointing import save as save_checkpoint
from .internal.factory import Factory
from .internal.training_observer import TrainingObserver

logger = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


@dataclass(frozen=True)
class TrainingSummary:
    num_examples: int
    final_loss: float | None
    checkpoint_path: str
    validation_loss: float | None


class Trainer:
    def __init__(
        self,
        factory: Factory,
        train_config: TrainConfig,
        data_loader_config: DataLoaderConfig = DataLoaderConfig(),
        model_config: ModelConfig | None = None,
        resume_run: str | None = None,
    ) -> None:
        """Create a trainer in one of two modes.

        Use `model_config` to train from scratch. Use `resume_run` to resume from a
        previous run.
        """

        def validate_dataset_max_seq_len(max_seq_len: int) -> None:
            configured_max_seq_len = getattr(factory.dataset_metadata, "configured_max_seq_len", None)
            if configured_max_seq_len is None or configured_max_seq_len <= max_seq_len:
                return
            message = (
                "Dataset configured_max_seq_len exceeds model max_seq_len: "
                "configured_max_seq_len="
                f"{configured_max_seq_len} max_seq_len={max_seq_len}"
            )
            if train_config.force:
                logger.warning("%s; continue because force=True.", message)
                return
            raise ValueError(f"{message}. Set train_config.force=True to continue anyway.")

        self._factory = factory
        self._train_config = train_config
        self._data_loader_config = data_loader_config

        if train_config.eval_every is not None and train_config.validation_dataset is None:
            raise ValueError("eval_every requires validation_dataset.")

        if (model_config is None) == (resume_run is None):
            raise ValueError("Exactly one of model_config or resume_run must be provided.")

        _set_seed(train_config.seed)
        self._device = _resolve_device(train_config.device)
        self._criterion = nn.CrossEntropyLoss(ignore_index=self._factory.dataset_metadata.tgt_pad_id)

        if resume_run is not None:
            loaded = load_checkpoint(
                train_config.training_runs_dir / resume_run / "checkpoint.pt", self._factory, self._device
            )
            validate_dataset_max_seq_len(loaded.model_config.max_seq_len)
            self._model = loaded.model
            self._optimizer = loaded.optimizer
            self._model_config = loaded.model_config
            return

        assert model_config is not None
        validate_dataset_max_seq_len(model_config.max_seq_len)
        self._model_config = model_config
        self._model = self._factory.create_model(model_config, self._device)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=train_config.lr)

    def _loss(self, logits: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        return self._criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))

    def _should_evaluate(self, step: int) -> bool:
        return self._train_config.eval_every is not None and step % self._train_config.eval_every == 0

    def evaluate(self, examples: Iterable[Example] | Sequence[Example]) -> float:
        loader = self._factory.create_data_loader(examples, self._data_loader_config, self._device)
        was_training = self._model.training
        self._model.eval()
        loss_sum = 0.0
        token_count = 0
        try:
            with torch.no_grad():
                for src, tgt, _ in loader:
                    src = src.to(self._device)
                    tgt = tgt.to(self._device)
                    logits = self._model(src, tgt)
                    loss = self._loss(logits, tgt)
                    valid_tokens = int((tgt[:, 1:] != self._model.tgt_pad_idx).sum())
                    loss_sum += loss.item() * valid_tokens
                    token_count += valid_tokens
        finally:
            if was_training:
                self._model.train()
        if token_count == 0:
            raise ValueError("Evaluation dataset contains no valid target tokens.")
        return loss_sum / token_count

    def train(
        self,
        examples: Iterable[Example] | Sequence[Example],
        validation_examples: Iterable[Example] | Sequence[Example] | None = None,
    ) -> TrainingSummary:
        def total_steps(loader: object) -> int | None:
            if not hasattr(loader, "__len__"):
                return None
            return len(loader) * self._train_config.epochs

        def create_training_observer(
            model: Seq2Seq, device: torch.device, total_steps: int | None
        ) -> TrainingObserver:
            tokenizer_model_name = getattr(self._factory.dataset_metadata, "tokenizer_model_name", None)
            if self._train_config.translate_every is None:
                return TrainingObserver(self._train_config, total_steps=total_steps)
            if tokenizer_model_name is None:
                raise ValueError("Preview translation requires dataset tokenizer_model_name.")
            tokenizer = create_tokenizer("hf", [], tokenizer_model_name)
            return TrainingObserver(
                self._train_config,
                total_steps=total_steps,
                translator=Translator(
                    model, tokenizer, device, getattr(self._factory.dataset_metadata, "tgt_bos_id", None)
                ),
            )

        # main flow
        if self._train_config.eval_every is not None and validation_examples is None:
            raise ValueError("eval_every requires validation_examples.")
        run_dir = self._train_config.training_runs_dir / self._train_config.run_name
        loader = self._factory.create_data_loader(examples, self._data_loader_config, self._device)
        observer = create_training_observer(self._model, self._device, total_steps(loader))
        self._model.train()

        for epoch in range(1, self._train_config.epochs + 1):
            for src, tgt, batch_ids in loader:
                src = src.to(self._device)
                tgt = tgt.to(self._device)

                self._optimizer.zero_grad()
                logits = self._model(src, tgt)
                loss = self._loss(logits, tgt)

                loss.backward()
                grad_norm = float(nn.utils.clip_grad_norm_(self._model.parameters(), 1.0))
                self._optimizer.step()

                # log batch metrics via observer (to keep logging details out of here)
                observer.on_batch_end(
                    epoch, loss.item(), grad_norm, batch_ids, tgt.size(0), tgt_token_count=tgt[:, 1:].numel()
                )

                # evaluate
                if self._should_evaluate(observer.global_step):
                    observer.on_evaluation(observer.global_step, epoch, self.evaluate(validation_examples))  # type: ignore[arg-type]

        checkpoint_file = save_checkpoint(
            run_dir, self._model, self._optimizer, self._model_config, self._factory.dataset_metadata
        )

        return TrainingSummary(observer.processed_examples, observer.loss_value, str(checkpoint_file), None)
