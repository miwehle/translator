"""Training orchestration.

Keeps the core train loop focused while adding:
- easy configuration of model, data loading and training via simple dataclasses,
- checkpointing via save and resume, and
- logging via an observer-style helper.
"""

from __future__ import annotations

import logging
import random
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import median

import torch
import torch.nn as nn

from ..inference import create_translation_preview_fn
from ..model import Seq2Seq
from ..shared import Example
from .checkpointing import load as load_checkpoint
from .checkpointing import save as save_checkpoint
from .config import DataLoaderConfig, ModelConfig, TrainConfig
from .factory import Factory
from .logging import TrainingLogger

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
    
class _TrainingObserver:
    def __init__(
        self,
        train_config: TrainConfig,
        log_path: str | Path,
        translation_preview_fn: Callable[[], list[tuple[str, str]]] | None = None,
    ) -> None:
        self.train_config = train_config
        self.training_logger = TrainingLogger(log_path=log_path)
        self.translation_preview_fn = translation_preview_fn
        self.loss_history: deque[float] = deque(
            maxlen=train_config.spike_window
        )
        self.global_step = 0
        self.processed_examples = 0
        self.loss_value: float | None = None

    def on_batch_end(
        self,
        epoch: int,
        loss_value: float,
        grad_norm: float,
        batch_ids: Sequence[int],
        tgt_size: int,
        tgt_token_count: int,
    ) -> None:
        self.global_step += 1
        self.processed_examples += tgt_size
        self.loss_value = loss_value
        median_loss = (
            median(self.loss_history)
            if self.loss_history
            else loss_value
        )
        is_spike = bool(self.loss_history) and (
            loss_value > (median_loss * self.train_config.spike_factor)
        )
        self.loss_history.append(loss_value)
        self.training_logger.add_decoder_tokens(tgt_token_count, tgt_size)

        if is_spike:
            self.training_logger.log(
                self.global_step, epoch, loss_value, median_loss,
                label="SPIKE", level=logging.WARNING, batch_ids=batch_ids,
            )

        if self.global_step == 1 or self.global_step % self.train_config.log_every == 0:
            self.training_logger.log(
                self.global_step, epoch, loss_value, median_loss,
                grad_norm=grad_norm, lr=self.train_config.lr,
            )
        if (
            self.train_config.translate_every is not None
            and self.translation_preview_fn is not None
            and (
                self.global_step == 1
                or self.global_step % self.train_config.translate_every == 0
            )
        ):
            try:
                self.training_logger.log_translations(
                    self.global_step,
                    epoch,
                    self.translation_preview_fn(),
                )
            except Exception as exc:
                self.training_logger.log_translation_failure(
                    self.global_step,
                    epoch,
                    exc,
                )

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
        self._factory = factory
        self._train_config = train_config
        self._data_loader_config = data_loader_config

        if (model_config is None) == (resume_run is None):
            raise ValueError("Exactly one of model_config or resume_run must be provided.")

        def validate_dataset_max_seq_len(max_seq_len: int) -> None:
            configured_max_seq_len = getattr(
                self._factory.dataset_metadata, "configured_max_seq_len", None)
            if configured_max_seq_len is None or configured_max_seq_len <= max_seq_len:
                return
            message = (
                "Dataset configured_max_seq_len exceeds model max_seq_len: "
                f"configured_max_seq_len={configured_max_seq_len} max_seq_len={max_seq_len}"
            )
            if self._train_config.force:
                logger.warning("%s; continue because force=True.", message)
                return
            raise ValueError(f"{message}. Set train_config.force=True to continue anyway.")

        _set_seed(train_config.seed)
        self._device = _resolve_device(train_config.device)
        self._criterion = nn.CrossEntropyLoss(
            ignore_index=self._factory.dataset_metadata.tgt_pad_id)
        
        if resume_run is not None:
            loaded = load_checkpoint(
                train_config.training_runs_dir / resume_run / "checkpoint.pt",
                self._factory, self._device)
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
        return self._criterion(logits.reshape(-1, logits.size(-1)),
                              tgt[:, 1:].reshape(-1))

    def evaluate(self, examples: Iterable[Example] | Sequence[Example]
    ) -> float | None:
        loader = self._factory.create_data_loader(examples, self._data_loader_config,
                                                 self._device)
        self._model.eval()
        loss_sum = 0.0
        token_count = 0
        with torch.no_grad():
            for src, tgt, _ in loader:
                src = src.to(self._device)
                tgt = tgt.to(self._device)
                logits = self._model(src, tgt)
                loss = self._loss(logits, tgt)
                valid_tokens = int((tgt[:, 1:] != self._model.tgt_pad_idx).sum())
                loss_sum += loss.item() * valid_tokens
                token_count += valid_tokens
        return None if token_count == 0 else loss_sum / token_count

    def train(self, examples: Iterable[Example] | Sequence[Example]
    ) -> TrainingSummary:
        def createTrainingObserver(model: Seq2Seq, device: torch.device, log_path: Path
        ) -> _TrainingObserver:
            tokenizer_model_name = getattr(
                self._factory.dataset_metadata, "tokenizer_model_name", None
            )
            tgt_bos_id = getattr(self._factory.dataset_metadata, "tgt_bos_id", None)
            return _TrainingObserver(
                self._train_config,
                log_path,
                translation_preview_fn=create_translation_preview_fn(
                    self._train_config.translate_every,
                    self._train_config.translate_examples, tokenizer_model_name,
                    tgt_bos_id, model, device))
        
        # main flow
        run_dir = self._train_config.training_runs_dir / self._train_config.run_name

        observer = createTrainingObserver(self._model, self._device, run_dir / "training.log")
        loader = self._factory.create_data_loader(
            examples, self._data_loader_config, self._device)
        self._model.train()

        for epoch in range(1, self._train_config.epochs + 1):
            for src, tgt, batch_ids in loader:
                src = src.to(self._device)
                tgt = tgt.to(self._device)

                self._optimizer.zero_grad()
                logits = self._model(src, tgt)
                loss = self._loss(logits, tgt)
                
                loss.backward()
                grad_norm = float(
                    nn.utils.clip_grad_norm_(self._model.parameters(), 1.0))
                self._optimizer.step()

                # log batch metrics via observer (to keep logging details out of here)
                observer.on_batch_end(
                    epoch, loss.item(), grad_norm, batch_ids,
                    tgt.size(0), tgt_token_count = tgt[:, 1:].numel())

        checkpoint_file = save_checkpoint(
            run_dir, self._model, self._optimizer, self._model_config,
            self._factory.dataset_metadata)

        return TrainingSummary(observer.processed_examples, observer.loss_value,
                               str(checkpoint_file))
