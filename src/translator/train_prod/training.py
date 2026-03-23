from __future__ import annotations

import logging
import random
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import torch
import torch.nn as nn

from ..model import Seq2Seq
from ..types import Example
from .config import DataLoaderConfig, ModelConfig, TrainConfig
from .factory import Factory
from .inference import create_translation_preview_fn
from .logging import TrainingLogger


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


def _save_training_checkpoint(
    checkpoint_path: str | Path,
    model: Seq2Seq,
) -> Path:
    checkpoint_file = Path(checkpoint_path)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
        },
        checkpoint_file,
    )
    return checkpoint_file


@dataclass(frozen=True)
class TrainingSummary:
    num_examples: int
    final_loss: float | None
    checkpoint_path: str
class TrainingObserver:
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
    def __init__(self, factory: Factory) -> None:
        self.factory = factory

    def train(
        self,
        examples: Iterable[Example] | Sequence[Example],
        *,
        train_config: TrainConfig,
        model_config: ModelConfig = ModelConfig(),
        data_loader_config: DataLoaderConfig = DataLoaderConfig(),
    ) -> TrainingSummary:
        run_dir = Path(train_config.runs_dir) / train_config.run_name
        checkpoint_path = run_dir / "checkpoint.pt"

        _set_seed(train_config.seed)
        device = _resolve_device(train_config.device)
        model = self.factory.create_model(model_config, device)
        observer = TrainingObserver(
            train_config,
            run_dir / "training.log",
            translation_preview_fn=create_translation_preview_fn(
                train_config,
                model_config,
                self.factory.dataset_metadata,
                model,
                device,
            ),
        )
        loader = self.factory.create_data_loader(examples, data_loader_config,
                                                 device)
        criterion = nn.CrossEntropyLoss(ignore_index=model.tgt_pad_idx)
        optim = torch.optim.Adam(model.parameters(), lr=train_config.lr)
        model.train()

        for epoch in range(1, train_config.epochs + 1):
            for src, tgt, batch_ids in loader:
                src = src.to(device)
                tgt = tgt.to(device)

                optim.zero_grad()
                logits = model(src, tgt)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tgt[:, 1:].reshape(-1),
                )
                loss.backward()
                grad_norm = float(
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                )
                optim.step()

                # log batch metrics via observer (to keep logging details out of here)
                observer.on_batch_end(
                    epoch, loss.item(), grad_norm, batch_ids,
                    tgt.size(0), tgt_token_count = tgt[:, 1:].numel()
                )

        checkpoint_file = _save_training_checkpoint(checkpoint_path, model)

        return TrainingSummary(observer.processed_examples, observer.loss_value,
                               str(checkpoint_file))
    
