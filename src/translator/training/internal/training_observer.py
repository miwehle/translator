from __future__ import annotations

import logging
from collections import deque
from collections.abc import Sequence
from statistics import median

from ...inference import Translator
from ..config import TrainConfig
from .logging import TrainingLogger


class TrainingObserver:
    def __init__(
        self, train_config: TrainConfig, total_steps: int | None = None, translator: Translator | None = None
    ) -> None:
        run_dir = train_config.training_runs_dir / train_config.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        self.train_config = train_config
        self.training_logger = TrainingLogger(run_dir / "training.log", total_steps=total_steps)
        self.translation_examples_path = run_dir / "translation_examples.txt"
        self.translate_examples = list(train_config.translate_examples)
        self.translator = translator
        self.loss_history: deque[float] = deque(maxlen=train_config.spike_window)
        self.global_step = 0
        self.processed_examples = 0
        self.loss_value: float | None = None

    def append_translation_examples(
        self, step: int, epoch: int, loss: float, translations: Sequence[tuple[str, str]]
    ) -> None:
        with self.translation_examples_path.open("a", encoding="utf-8") as handle:
            handle.write(f"step={step} ep={epoch} loss={loss:.4f}\n")
            handle.write("---\n")
            for index, (source_text, translated_text) in enumerate(translations):
                if index > 0:
                    handle.write("\n")
                handle.write(f"src: {source_text}\n")
                handle.write(f"pred: {translated_text}\n")
            handle.write("\n")

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
        median_loss = median(self.loss_history) if self.loss_history else loss_value
        is_spike = bool(self.loss_history) and (loss_value > (median_loss * self.train_config.spike_factor))
        self.loss_history.append(loss_value)
        self.training_logger.add_decoder_tokens(tgt_token_count, tgt_size)

        if is_spike:
            self.training_logger.log(
                self.global_step,
                epoch,
                loss_value,
                median_loss,
                label="SPIKE",
                level=logging.WARNING,
                batch_ids=batch_ids,
            )

        if self.global_step == 1 or self.global_step % self.train_config.log_every == 0:
            self.training_logger.log(
                self.global_step, epoch, loss_value, median_loss, grad_norm=grad_norm, lr=self.train_config.lr
            )
        if (
            self.train_config.translate_every is not None
            and self.translator is not None
            and (self.global_step == 1 or self.global_step % self.train_config.translate_every == 0)
        ):
            try:
                translations = self.translator.translate_many(self.translate_examples)
                self.append_translation_examples(
                    self.global_step,
                    epoch,
                    loss_value,
                    list(zip(self.translate_examples, translations, strict=True)),
                )
            except Exception as exc:
                self.training_logger.log_translation_failure(self.global_step, epoch, exc)
