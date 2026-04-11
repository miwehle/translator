from __future__ import annotations

import logging
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from lab_infrastructure.compute_metrics import (
    detect_compute_hardware,
    estimate_compute_units,
    estimate_cost,
    get_gpu_util,
)
from lab_infrastructure.logging import get_logger


@dataclass
class TrainingLogger:
    log_path: str | Path | None = None
    total_steps: int | None = None
    euro_per_cu: float = 0.10
    start_time: datetime = field(default_factory=datetime.now)
    hardware_type: str = field(default_factory=detect_compute_hardware)
    last_log_time: float = field(default_factory=time.time)
    decoder_token_count: int = 0
    total_decoder_token_count: int = 0
    decoder_sequence_count: int = 0
    logger: logging.Logger = field(init=False)

    def __post_init__(self) -> None:
        get_logger("translator", log_path=self.log_path, stream=True)
        self.logger = logging.getLogger("translator.training.trainer")

    def add_decoder_tokens(self, count: int, num_sequences: int) -> None:
        positive_count = max(0, count)
        self.decoder_token_count += positive_count
        self.total_decoder_token_count += positive_count
        self.decoder_sequence_count += max(0, num_sequences)

    def _decoder_tokens_per_second(self) -> float | None:
        elapsed_seconds = time.time() - self.last_log_time
        if elapsed_seconds <= 0:
            return None
        return self.decoder_token_count / elapsed_seconds

    def _average_target_length(self) -> float | None:
        if self.decoder_sequence_count <= 0:
            return None
        return self.decoder_token_count / self.decoder_sequence_count

    def _progress_percent(self, step: int) -> int | None:
        if self.total_steps is None or self.total_steps <= 0:
            return None
        return min(100, round(step / self.total_steps * 100))

    @staticmethod
    def _format_float(value: float | None, decimals: int = 2) -> str:
        if value is None:
            return "-"
        return f"{value:.{decimals}f}"

    @staticmethod
    def _format_metric(value: object, unknown: str = "-") -> str:
        if value is None:
            return unknown
        return str(value)

    @staticmethod
    def _format_scaled(value: float | int | None, *, scale: float, decimals: int = 1) -> str:
        if value is None:
            return "-"
        return f"{value / scale:.{decimals}f}"

    def _build_message(
        self,
        *,
        label: str | None,
        step: int,
        epoch: int,
        loss: float,
        median_loss: float | None,
        grad_norm: float | None,
        lr: float | None,
        batch_ids: Sequence[int] | None,
    ) -> str:
        dec_tok_s = self._decoder_tokens_per_second()
        avg_tgt_len = self._average_target_length()
        gpu_util = get_gpu_util()
        used_cu = estimate_compute_units(self.hardware_type, self.start_time)
        used_eur = estimate_cost(used_cu, self.euro_per_cu)
        progress = self._progress_percent(step)
        gpu_text = f"{gpu_util}%" if gpu_util is not None else "-"
        progress_text = f"{progress}%" if progress is not None else "-"
        total_mtok = self._format_scaled(self.total_decoder_token_count, scale=1_000_000)
        ktok_s = self._format_scaled(dec_tok_s, scale=1_000)
        prefix = f"{label} " if label else ""
        batch_ids_text = f" batch_ids={batch_ids}" if batch_ids is not None else ""

        return (
            f"{prefix}prog={progress_text} step={step} ep={epoch} loss={loss:.3f} "
            f"med={self._format_float(median_loss, decimals=3)} "
            f"grad={self._format_float(grad_norm, decimals=3)} "
            f"lr={self._format_metric(lr) if lr is not None else '-'} "
            f"mtok={total_mtok} "
            f"ktok_s={ktok_s} "
            f"len={self._format_float(avg_tgt_len, decimals=1)} "
            f"gpu={gpu_text} cu={self._format_float(used_cu, decimals=2)} "
            f"eur={self._format_float(used_eur, decimals=2)}{batch_ids_text}"
        )

    def log_translation_failure(self, step: int, epoch: int, exc: Exception) -> None:
        cause = exc.__cause__
        traceback_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()
        self.logger.log(
            logging.WARNING,
            (
                f"TRANSLATE_FAILED step={step} ep={epoch} "
                f"preview translation failed; training continues. "
                f"exception_type={type(exc).__name__} "
                f"exception={exc!r} "
                f"cause_type={type(cause).__name__ if cause is not None else '-'} "
                f"cause={cause!r}\n"
                f"{traceback_text}"
            ),
        )

    def log(
        self,
        step: int,
        epoch: int,
        loss: float,
        median_loss: float | None,
        *,
        label: str | None = None,
        level: int = logging.INFO,
        grad_norm: float | None = None,
        lr: float | None = None,
        batch_ids: Sequence[int] | None = None,
    ) -> str:
        message = self._build_message(
            label=label,
            step=step,
            epoch=epoch,
            loss=loss,
            median_loss=median_loss,
            grad_norm=grad_norm,
            lr=lr,
            batch_ids=batch_ids,
        )
        self.logger.log(level, message)
        self.last_log_time = time.time()
        self.decoder_token_count = 0
        self.decoder_sequence_count = 0
        return message

