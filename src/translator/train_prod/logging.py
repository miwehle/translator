from __future__ import annotations

import logging
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..logging_utils import configure_translator_logging, detect_hardware_type

_CU_RATES = {
    "T4": 1.8,
    "V100": 5.0,
    "A100": 12.5,
    "H100": 22.5,
    "RTXPRO6000": 22.5,
    "CPU": 0.5,
}


def _get_gpu_util() -> int | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    try:
        return int(out.strip())
    except ValueError:
        return None
@dataclass
class TrainingLogger:
    log_path: str | Path | None = None
    euro_per_cu: float = 0.10
    start_time: datetime = field(default_factory=datetime.now)
    hardware_type: str = field(default_factory=detect_hardware_type)
    last_log_time: float = field(default_factory=time.time)
    decoder_token_count: int = 0
    decoder_sequence_count: int = 0
    logger_name: str = "translator.train_prod.training"
    logger: logging.Logger = field(init=False)

    def __post_init__(self) -> None:
        configure_translator_logging(log_path=self.log_path)
        self.logger = logging.getLogger(self.logger_name)

    def add_decoder_tokens(self, count: int, num_sequences: int) -> None:
        self.decoder_token_count += max(0, count)
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

    def _estimate_compute_units_used(self) -> float | None:
        if self.hardware_type not in _CU_RATES:
            return None

        elapsed_seconds = (datetime.now() - self.start_time).total_seconds()
        if elapsed_seconds < 0:
            return None

        elapsed_hours = elapsed_seconds / 3600.0
        return elapsed_hours * _CU_RATES[self.hardware_type]

    def _estimate_euro_cost(self, used_cu: float | None) -> float | None:
        if used_cu is None or self.euro_per_cu < 0:
            return None
        return used_cu * self.euro_per_cu

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
        gpu_util = _get_gpu_util()
        used_cu = self._estimate_compute_units_used()
        used_eur = self._estimate_euro_cost(used_cu)
        gpu_text = f"{gpu_util}%" if gpu_util is not None else "-"
        prefix = f"{label} " if label else ""
        batch_ids_text = f" batch_ids={batch_ids}" if batch_ids is not None else ""

        return (
            f"{prefix}step={step} ep={epoch} loss={loss:.4f} "
            f"med={self._format_float(median_loss, decimals=4)} "
            f"grad={self._format_float(grad_norm, decimals=4)} "
            f"lr={self._format_metric(lr) if lr is not None else '-'} "
            f"tok/s={self._format_float(dec_tok_s, decimals=0)} "
            f"len={self._format_float(avg_tgt_len, decimals=1)} "
            f"gpu={gpu_text} cu={self._format_float(used_cu, decimals=2)} "
            f"~eur={self._format_float(used_eur, decimals=2)} "
            f"{batch_ids_text}"
        )

    def _emit(self, *, message: str, level: int) -> None:
        self.logger.log(level, message)

    def log_translations(
        self,
        step: int,
        epoch: int,
        translations: Sequence[tuple[str, str]],
    ) -> None:
        lines = [f"TRANSLATE step={step} ep={epoch}"]
        for source_text, translated_text in translations:
            lines.append(f"src={source_text}")
            lines.append(f"pred={translated_text}")
        self._emit(message="\n".join(lines), level=logging.INFO)

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
        self._emit(message=message, level=level)
        self.last_log_time = time.time()
        self.decoder_token_count = 0
        self.decoder_sequence_count = 0
        return message
