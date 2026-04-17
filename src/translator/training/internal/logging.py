from __future__ import annotations

import logging
import sys
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter, time

from lab_infrastructure.compute_metrics import (
    detect_compute_hardware,
    estimate_compute_units,
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
    last_log_time: float = field(default_factory=time)
    decoder_token_count: int = 0
    total_decoder_token_count: int = 0
    decoder_sequence_count: int = 0
    logger: logging.Logger = field(init=False)
    metrics_path: Path = field(init=False)
    metrics_handle: object = field(init=False)
    metric_row_count: int = 0
    metrics_file_time_s: float = 0.0
    metrics_stdout_time_s: float = 0.0
    diagnostic_log_time_s: float = 0.0

    _COLS = (
        ("TIME", 19),
        ("TYPE", 5),
        ("PROG", 4),
        ("STEP", 6),
        ("EP", 3),
        ("LOSS", 7),
        ("MLOSS", 7),
        ("VLOSS", 7),
        ("GRAD", 7),
        ("LR", 8),
        ("MTOK", 5),
        ("KTOK_S", 6),
        ("LEN", 5),
        ("GPU", 4),
        ("CU", 5),
    )

    def __post_init__(self) -> None:
        get_logger("translator", log_path=self.log_path, stream=False)
        self.logger = logging.getLogger("translator.training.trainer")
        self.metrics_path = (
            Path(self.log_path).with_name("training_metrics.log")
            if self.log_path is not None
            else Path("training_metrics.log")
        )
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_handle = self.metrics_path.open("a", encoding="utf-8")

    @classmethod
    def _format_header(cls) -> str:
        return " ".join(name.ljust(width) for name, width in cls._COLS)

    @classmethod
    def _format_cell(cls, value: str, width: int) -> str:
        return value.rjust(width) if value else " " * width

    def add_decoder_tokens(self, count: int, num_sequences: int) -> None:
        positive_count = max(0, count)
        self.decoder_token_count += positive_count
        self.total_decoder_token_count += positive_count
        self.decoder_sequence_count += max(0, num_sequences)

    def _decoder_tokens_per_second(self) -> float | None:
        elapsed_seconds = time() - self.last_log_time
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
            return ""
        return f"{value:.{decimals}f}"

    @staticmethod
    def _format_metric(value: object, unknown: str = "") -> str:
        if value is None:
            return unknown
        return str(value)

    @staticmethod
    def _format_scaled(value: float | int | None, *, scale: float, decimals: int = 1) -> str:
        if value is None:
            return ""
        return f"{value / scale:.{decimals}f}"

    def _write_metrics_line(self, line: str) -> None:
        t0 = perf_counter()
        self.metrics_handle.write(line + "\n")
        self.metrics_handle.flush()
        self.metrics_file_time_s += perf_counter() - t0
        t0 = perf_counter()
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
        self.metrics_stdout_time_s += perf_counter() - t0

    def _write_metrics_row(self, values: dict[str, str]) -> None:
        if self.metric_row_count == 0 or self.metric_row_count % 10 == 0:
            self._write_metrics_line(self._format_header())

        cells = []
        for name, width in self._COLS:
            value = values.get(name, "")
            if name in {"TIME", "TYPE"}:
                cells.append(value.ljust(width))
            else:
                cells.append(self._format_cell(value, width))
        self._write_metrics_line(" ".join(cells))
        self.metric_row_count += 1

    def _log_diagnostics(self, level: int, message: str, *args: object) -> None:
        t0 = perf_counter()
        self.logger.log(level, message, *args)
        self.diagnostic_log_time_s += perf_counter() - t0

    def _build_metrics_row(
        self,
        *,
        event_type: str,
        step: int,
        epoch: int,
        loss: float | None = None,
        median_loss: float | None,
        validation_loss: float | None = None,
        grad_norm: float | None,
        lr: float | None,
    ) -> dict[str, str]:
        dec_tok_s = self._decoder_tokens_per_second()
        avg_tgt_len = self._average_target_length()
        gpu_util = get_gpu_util()
        used_cu = estimate_compute_units(self.hardware_type, self.start_time)
        progress = self._progress_percent(step)
        gpu_text = f"{gpu_util}%" if gpu_util is not None else ""
        progress_text = f"{progress}%" if progress is not None else ""
        total_mtok = self._format_scaled(self.total_decoder_token_count, scale=1_000_000)
        ktok_s = self._format_scaled(dec_tok_s, scale=1_000)
        return {
            "TIME": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "TYPE": event_type,
            "PROG": progress_text,
            "STEP": str(step),
            "EP": str(epoch),
            "LOSS": self._format_float(loss, decimals=4),
            "MLOSS": self._format_float(median_loss, decimals=4),
            "VLOSS": self._format_float(validation_loss, decimals=4),
            "GRAD": self._format_float(grad_norm, decimals=4),
            "LR": self._format_float(lr, decimals=5),
            "MTOK": total_mtok,
            "KTOK_S": ktok_s,
            "LEN": self._format_float(avg_tgt_len, decimals=1),
            "GPU": gpu_text,
            "CU": self._format_float(used_cu, decimals=2),
        }

    def log_translation_failure(self, step: int, epoch: int, exc: Exception) -> None:
        cause = exc.__cause__
        traceback_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()
        self._log_diagnostics(
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
        validation_loss: float | None = None,
        label: str | None = None,
        level: int = logging.INFO,
        grad_norm: float | None = None,
        lr: float | None = None,
        batch_ids: Sequence[int] | None = None,
    ) -> str:
        event_type = "SPIKE" if label == "SPIKE" else "TRAIN"
        row = self._build_metrics_row(
            event_type=event_type,
            step=step,
            epoch=epoch,
            loss=loss,
            median_loss=median_loss,
            validation_loss=validation_loss,
            grad_norm=grad_norm,
            lr=lr,
        )
        self._write_metrics_row(row)
        message = " ".join(f"{key}={value}" for key, value in row.items() if value)
        if event_type == "SPIKE":
            self._log_diagnostics(level, "%s batch_ids=%s", message, batch_ids)
        self.last_log_time = time()
        self.decoder_token_count = 0
        self.decoder_sequence_count = 0
        return message

    def close(self) -> None:
        t0 = perf_counter()
        if getattr(self, "metrics_handle", None) is not None and not self.metrics_handle.closed:
            self.metrics_handle.close()
        close_time_s = perf_counter() - t0
        self.logger.info(
            "TRAINING_METRICS_IO rows=%s total_s=%.3f file_s=%.3f stdout_s=%.3f diagnostic_s=%.3f close_s=%.3f",
            self.metric_row_count,
            self.metrics_file_time_s + self.metrics_stdout_time_s + self.diagnostic_log_time_s + close_time_s,
            self.metrics_file_time_s,
            self.metrics_stdout_time_s,
            self.diagnostic_log_time_s,
            close_time_s,
        )

