from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

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


def _detect_hardware_type() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "CPU"

    gpu_name = result.stdout.strip()
    if not gpu_name:
        return "(unknown)"

    if "H100" in gpu_name:
        return "H100"
    if "A100" in gpu_name:
        return "A100"
    if "V100" in gpu_name:
        return "V100"
    if "T4" in gpu_name:
        return "T4"
    if "RTX PRO 6000" in gpu_name:
        return "RTXPRO6000"
    return gpu_name


def _estimate_compute_units_used(
    start_time: datetime,
    hardware_type: str,
) -> float | None:
    if hardware_type not in _CU_RATES:
        return None

    elapsed_seconds = (datetime.now() - start_time).total_seconds()
    if elapsed_seconds < 0:
        return None

    elapsed_hours = elapsed_seconds / 3600.0
    return elapsed_hours * _CU_RATES[hardware_type]


def _estimate_euro_cost(
    used_cu: float | None,
    euro_per_cu: float,
) -> float | None:
    if used_cu is None or euro_per_cu < 0:
        return None
    return used_cu * euro_per_cu


def _format_float(value: float | None, decimals: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{decimals}f}"


def _format_metric(value: object, unknown: str = "-") -> str:
    if value is None:
        return unknown
    return str(value)


@dataclass
class TrainingLogger:
    print_enabled: bool = True
    log_path: str | Path | None = None
    euro_per_cu: float = 0.10
    start_time: datetime = field(default_factory=datetime.now)
    hardware_type: str = field(default_factory=_detect_hardware_type)
    last_log_time: float = field(default_factory=time.time)
    decoder_token_count: int = 0
    decoder_sequence_count: int = 0

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
        batch_ids: list[int],
    ) -> str:
        dec_tok_s = self._decoder_tokens_per_second()
        avg_tgt_len = self._average_target_length()
        gpu_util = _get_gpu_util()
        used_cu = _estimate_compute_units_used(self.start_time, self.hardware_type)
        used_eur = _estimate_euro_cost(used_cu, self.euro_per_cu)
        gpu_text = f"{gpu_util}%" if gpu_util is not None else "-"
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prefix = f"{label} " if label else ""
        batch_ids_text = f" batch_ids={batch_ids}" if label == "SPIKE" else ""

        return (
            f"{now} {prefix}step={step} ep={epoch} loss={loss:.4f} "
            f"med={_format_float(median_loss, decimals=4)} "
            f"grad={_format_float(grad_norm, decimals=4)} "
            f"lr={_format_metric(lr) if lr is not None else '-'} "
            f"tok={self.decoder_token_count} "
            f"tok/s={_format_float(dec_tok_s, decimals=0)} "
            f"len={_format_float(avg_tgt_len, decimals=1)} "
            f"gpu={gpu_text} cu={_format_float(used_cu, decimals=2)} "
            f"eur={_format_float(used_eur, decimals=2)} "
            f"hw={_format_metric(self.hardware_type, unknown='(unknown)')} "
            f"{batch_ids_text}"
        )

    def _emit(self, message: str) -> None:
        if self.print_enabled:
            print(message)
        if self.log_path is not None:
            log_file = Path(self.log_path)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with log_file.open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")

    def log(
        self,
        *,
        label: str | None = None,
        step: int,
        epoch: int,
        loss: float,
        median_loss: float | None,
        grad_norm: float | None = None,
        lr: float | None = None,
        batch_ids: list[int],
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
        self._emit(message)
        self.last_log_time = time.time()
        self.decoder_token_count = 0
        self.decoder_sequence_count = 0
        return message
