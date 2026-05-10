from __future__ import annotations

from pathlib import Path
from typing import Any

_FLUSH_SECS = 10
_LOG_DIR_NAME = "tensorboard"


def _create_summary_writer(log_dir: Path) -> Any:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorBoard logging requires the 'tensorboard' package. Install it or disable "
            "TrainRunConfig.enable_tensorboard."
        ) from exc
    return SummaryWriter(str(log_dir), flush_secs=_FLUSH_SECS)


class TensorBoardLogger:
    def __init__(self, run_dir: Path) -> None:
        self.log_dir = run_dir / _LOG_DIR_NAME
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = _create_summary_writer(self.log_dir)

    def log_scalars(self, step: int, *, loss: float, validation_loss: float | None = None) -> None:
        self._writer.add_scalar("loss/train", loss, step)
        if validation_loss is not None:
            self._writer.add_scalar("loss/validation", validation_loss, step)
            self._writer.flush()

    def close(self) -> None:
        self._writer.close()
