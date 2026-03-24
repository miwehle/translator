from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def detect_hardware_type() -> str:
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


def configure_translator_logging(*, log_path: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger("translator")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    close_translator_logging()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path is not None:
        log_file = Path(log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def close_translator_logging() -> None:
    logger = logging.getLogger("translator")
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
