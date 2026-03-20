from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class ModelConfig:
    emb_dim: int = 256
    hidden_dim: int = 1024
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    attention: str = "torch"


@dataclass(frozen=True, kw_only=True)
class TrainConfig:
    runs_dir: str | Path
    run_name: str = "run1"
    device: str | torch.device | None = None
    seed: int = 42
    lr: float = 3e-4
    epochs: int = 1
    log_every: int = 50
    spike_window: int = 100
    spike_factor: float = 3.0


@dataclass(frozen=True)
class DataLoaderConfig:
    batch_size: int = 64
    shuffle: bool = True
    num_workers: int = 0
    prefetch_factor: int | None = None
    persistent_workers: bool | None = None
    pin_memory: bool | None = None
