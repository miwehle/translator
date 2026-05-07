from __future__ import annotations

from dataclasses import field
from pathlib import Path

import torch
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

_CONFIG = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


@dataclass(frozen=True, config=_CONFIG)
class ModelConfig:
    d_model: int = 256
    ff_dim: int = 1024
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    max_seq_len: int = 1024
    attention: str = "torch"


@dataclass(frozen=True, kw_only=True, config=_CONFIG)
class TrainConfig:
    artifacts_dir: str = "/content/drive/MyDrive/nmt_lab/artifacts"
    dataset: str
    experiment: str | None = None
    validation_dataset: str | None = None
    validate_every: int | None = None
    enable_tensorboard: bool = False
    use_bf16: bool = False
    run_name: str = ""
    force: bool = False
    device: str | torch.device | None = None
    seed: int = 42
    lr: float = 3e-4
    epochs: int = 1
    log_every: int = 100
    translate_every: int | None = None
    translate_examples: tuple[str, ...] = ()
    spike_window: int = 100
    spike_factor: float = 3.0

    @property
    def datasets_dir(self) -> Path:
        return Path(self.artifacts_dir) / "datasets"

    @property
    def training_runs_dir(self) -> Path:
        return Path(self.artifacts_dir) / "training_runs"


@dataclass(frozen=True, config=_CONFIG)
class DataLoaderConfig:
    batch_size: int = 64
    shuffle: bool = True
    num_workers: int = 0
    prefetch_factor: int | None = None
    persistent_workers: bool | None = None
    pin_memory: bool | None = None


@dataclass(frozen=True, kw_only=True, config=_CONFIG)
class TrainRunConfig:
    train_config: TrainConfig
    data_loader_config: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    model_config: ModelConfig | None = None
    resume_run: str | None = None


@dataclass(frozen=True, kw_only=True, config=_CONFIG)
class PreflightCheckRunConfig:
    dataset_path: str
    require_unique_ids: bool = True
    min_seq_len: int = 2
