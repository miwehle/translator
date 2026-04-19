from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    path: str
    name: str | None = None
    split: str = "test"
    data_file: str | None = None
    datasets_dir: str = "/content/drive/MyDrive/nmt_lab/artifacts/datasets"


@dataclass(frozen=True)
class MappingConfig:
    src: str
    ref: str


@dataclass(frozen=True, kw_only=True)
class CometScoreConfig:
    checkpoint: str
    dataset_config: DatasetConfig
    mapping_config: MappingConfig
    model: str = "Unbabel/wmt22-comet-da"
    output_path: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_config, DatasetConfig):
            object.__setattr__(self, "dataset_config", DatasetConfig(**self.dataset_config))
        if not isinstance(self.mapping_config, MappingConfig):
            object.__setattr__(self, "mapping_config", MappingConfig(**self.mapping_config))

    @property
    def checkpoint_file(self) -> Path:
        return Path("/content/drive/MyDrive/nmt_lab/artifacts/training_runs") / self.checkpoint / "checkpoint.pt"
