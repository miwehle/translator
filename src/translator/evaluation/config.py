from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    path: str
    config: str | None = None
    split: str = "test"


@dataclass(frozen=True)
class MappingConfig:
    src: str
    ref: str


@dataclass(frozen=True, kw_only=True)
class CometScoreConfig:
    checkpoint: str
    dataset: DatasetConfig
    mapping: MappingConfig
    model: str = "Unbabel/wmt22-comet-da"
    output_path: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, DatasetConfig):
            object.__setattr__(self, "dataset", DatasetConfig(**self.dataset))
        if not isinstance(self.mapping, MappingConfig):
            object.__setattr__(self, "mapping", MappingConfig(**self.mapping))

    @property
    def checkpoint_file(self) -> Path:
        return (
            Path("/content/drive/MyDrive/nmt_lab/artifacts/training_runs") / self.checkpoint / "checkpoint.pt"
        )
