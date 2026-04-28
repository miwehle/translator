from __future__ import annotations

from pathlib import Path

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

_CONFIG = ConfigDict(extra="forbid")


@dataclass(frozen=True, config=_CONFIG)
class DatasetConfig:
    path: str
    name: str | None = None
    split: str = "test"
    data_file: str | None = None
    datasets_dir: str = "/content/drive/MyDrive/nmt_lab/artifacts/datasets"


@dataclass(frozen=True, config=_CONFIG)
class MappingConfig:
    src: str
    ref: str


@dataclass(frozen=True, kw_only=True, config=_CONFIG)
class CometScoreConfig:
    checkpoint: str
    dataset_config: DatasetConfig
    mapping_config: MappingConfig
    model: str = "Unbabel/wmt22-comet-da"
    output_path: str | None = None

    @property
    def checkpoint_file(self) -> Path:
        return (
            Path("/content/drive/MyDrive/nmt_lab/artifacts/training_runs") / self.checkpoint / "checkpoint.pt"
        )
