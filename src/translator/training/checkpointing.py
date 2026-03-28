from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from ..model import Seq2Seq
from .config import ModelConfig

_CHECKPOINT_FILE_NAME = "checkpoint.pt"
_CHECKPOINT_MANIFEST_FILE_NAME = "checkpoint_manifest.yaml"


@dataclass(frozen=True)
class LoadedCheckpoint:
    model: Seq2Seq
    optimizer: torch.optim.Optimizer
    model_config: ModelConfig


def checkpoint_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / _CHECKPOINT_FILE_NAME


def checkpoint_manifest_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / _CHECKPOINT_MANIFEST_FILE_NAME


def load(
    checkpoint_path: str | Path,
    factory: Any,
    device: torch.device,
) -> LoadedCheckpoint:
    checkpoint_file = Path(checkpoint_path)
    manifest_path = checkpoint_manifest_path(checkpoint_file.parent)
    manifest = _load_manifest(manifest_path)
    _validate_dataset_metadata(manifest["dataset"], factory.dataset_metadata)

    model_config = ModelConfig(**manifest["model_config"])
    model = factory.create_model(model_config, device)
    optimizer = _create_optimizer(model, manifest["optimizer"])

    payload = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])

    return LoadedCheckpoint(model, optimizer, model_config)


def save(
    run_dir: str | Path,
    model: Seq2Seq,
    optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    dataset_metadata: Any,
) -> Path:
    run_dir_path = Path(run_dir)
    run_dir_path.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_path(run_dir_path)
    manifest_path = checkpoint_manifest_path(run_dir_path)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_file,
    )
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "checkpoint_file": checkpoint_file.name,
                "model_config": asdict(model_config),
                "optimizer": {
                    "type": _optimizer_type(optimizer),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                },
                "dataset": {
                    "src_vocab_size": int(dataset_metadata.src_vocab_size),
                    "tgt_vocab_size": int(dataset_metadata.tgt_vocab_size),
                    "src_pad_id": int(dataset_metadata.src_pad_id),
                    "tgt_pad_id": int(dataset_metadata.tgt_pad_id),
                    "tgt_bos_id": int(dataset_metadata.tgt_bos_id),
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return checkpoint_file


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Checkpoint manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle) or {}
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid checkpoint manifest: {manifest_path}")
    return manifest


def _create_optimizer(
    model: Seq2Seq,
    optimizer_manifest: dict[str, Any],
) -> torch.optim.Optimizer:
    optimizer_type = str(optimizer_manifest["type"]).lower()
    if optimizer_type != "adam":
        raise ValueError(f"Unsupported optimizer type in checkpoint: {optimizer_type!r}")
    return torch.optim.Adam(model.parameters(), lr=float(optimizer_manifest["lr"]))


def _optimizer_type(optimizer: torch.optim.Optimizer) -> str:
    if isinstance(optimizer, torch.optim.Adam):
        return "adam"
    raise ValueError(f"Unsupported optimizer type for checkpoint save: {type(optimizer).__name__}")


def _validate_dataset_metadata(
    manifest_dataset: dict[str, Any],
    dataset_metadata: Any,
) -> None:
    current_dataset = {
        "src_vocab_size": int(dataset_metadata.src_vocab_size),
        "tgt_vocab_size": int(dataset_metadata.tgt_vocab_size),
        "src_pad_id": int(dataset_metadata.src_pad_id),
        "tgt_pad_id": int(dataset_metadata.tgt_pad_id),
        "tgt_bos_id": int(dataset_metadata.tgt_bos_id),
    }
    for field_name, current_value in current_dataset.items():
        manifest_value = int(manifest_dataset[field_name])
        if manifest_value != current_value:
            raise ValueError(
                "Checkpoint dataset metadata mismatch for "
                f"{field_name}: checkpoint={manifest_value} current={current_value}"
            )
