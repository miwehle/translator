from __future__ import annotations

import json
import random
from collections import deque
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

import torch
import torch.nn as nn

from ..model import Seq2Seq
from ..types import Example
from .factory import Factory
from .logging import TrainingLogger


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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def _save_training_checkpoint(
    *,
    checkpoint_path: str | Path,
    model: Seq2Seq,
    summary: dict[str, Any],
    train_config: dict[str, Any],
) -> Path:
    checkpoint_file = Path(checkpoint_path)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "summary": summary,
            "train_config": train_config,
        },
        checkpoint_file,
    )
    return checkpoint_file


def _write_summary_json(summary_path: str | Path, summary: dict[str, Any]) -> Path:
    summary_file = Path(summary_path)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary_file


class Trainer:
    def __init__(self, factory: Factory) -> None:
        self.factory = factory

    def train(
        self,
        examples: Iterable[Example],
        *,
        train_config: TrainConfig,
        model_config: ModelConfig = ModelConfig(),
        data_loader_config: DataLoaderConfig = DataLoaderConfig(),
    ) -> dict[str, Any]:
        _set_seed(train_config.seed)
        resolved_device = _resolve_device(train_config.device)
        run_dir = Path(train_config.runs_dir) / train_config.run_name
        checkpoint_path = run_dir / "checkpoint.pt"
        summary_path = run_dir / "summary.json"
        model = self.factory.create_model(
            model_config=model_config,
            device=resolved_device,
        )

        loader = self.factory.create_data_loader(
            examples,
            data_loader_config=data_loader_config,
            device=resolved_device,
        )
        criterion = nn.CrossEntropyLoss(ignore_index=model.tgt_pad_idx)
        optim = torch.optim.Adam(model.parameters(), lr=train_config.lr)
        model.train()

        loss_history: deque[float] = deque(maxlen=train_config.spike_window)
        global_step = 0
        processed_examples = 0
        loss_value = None
        training_logger = TrainingLogger()

        for epoch in range(1, train_config.epochs + 1):
            for src, tgt, batch_ids in loader:
                src = src.to(resolved_device)
                tgt = tgt.to(resolved_device)

                optim.zero_grad()
                logits = model(src, tgt)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tgt[:, 1:].reshape(-1),
                )
                loss.backward()
                grad_norm = float(
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                )
                optim.step()

                global_step += 1
                processed_examples += tgt.size(0)
                loss_value = float(loss.item())
                median_loss = median(loss_history) if loss_history else loss_value
                is_spike = bool(loss_history) and (
                    loss_value > (median_loss * train_config.spike_factor)
                )
                loss_history.append(loss_value)
                training_logger.add_decoder_tokens(
                    tgt[:, 1:].numel(),
                    tgt.size(0),
                )

                if is_spike:
                    training_logger.log(
                        label="SPIKE",
                        step=global_step,
                        epoch=epoch,
                        loss=loss_value,
                        median_loss=median_loss,
                        batch_ids=list(batch_ids),
                    )

                if global_step % train_config.log_every == 0:
                    training_logger.log(
                        step=global_step,
                        epoch=epoch,
                        loss=loss_value,
                        median_loss=median_loss,
                        grad_norm=grad_norm,
                        lr=float(optim.param_groups[0]["lr"]),
                        batch_ids=list(batch_ids),
                    )

        summary = {
            "num_examples": processed_examples,
            "final_loss": loss_value,
            "global_step": global_step,
        }
        checkpoint_file = _save_training_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            summary={},
            train_config={
                "model_config": asdict(model_config),
                "train_config": asdict(train_config),
                "data_loader_config": asdict(data_loader_config),
            },
        )
        summary["checkpoint_path"] = str(checkpoint_file)
        summary_file = _write_summary_json(summary_path, summary)
        summary["summary_path"] = str(summary_file)
        return summary
