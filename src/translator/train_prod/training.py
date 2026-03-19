from __future__ import annotations

import json
import random
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from statistics import median
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from ..data_prod import collate_fn_prod
from ..model import Seq2Seq
from ..types import Example
from .logging import TrainingLogger


@dataclass(frozen=True)
class TrainerConfig:
    id_field: str = "id"
    src_field: str = "src_ids"
    tgt_field: str = "tgt_ids"
    batch_size: int = 64
    shuffle: bool = True
    device: str | torch.device | None = None
    seed: int = 42


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


def build_model(
    *,
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_pad_idx: int,
    tgt_pad_idx: int,
    tgt_sos_idx: int,
    emb_dim: int = 256,
    hidden_dim: int = 1024,
    num_heads: int = 8,
    num_layers: int = 4,
    dropout: float = 0.1,
    attention: str = "torch",
    device: str | torch.device | None = None,
    seed: int | None = None,
) -> Seq2Seq:
    if seed is not None:
        _set_seed(seed)
    resolved_device = _resolve_device(device)
    return Seq2Seq(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=emb_dim,
        ff_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        src_pad_idx=src_pad_idx,
        tgt_pad_idx=tgt_pad_idx,
        tgt_sos_idx=tgt_sos_idx,
        dropout=dropout,
        attention=attention,
    ).to(resolved_device)


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


def _collate_examples(
    batch: list[Example],
    *,
    id_field: str,
    src_field: str,
    tgt_field: str,
    pad_idx_src: int,
    pad_idx_tgt: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    normalized = [
        (
            int(item[id_field]),
            [int(x) for x in item[src_field]],
            [int(x) for x in item[tgt_field]],
        )
        for item in batch
    ]
    return collate_fn_prod(
        normalized,
        pad_idx_src=pad_idx_src,
        pad_idx_tgt=pad_idx_tgt,
    )


class _ExampleIterableDataset(IterableDataset):
    def __init__(self, examples: Iterable[Example]) -> None:
        self.examples = examples

    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            yield from self.examples
            return

        for index, example in enumerate(self.examples):
            if index % worker.num_workers == worker.id:
                yield example


def _create_data_loader(
    examples: Iterable[Example],
    *,
    id_field: str,
    src_field: str,
    tgt_field: str,
    pad_idx_src: int,
    pad_idx_tgt: int,
    shuffle: bool,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int | None,
    persistent_workers: bool | None,
    pin_memory: bool | None,
    device: torch.device,
) -> DataLoader:
    collate = partial(
        _collate_examples,
        id_field=id_field,
        src_field=src_field,
        tgt_field=tgt_field,
        pad_idx_src=pad_idx_src,
        pad_idx_tgt=pad_idx_tgt,
    )
    if hasattr(examples, "__len__") and hasattr(examples, "__getitem__"):
        dataset = examples
        loader_shuffle = shuffle
    else:
        dataset = _ExampleIterableDataset(examples)
        loader_shuffle = False

    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": loader_shuffle,
        "collate_fn": collate,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda" if pin_memory is None else pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = (
            True if persistent_workers is None else persistent_workers
        )
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(dataset, **loader_kwargs)


class Trainer:
    def __init__(self, model: Seq2Seq, config: TrainerConfig) -> None:
        self.config = config
        _set_seed(config.seed)
        self.device = _resolve_device(config.device)
        self.model = model.to(self.device)

    def train(
        self,
        examples: Iterable[Example],
        *,
        lr: float = 3e-4,
        epochs: int = 1,
        log_every: int = 50,
        spike_window: int = 100,
        spike_factor: float = 3.0,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        persistent_workers: bool | None = None,
        pin_memory: bool | None = None,
        checkpoint_path: str | Path | None = None,
        summary_path: str | Path | None = None,
    ) -> dict[str, Any]:
        loader = _create_data_loader(
            examples,
            id_field=self.config.id_field,
            src_field=self.config.src_field,
            tgt_field=self.config.tgt_field,
            pad_idx_src=self.model.src_pad_idx,
            pad_idx_tgt=self.model.tgt_pad_idx,
            shuffle=self.config.shuffle,
            batch_size=self.config.batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            device=self.device,
        )
        criterion = nn.CrossEntropyLoss(ignore_index=self.model.tgt_pad_idx)
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()

        loss_history: deque[float] = deque(maxlen=spike_window)
        global_step = 0
        processed_examples = 0
        loss_value = None
        training_logger = TrainingLogger()

        for epoch in range(1, epochs + 1):
            for src, tgt, batch_ids in loader:
                src = src.to(self.device)
                tgt = tgt.to(self.device)

                optim.zero_grad()
                logits = self.model(src, tgt)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tgt[:, 1:].reshape(-1),
                )
                loss.backward()
                grad_norm = float(
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                )
                optim.step()

                global_step += 1
                processed_examples += tgt.size(0)
                loss_value = float(loss.item())
                median_loss = median(loss_history) if loss_history else loss_value
                is_spike = bool(loss_history) and (
                    loss_value > (median_loss * spike_factor)
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

                if global_step % log_every == 0:
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
            "global_step": global_step
        }
        if checkpoint_path is not None:
            checkpoint_file = _save_training_checkpoint(
                checkpoint_path=checkpoint_path,
                model=self.model,
                summary={},
                train_config={},
            )
            summary["checkpoint_path"] = str(checkpoint_file)
        if summary_path is not None:
            summary_file = _write_summary_json(summary_path, summary)
            summary["summary_path"] = str(summary_file)
        return summary
