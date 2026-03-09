from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from statistics import median
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..data_prod import (
    ArrowTranslationDataset,
    collate_fn_prod,
    load_arrow_records,
)
from ..model import Seq2Seq
from .preflight import validate_records_contract


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


def train_prod(
    *,
    dataset_path: str | Path,
    id_field: str = "id",
    src_field: str = "src_ids",
    tgt_field: str = "tgt_ids",
    src_pad_idx: int,
    tgt_pad_idx: int,
    tgt_sos_idx: int | None = None,
    emb_dim: int = 256,
    hidden_dim: int = 1024,
    num_heads: int = 8,
    num_layers: int = 4,
    dropout: float = 0.1,
    lr: float = 3e-4,
    batch_size: int = 64,
    epochs: int = 1,
    seed: int = 42,
    attention: str = "torch",
    max_examples: int | None = None,
    shuffle: bool = True,
    log_every: int = 50,
    spike_window: int = 100,
    spike_factor: float = 3.0,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    if tgt_sos_idx is None:
        tgt_sos_idx = tgt_pad_idx

    _set_seed(seed)
    train_device = _resolve_device(device)

    raw_records = load_arrow_records(dataset_path)
    if max_examples is not None:
        raw_records = raw_records.select(range(min(max_examples, len(raw_records))))

    stats = validate_records_contract(
        raw_records,
        id_field=id_field,
        src_field=src_field,
        tgt_field=tgt_field,
    )

    src_vocab_size = int(stats["max_src_token_id"]) + 1
    tgt_vocab_size = int(stats["max_tgt_token_id"]) + 1

    dataset = ArrowTranslationDataset(
        dataset_path,
        id_field=id_field,
        src_field=src_field,
        tgt_field=tgt_field,
        max_examples=max_examples,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn_prod(batch, src_pad_idx, tgt_pad_idx),
    )

    model = Seq2Seq(
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
    ).to(train_device)

    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    loss_history: deque[float] = deque(maxlen=spike_window)
    global_step = 0
    last_loss = None
    last_spike: dict[str, Any] | None = None

    for epoch in range(1, epochs + 1):
        for src, tgt, batch_ids in loader:
            src = src.to(train_device)
            tgt = tgt.to(train_device)

            optim.zero_grad()
            logits = model(src, tgt)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt[:, 1:].reshape(-1),
            )
            loss.backward()
            grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            optim.step()

            global_step += 1
            current_loss = float(loss.item())
            median_loss = median(loss_history) if loss_history else current_loss
            is_spike = bool(loss_history) and (
                current_loss > (median_loss * spike_factor)
            )
            loss_history.append(current_loss)
            last_loss = current_loss

            if is_spike:
                last_spike = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": current_loss,
                    "median_window_loss": median_loss,
                    "batch_ids": list(batch_ids),
                }
                print(
                    "SPIKE "
                    f"step={global_step} epoch={epoch} loss={current_loss:.4f} "
                    f"median={median_loss:.4f} batch_ids={batch_ids}"
                )

            if global_step % log_every == 0:
                current_lr = float(optim.param_groups[0]["lr"])
                print(
                    f"step={global_step} epoch={epoch} loss={current_loss:.4f} "
                    f"grad_norm={grad_norm:.4f} lr={current_lr:.6g} "
                    f"batch_ids={batch_ids}"
                )

    return {
        "num_examples": stats["num_examples"],
        "final_loss": last_loss,
        "global_step": global_step,
        "device": str(train_device),
        "src_vocab_size": src_vocab_size,
        "tgt_vocab_size": tgt_vocab_size,
        "last_spike": last_spike,
    }
