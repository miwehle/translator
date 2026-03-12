from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..data_prod import infer_pad_idx
from .training import Trainer, _write_summary_json, train_prod

TrainProdConfig = Mapping[str, Any]


def build_train_prod_config(
    *,
    dataset_path: str | Path,
    id_field: str = "id",
    src_field: str = "src_ids",
    tgt_field: str = "tgt_ids",
    src_pad_idx: int | None = None,
    tgt_pad_idx: int | None = None,
    **train_kwargs: Any,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    resolved: dict[str, Any] = {
        "dataset_path": dataset_path,
        "id_field": id_field,
        "src_field": src_field,
        "tgt_field": tgt_field,
        **train_kwargs,
    }
    resolved["src_pad_idx"] = (
        src_pad_idx
        if src_pad_idx is not None
        else infer_pad_idx(dataset_path, src_field)
    )
    resolved["tgt_pad_idx"] = (
        tgt_pad_idx
        if tgt_pad_idx is not None
        else infer_pad_idx(dataset_path, tgt_field)
    )
    return resolved


def run_train_prod(config: TrainProdConfig) -> dict[str, Any]:
    return train_prod(**dict(config))


def run_train_prod_2(config: TrainProdConfig) -> dict[str, Any]:
    resolved = dict(config)
    trainer = Trainer(
        dataset_path=resolved["dataset_path"],
        id_field=resolved.get("id_field", "id"),
        src_field=resolved.get("src_field", "src_ids"),
        tgt_field=resolved.get("tgt_field", "tgt_ids"),
        src_pad_idx=resolved["src_pad_idx"],
        tgt_pad_idx=resolved["tgt_pad_idx"],
        tgt_sos_idx=resolved.get("tgt_sos_idx"),
        emb_dim=resolved.get("emb_dim", 256),
        hidden_dim=resolved.get("hidden_dim", 1024),
        num_heads=resolved.get("num_heads", 8),
        num_layers=resolved.get("num_layers", 4),
        dropout=resolved.get("dropout", 0.1),
        attention=resolved.get("attention", "torch"),
        batch_size=resolved.get("batch_size", 64),
        shuffle=resolved.get("shuffle", True),
        max_examples=resolved.get("max_examples"),
        device=resolved.get("device"),
        seed=resolved.get("seed", 42),
    )
    summary = trainer.train(
        lr=resolved.get("lr", 3e-4),
        epochs=resolved.get("epochs", 1),
        log_every=resolved.get("log_every", 50),
        spike_window=resolved.get("spike_window", 100),
        spike_factor=resolved.get("spike_factor", 3.0),
        checkpoint_path=resolved.get("checkpoint_path"),
    )
    summary_path = resolved.get("summary_path")
    if summary_path is not None:
        summary_file = _write_summary_json(summary_path, summary)
        summary["summary_path"] = str(summary_file)
    return summary
