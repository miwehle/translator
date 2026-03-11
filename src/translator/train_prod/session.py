from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..data_prod import infer_pad_idx
from .training import train_prod

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
