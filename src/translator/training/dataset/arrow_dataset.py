from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch


def load_arrow_records(dataset_path: str | Path):
    from datasets import load_from_disk
    from datasets.dataset_dict import DatasetDict

    loaded = load_from_disk(str(dataset_path))
    if isinstance(loaded, DatasetDict):
        split_names = list(loaded.keys())
        if not split_names:
            raise ValueError("Loaded DatasetDict has no splits.")
        return loaded[split_names[0]]
    return loaded


def infer_pad_idx(
    dataset_path: str | Path,
    field: str,
) -> int:
    records = load_arrow_records(dataset_path)
    max_token = -1
    for row in records:
        item = cast(Mapping[str, Any], row)
        values_obj = item.get(field)
        if not isinstance(values_obj, list):
            raise ValueError(
                f"Expected list[int] in field '{field}', got "
                f"{type(values_obj).__name__}."
            )
        values = [int(x) for x in values_obj]
        if not values:
            raise ValueError(f"Field '{field}' contains an empty sequence.")
        max_token = max(max_token, max(values))

    if max_token < 0:
        raise ValueError(f"Could not infer pad idx from field '{field}'.")
    return max_token + 1


def collate_fn_prod(
    batch: list[tuple[int, list[int], list[int]]],
    pad_idx_src: int,
    pad_idx_tgt: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    max_src = max(len(src) for _, src, _ in batch)
    max_tgt = max(len(tgt) for _, _, tgt in batch)
    bsz = len(batch)

    src_batch = torch.full((bsz, max_src), pad_idx_src, dtype=torch.long)
    tgt_batch = torch.full((bsz, max_tgt), pad_idx_tgt, dtype=torch.long)
    batch_ids: list[int] = []

    for i, (ex_id, src, tgt) in enumerate(batch):
        batch_ids.append(ex_id)
        src_batch[i, : len(src)] = torch.tensor(src, dtype=torch.long)
        tgt_batch[i, : len(tgt)] = torch.tensor(tgt, dtype=torch.long)
    return src_batch, tgt_batch, batch_ids
