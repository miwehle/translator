from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import Dataset


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


class ArrowTranslationDataset(Dataset):
    def __init__(
        self,
        dataset_path: str | Path,
        *,
        id_field: str = "id",
        src_field: str = "src_ids",
        tgt_field: str = "tgt_ids",
        max_examples: int | None = None,
    ):
        self.records = load_arrow_records(dataset_path)
        if max_examples is not None:
            self.records = self.records.select(
                range(min(max_examples, len(self.records)))
            )

        self.id_field = id_field
        self.src_field = src_field
        self.tgt_field = tgt_field

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[int, list[int], list[int]]:
        item = cast(Mapping[str, Any], self.records[idx])
        ex_id = int(item[self.id_field])
        src = [int(x) for x in item[self.src_field]]
        tgt = [int(x) for x in item[self.tgt_field]]
        return ex_id, src, tgt


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


def iter_records(
    records: Iterable[Mapping[str, Any]],
    *,
    id_field: str = "id",
    src_field: str = "src_ids",
    tgt_field: str = "tgt_ids",
):
    for item in records:
        yield (
            int(item[id_field]),
            [int(x) for x in item[src_field]],
            [int(x) for x in item[tgt_field]],
        )
