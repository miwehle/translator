from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from ..data_prod import DatasetMetadata, collate_fn_prod
from ..model import Seq2Seq
from ..types import Example
from .config import DataLoaderConfig

if TYPE_CHECKING:
    from .config import ModelConfig


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


class Factory:
    def __init__(self, dataset_metadata: DatasetMetadata) -> None:
        self.dataset_metadata = dataset_metadata

    def _collate_examples(
        self,
        batch: list[Example],
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        normalized = [
            (
                int(item[self.dataset_metadata.id_field]),
                [int(x) for x in item[self.dataset_metadata.src_field]],
                [int(x) for x in item[self.dataset_metadata.tgt_field]],
            )
            for item in batch
        ]
        return collate_fn_prod(
            normalized,
            pad_idx_src=self.dataset_metadata.src_pad_id,
            pad_idx_tgt=self.dataset_metadata.tgt_pad_id,
        )

    def create_model(
        self,
        *,
        model_config: ModelConfig,
        device: torch.device,
    ) -> Seq2Seq:
        return Seq2Seq(
            src_vocab_size=self.dataset_metadata.src_vocab_size,
            tgt_vocab_size=self.dataset_metadata.tgt_vocab_size,
            d_model=model_config.d_model,
            ff_dim=model_config.ff_dim,
            num_heads=model_config.num_heads,
            num_layers=model_config.num_layers,
            src_pad_idx=self.dataset_metadata.src_pad_id,
            tgt_pad_idx=self.dataset_metadata.tgt_pad_id,
            tgt_sos_idx=self.dataset_metadata.tgt_bos_id,
            dropout=model_config.dropout,
            max_len=model_config.max_seq_len,
            attention=model_config.attention,
        ).to(device)

    def create_data_loader(
        self,
        examples: Iterable[Example] | Sequence[Example],
        *,
        data_loader_config: DataLoaderConfig,
        device: torch.device,
    ) -> DataLoader:
        if hasattr(examples, "__len__") and hasattr(examples, "__getitem__"):
            dataset = examples
            loader_shuffle = data_loader_config.shuffle
        else:
            dataset = _ExampleIterableDataset(examples)
            loader_shuffle = False

        loader_kwargs: dict[str, Any] = {
            "batch_size": data_loader_config.batch_size,
            "shuffle": loader_shuffle,
            "collate_fn": self._collate_examples,
            "num_workers": data_loader_config.num_workers,
            "pin_memory": (
                device.type == "cuda"
                if data_loader_config.pin_memory is None
                else data_loader_config.pin_memory
            ),
        }
        if data_loader_config.num_workers > 0:
            loader_kwargs["persistent_workers"] = (
                True
                if data_loader_config.persistent_workers is None
                else data_loader_config.persistent_workers
            )
            if data_loader_config.prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = data_loader_config.prefetch_factor

        return DataLoader(dataset, **loader_kwargs)
