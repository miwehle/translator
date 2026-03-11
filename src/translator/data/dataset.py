import random

import torch
from torch.utils.data import Dataset

from .factory import TokenizerProtocol


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


class TranslationDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[str, str]],
        src_tokenizer: TokenizerProtocol,
        tgt_tokenizer: TokenizerProtocol,
    ):
        self.data = [
            (src_tokenizer.encode(src), tgt_tokenizer.encode(tgt)) for src, tgt in pairs
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        return self.data[idx]


def collate_fn(
    batch: list[tuple[list[int], list[int]]], pad_idx_src: int, pad_idx_tgt: int
) -> tuple[torch.Tensor, torch.Tensor]:
    max_src = max(len(src) for src, _ in batch)
    max_tgt = max(len(tgt) for _, tgt in batch)
    bsz = len(batch)

    src_batch = torch.full((bsz, max_src), pad_idx_src, dtype=torch.long)
    tgt_batch = torch.full((bsz, max_tgt), pad_idx_tgt, dtype=torch.long)

    for i, (src, tgt) in enumerate(batch):
        src_batch[i, : len(src)] = torch.tensor(src, dtype=torch.long)
        tgt_batch[i, : len(tgt)] = torch.tensor(tgt, dtype=torch.long)
    return src_batch, tgt_batch


def tiny_parallel_corpus() -> list[tuple[str, str]]:
    return [
        ("ich bin muede", "i am tired"),
        ("ich bin hungrig", "i am hungry"),
        ("ich liebe kaffee", "i love coffee"),
        ("ich trinke wasser", "i drink water"),
        ("guten morgen", "good morning"),
        ("gute nacht", "good night"),
        ("wie geht es dir", "how are you"),
        ("mir geht es gut", "i am fine"),
        ("danke", "thank you"),
        ("bis spaeter", "see you later"),
        ("wo ist der bahnhof", "where is the train station"),
        ("ich lerne deutsch", "i am learning german"),
    ]


__all__ = [
    "TranslationDataset",
    "collate_fn",
    "set_seed",
    "tiny_parallel_corpus",
]
