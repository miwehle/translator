import argparse

import torch

from translator.data import (
    Tokenizer,
    deserialize_tokenizer,
    serialize_tokenizer,
    tiny_parallel_corpus,
)
from translator.data.factory import (
    TOKENIZER_CHOICES,
    TokenizerProtocol,
    create_tokenizer,
)
from translator.train import build_model


class HfLikeTokenizer:
    def __init__(self, vocab_size: int = 32):
        self._vocab_size = vocab_size

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        del text, add_special_tokens
        return [1, 2, 3]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del ids, skip_special_tokens
        return "decoded"

    def __call__(
        self,
        texts: str | list[str],
        padding: bool | str = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, object]:
        del texts, padding, truncation, max_length, return_tensors, add_special_tokens
        return {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def pad_token_id(self) -> int:
        return 0

    @property
    def bos_token_id(self) -> int:
        return 2

    @property
    def eos_token_id(self) -> int:
        return 1


def test_custom_tokenizer_implements_tokenizer_protocol():
    pairs = tiny_parallel_corpus()
    tokenizer = Tokenizer.build([p[0] for p in pairs])
    assert isinstance(tokenizer, TokenizerProtocol)


def test_build_model_accepts_hf_like_tokenizer_protocol():
    args = argparse.Namespace(
        emb_dim=16,
        hidden_dim=32,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
        attention="torch",
    )
    tokenizer = HfLikeTokenizer(vocab_size=64)

    model = build_model(args, tokenizer, tokenizer, torch.device("cpu"))

    assert model is not None


def test_tokenizer_choices_include_custom_and_hf():
    assert TOKENIZER_CHOICES == ("custom", "hf")


def test_create_tokenizer_custom_builds_project_tokenizer():
    pairs = tiny_parallel_corpus()
    tokenizer = create_tokenizer("custom", [p[0] for p in pairs], "unused")

    assert isinstance(tokenizer, Tokenizer)


def test_custom_tokenizer_checkpoint_payload_roundtrip():
    pairs = tiny_parallel_corpus()
    tokenizer = Tokenizer.build([p[0] for p in pairs])

    payload = serialize_tokenizer(tokenizer)
    restored = deserialize_tokenizer(payload)

    assert isinstance(restored, Tokenizer)
    assert restored.stoi == tokenizer.stoi
    assert restored.itos == tokenizer.itos
