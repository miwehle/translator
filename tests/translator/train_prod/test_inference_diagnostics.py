from __future__ import annotations

import torch
import pytest

from translator.data_prod import DatasetMetadata
from translator.train_prod.inference import (
    PreviewTranslationFailure,
    translate_examples,
)


class _FakeTokenizer:
    vocab_size = 8
    bos_token_id = 1
    eos_token_id = 0
    pad_token_id = 7

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [self.bos_token_id, 3, self.eos_token_id]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        if any(token_id >= self.vocab_size for token_id in ids):
            raise IndexError("piece id is out of range.")
        return "ok"


class _FakeModel:
    training = True

    def eval(self) -> None:
        self.training = False

    def train(self, mode: bool = True) -> None:
        self.training = mode

    def translate(
        self, src_ids: list[int], max_len: int, device: torch.device, eos_idx: int
    ) -> list[int]:
        return [1, 8, eos_idx]


class _PreviewBosModel(_FakeModel):
    def translate(
        self, src_ids: list[int], max_len: int, device: torch.device, eos_idx: int
    ) -> list[int]:
        return [8, 1, 3, eos_idx]


class _InvalidTokenAfterBosModel(_FakeModel):
    def translate(
        self, src_ids: list[int], max_len: int, device: torch.device, eos_idx: int
    ) -> list[int]:
        return [8, 1, 8, eos_idx]


def test_translate_examples_raises_preview_failure_with_diagnostics() -> None:
    dataset_metadata = DatasetMetadata(
        schema_version=1,
        tokenizer_model_name="dummy-tokenizer",
        src_lang="de",
        tgt_lang="en",
        id_field="id",
        src_field="src_ids",
        tgt_field="tgt_ids",
        base_vocab_size=8,
        src_vocab_size=8,
        tgt_vocab_size=10,
        src_pad_id=7,
        tgt_pad_id=7,
        tgt_bos_id=8,
        tgt_eos_id=0,
        num_examples=1,
    )

    with pytest.raises(PreviewTranslationFailure) as exc_info:
        translate_examples(
            _InvalidTokenAfterBosModel(),
            _FakeTokenizer(),
            ["Hallo Welt"],
            torch.device("cpu"),
            max_len=16,
            dataset_metadata=dataset_metadata,
        )

    message = str(exc_info.value)
    assert "source_text='Hallo Welt'" in message
    assert "predicted_ids=[8, 1, 8, 0]" in message
    assert "invalid_predicted_ids=[8, 8]" in message
    assert "tokenizer_vocab_size=8" in message
    assert "dataset_tgt_bos_id=8" in message


def test_translate_examples_strips_leading_dataset_bos_for_preview_decode() -> None:
    dataset_metadata = DatasetMetadata(
        schema_version=1,
        tokenizer_model_name="dummy-tokenizer",
        src_lang="de",
        tgt_lang="en",
        id_field="id",
        src_field="src_ids",
        tgt_field="tgt_ids",
        base_vocab_size=8,
        src_vocab_size=8,
        tgt_vocab_size=10,
        src_pad_id=7,
        tgt_pad_id=7,
        tgt_bos_id=8,
        tgt_eos_id=0,
        num_examples=1,
    )

    translations = translate_examples(
        _PreviewBosModel(),
        _FakeTokenizer(),
        ["Hallo Welt"],
        torch.device("cpu"),
        max_len=16,
        dataset_metadata=dataset_metadata,
    )

    assert translations == [("Hallo Welt", "ok")]
