from __future__ import annotations

from pathlib import Path

import torch
import yaml

from translator.inference import Translator
from translator.model import Seq2Seq


class _FakeTokenizer:
    eos_token_id = 9
    bos_token_id = None
    vocab_size = 5

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
        return [len(text), 1]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        decoded_ids = ids
        return "Hello" if decoded_ids == [4, 9] else "decoded-text"


class _FakeModel:
    def __init__(self) -> None:
        self.training = True
        self.calls: list[tuple[list[int], int, torch.device, int]] = []

    def eval(self) -> "_FakeModel":
        self.training = False
        return self

    def train(self, mode: bool = True) -> "_FakeModel":
        self.training = mode
        return self

    def translate(
        self, src_ids: list[int], max_len: int, device: torch.device, eos_idx: int
    ) -> list[int]:
        self.calls.append((src_ids, max_len, device, eos_idx))
        return [4, eos_idx]


class TestTranslator:
    def test_translate_many_restores_training_mode(self) -> None:
        model = _FakeModel()
        translator = Translator(model, _FakeTokenizer(), torch.device("cpu"), 7)

        out = translator.translate_many(["Hallo"])

        assert out == ["Hello"]
        assert model.training is True
        assert model.calls == [([5, 1], 11, torch.device("cpu"), 9)]

    def test_from_checkpoint_loads_model_and_tokenizer(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        model = Seq2Seq(16, 16, 8, 16, 2, 1, 0, 1, 2, dropout=0.0, max_len=32)
        checkpoint_path = tmp_path / "checkpoint.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": {},
            },
            checkpoint_path,
        )
        (tmp_path / "checkpoint_manifest.yaml").write_text(
            yaml.safe_dump(
                {
                    "checkpoint_file": "checkpoint.pt",
                    "model_config": {
                        "d_model": 8,
                        "ff_dim": 16,
                        "num_heads": 2,
                        "num_layers": 1,
                        "dropout": 0.0,
                        "max_seq_len": 32,
                        "attention": "torch",
                    },
                    "optimizer": {"type": "adam", "lr": 0.001},
                    "tokenizer": {
                        "model_name": "dummy-tokenizer",
                        "src_vocab_size": 16,
                        "tgt_vocab_size": 16,
                        "src_pad_id": 0,
                        "tgt_pad_id": 1,
                        "tgt_bos_id": 2,
                    },
                }
            ),
            encoding="utf-8",
        )
        fake_tokenizer = _FakeTokenizer()
        monkeypatch.setattr(
            "translator.inference.translator.create_tokenizer",
            lambda tokenizer, texts, hf_tokenizer_name: fake_tokenizer,
        )

        translator = Translator.from_checkpoint(checkpoint_path, "cpu")

        assert translator.tokenizer is fake_tokenizer
        assert translator.device == torch.device("cpu")
        assert translator.tgt_bos_id == 2
