from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from ..data import HuggingFaceTokenizerAdapter, TokenizerProtocol
from ..model import Seq2Seq
from .config import ModelConfig, TrainConfig


def load_preview_tokenizer(tokenizer_model_name: str) -> TokenizerProtocol:
    return HuggingFaceTokenizerAdapter.from_pretrained(tokenizer_model_name)


def translate_examples(
    model: Seq2Seq,
    tokenizer: TokenizerProtocol,
    texts: Sequence[str],
    device: torch.device,
    *,
    max_len: int,
) -> list[tuple[str, str]]:
    eos_idx = tokenizer.eos_token_id
    if eos_idx is None:
        raise ValueError("Tokenizer has no eos_token_id for preview translation.")

    was_training = model.training
    model.eval()
    try:
        return [
            (
                text,
                tokenizer.decode(
                    model.translate(
                        tokenizer.encode(text),
                        max_len=max_len,
                        device=device,
                        eos_idx=eos_idx,
                    )
                ),
            )
            for text in texts
        ]
    finally:
        model.train(was_training)


def create_translation_preview_fn(
    train_config: TrainConfig,
    model_config: ModelConfig,
    tokenizer_model_name: str | None,
    model: Seq2Seq,
    device: torch.device,
) -> Callable[[], list[tuple[str, str]]] | None:
    if train_config.translate_every is None or not train_config.translate_examples:
        return None
    if tokenizer_model_name is None:
        raise ValueError("Preview translation requires dataset tokenizer_model_name.")

    preview_tokenizer = load_preview_tokenizer(tokenizer_model_name)
    preview_max_len = min(model_config.max_seq_len, 64)

    def translation_preview_fn() -> list[tuple[str, str]]:
        return translate_examples(
            model,
            preview_tokenizer,
            train_config.translate_examples,
            device,
            max_len=preview_max_len,
        )

    return translation_preview_fn
