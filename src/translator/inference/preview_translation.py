from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from ..model import Seq2Seq
from .tokenizer import TokenizerProtocol, create_tokenizer
from .translator import Translator


def _load_preview_tokenizer(tokenizer_model_name: str) -> TokenizerProtocol:
    return create_tokenizer("hf", [], tokenizer_model_name)


def translate_examples(
    model: Seq2Seq,
    tokenizer: TokenizerProtocol,
    texts: Sequence[str],
    device: torch.device,
    *,
    tgt_bos_id: int | None,
) -> list[tuple[str, str]]:
    translations = Translator(model, tokenizer, device, tgt_bos_id).translate_many(texts)
    return list(zip(texts, translations, strict=True))


def create_translation_preview_fn(
    translate_every: int | None,
    translate_examples_texts: Sequence[str],
    tokenizer_model_name: str | None,
    tgt_bos_id: int | None,
    model: Seq2Seq,
    device: torch.device,
) -> Callable[[], list[tuple[str, str]]] | None:
    if translate_every is None or not translate_examples_texts:
        return None
    if tokenizer_model_name is None:
        raise ValueError("Preview translation requires dataset tokenizer_model_name.")

    preview_tokenizer = _load_preview_tokenizer(tokenizer_model_name)

    def translation_preview_fn() -> list[tuple[str, str]]:
        return translate_examples(
            model,
            preview_tokenizer,
            translate_examples_texts,
            device,
            tgt_bos_id=tgt_bos_id,
        )

    return translation_preview_fn
