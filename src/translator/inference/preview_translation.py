from __future__ import annotations

from collections.abc import Callable, Sequence
from math import ceil

import torch

from ..model import Seq2Seq
from .tokenizer import HuggingFaceTokenizerAdapter, TokenizerProtocol


def _load_preview_tokenizer(tokenizer_model_name: str) -> TokenizerProtocol:
    return HuggingFaceTokenizerAdapter.from_pretrained(tokenizer_model_name)


def _estimate_english_token_count(german_token_count: int) -> int:
    """Estimate a safe DE->EN target token count for preview decoding.

    On the full local Europarl DE->EN training set with
    Helsinki-NLP/opus-mt-de-en, p99(tgt_len / src_len) is about 2.74.
    """
    if german_token_count < 0:
        raise ValueError("german_token_count must be non-negative.")
    return ceil(2.8 * german_token_count) + 5


def _prepare_predicted_ids_for_preview_decode(
    predicted_ids: Sequence[int],
    *,
    tokenizer: TokenizerProtocol,
    tgt_bos_id: int | None,
) -> list[int]:
    prepared_ids = list(predicted_ids)

    tokenizer_bos_id = getattr(tokenizer, "bos_token_id", None)
    tokenizer_vocab_size = getattr(tokenizer, "vocab_size", None)
    should_strip_leading_dataset_bos = (
        bool(prepared_ids)
        and tgt_bos_id is not None
        and prepared_ids[0] == tgt_bos_id
        and (
            tokenizer_bos_id is None or tokenizer_bos_id != tgt_bos_id
            or (
                isinstance(tokenizer_vocab_size, int)
                and tgt_bos_id >= tokenizer_vocab_size
            )
        )
    )
    if should_strip_leading_dataset_bos:
        prepared_ids = prepared_ids[1:]

    return prepared_ids


class PreviewTranslationFailure(RuntimeError):
    pass


def _invalid_predicted_ids(
    predicted_ids: Sequence[int], *, tokenizer: TokenizerProtocol
) -> list[int]:
    tokenizer_vocab_size = getattr(tokenizer, "vocab_size", None)
    if not isinstance(tokenizer_vocab_size, int):
        return []
    return [
        token_id
        for token_id in predicted_ids
        if token_id >= tokenizer_vocab_size
    ]


def _build_preview_translation_failure(
    *,
    source_text: str,
    encoded_source_ids: Sequence[int],
    predicted_ids: Sequence[int],
    tokenizer: TokenizerProtocol,
    tgt_bos_id: int | None,
    cause: Exception,
) -> PreviewTranslationFailure:
    return PreviewTranslationFailure(
        "Preview translation decode failed. "
        f"source_text={source_text!r} "
        f"src_ids={list(encoded_source_ids)} "
        f"predicted_ids={list(predicted_ids)} "
        f"invalid_predicted_ids={_invalid_predicted_ids(predicted_ids, tokenizer=tokenizer)} "
        f"tokenizer_vocab_size={getattr(tokenizer, 'vocab_size', None)} "
        f"dataset_tgt_bos_id={tgt_bos_id} "
        f"cause={cause!r}"
    )


def translate_examples(
    model: Seq2Seq,
    tokenizer: TokenizerProtocol,
    texts: Sequence[str],
    device: torch.device,
    *,
    tgt_bos_id: int | None,
) -> list[tuple[str, str]]:
    def translate_example(text: str, eos_idx: int) -> str:
        encoded_source_ids = tokenizer.encode(text)
        max_len = _estimate_english_token_count(len(encoded_source_ids))
        predicted_ids = model.translate(
            encoded_source_ids,
            max_len=max_len,
            device=device,
            eos_idx=eos_idx,
        )
        prepared_predicted_ids = _prepare_predicted_ids_for_preview_decode(
            predicted_ids,
            tokenizer=tokenizer,
            tgt_bos_id=tgt_bos_id,
        )
        try:
            return tokenizer.decode(prepared_predicted_ids)
        except Exception as exc:
            raise _build_preview_translation_failure(
                source_text=text,
                encoded_source_ids=encoded_source_ids,
                predicted_ids=predicted_ids,
                tokenizer=tokenizer,
                tgt_bos_id=tgt_bos_id,
                cause=exc,
            ) from exc

    eos_idx = tokenizer.eos_token_id
    if eos_idx is None:
        raise ValueError("Tokenizer has no eos_token_id for preview translation.")

    was_training = model.training
    model.eval()
    try:
        translations: list[tuple[str, str]] = []
        for text in texts:
            translated_text = translate_example(text, eos_idx)
            translations.append((text, translated_text))
        return translations
    finally:
        model.train(was_training)


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
