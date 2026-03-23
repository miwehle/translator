from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from ..data import HuggingFaceTokenizerAdapter, TokenizerProtocol
from ..data_prod import DatasetMetadata
from ..model import Seq2Seq
from .config import ModelConfig, TrainConfig


def load_preview_tokenizer(tokenizer_model_name: str) -> TokenizerProtocol:
    return HuggingFaceTokenizerAdapter.from_pretrained(tokenizer_model_name)


def _prepare_predicted_ids_for_preview_decode(
    predicted_ids: Sequence[int],
    *,
    tokenizer: TokenizerProtocol,
    dataset_metadata: DatasetMetadata,
) -> list[int]:
    prepared_ids = list(predicted_ids)

    dataset_bos_id = dataset_metadata.tgt_bos_id
    tokenizer_bos_id = getattr(tokenizer, "bos_token_id", None)
    tokenizer_vocab_size = getattr(tokenizer, "vocab_size", None)
    should_strip_leading_dataset_bos = (
        bool(prepared_ids)
        and dataset_bos_id is not None
        and prepared_ids[0] == dataset_bos_id
        and (
            tokenizer_bos_id is None
            or tokenizer_bos_id != dataset_bos_id
            or (
                isinstance(tokenizer_vocab_size, int)
                and dataset_bos_id >= tokenizer_vocab_size
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
    dataset_metadata: DatasetMetadata,
    cause: Exception,
) -> PreviewTranslationFailure:
    return PreviewTranslationFailure(
        "Preview translation decode failed. "
        f"source_text={source_text!r} "
        f"src_ids={list(encoded_source_ids)} "
        f"predicted_ids={list(predicted_ids)} "
        f"invalid_predicted_ids={_invalid_predicted_ids(predicted_ids, tokenizer=tokenizer)} "
        f"tokenizer_vocab_size={getattr(tokenizer, 'vocab_size', None)} "
        f"dataset_tgt_bos_id={dataset_metadata.tgt_bos_id} "
        f"cause={cause!r}"
    )


def translate_examples(
    model: Seq2Seq,
    tokenizer: TokenizerProtocol,
    texts: Sequence[str],
    device: torch.device,
    *,
    max_len: int,
    dataset_metadata: DatasetMetadata,
) -> list[tuple[str, str]]:
    eos_idx = tokenizer.eos_token_id
    if eos_idx is None:
        raise ValueError("Tokenizer has no eos_token_id for preview translation.")

    was_training = model.training
    model.eval()
    try:
        translations: list[tuple[str, str]] = []
        for text in texts:
            encoded_source_ids = tokenizer.encode(text)
            predicted_ids = model.translate(
                encoded_source_ids,
                max_len=max_len,
                device=device,
                eos_idx=eos_idx,
            )
            prepared_predicted_ids = _prepare_predicted_ids_for_preview_decode(
                predicted_ids,
                tokenizer=tokenizer,
                dataset_metadata=dataset_metadata,
            )
            try:
                translated_text = tokenizer.decode(prepared_predicted_ids)
            except Exception as exc:
                raise _build_preview_translation_failure(
                    source_text=text,
                    encoded_source_ids=encoded_source_ids,
                    predicted_ids=predicted_ids,
                    tokenizer=tokenizer,
                    dataset_metadata=dataset_metadata,
                    cause=exc,
                ) from exc
            translations.append((text, translated_text))
        return translations
    finally:
        model.train(was_training)


def create_translation_preview_fn(
    train_config: TrainConfig,
    model_config: ModelConfig,
    dataset_metadata: DatasetMetadata,
    model: Seq2Seq,
    device: torch.device,
) -> Callable[[], list[tuple[str, str]]] | None:
    if train_config.translate_every is None or not train_config.translate_examples:
        return None
    if dataset_metadata.tokenizer_model_name is None:
        raise ValueError("Preview translation requires dataset tokenizer_model_name.")

    preview_tokenizer = load_preview_tokenizer(dataset_metadata.tokenizer_model_name)
    preview_max_len = min(model_config.max_seq_len, 64)

    def translation_preview_fn() -> list[tuple[str, str]]:
        return translate_examples(
            model,
            preview_tokenizer,
            train_config.translate_examples,
            device,
            max_len=preview_max_len,
            dataset_metadata=dataset_metadata,
        )

    return translation_preview_fn
