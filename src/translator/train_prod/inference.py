from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from ..data import HuggingFaceTokenizerAdapter, TokenizerProtocol
from ..data_prod import DatasetMetadata
from ..model import Seq2Seq
from .config import ModelConfig, TrainConfig


def load_preview_tokenizer(tokenizer_model_name: str) -> TokenizerProtocol:
    return HuggingFaceTokenizerAdapter.from_pretrained(tokenizer_model_name)


@dataclass(frozen=True)
class PreviewTranslationFailure(RuntimeError):
    source_text: str
    encoded_source_ids: list[int]
    predicted_ids: list[int]
    tokenizer_vocab_size: int | None
    tokenizer_bos_id: int | None
    tokenizer_eos_id: int | None
    tokenizer_pad_id: int | None
    dataset_tgt_bos_id: int | None
    dataset_tgt_eos_id: int | None
    dataset_tgt_pad_id: int | None
    max_len: int

    def __init__(
        self,
        *,
        source_text: str,
        encoded_source_ids: list[int],
        predicted_ids: list[int],
        tokenizer: TokenizerProtocol,
        dataset_metadata: DatasetMetadata,
        max_len: int,
        cause: Exception,
    ) -> None:
        tokenizer_vocab_size = getattr(tokenizer, "vocab_size", None)
        invalid_ids = (
            [
                token_id
                for token_id in predicted_ids
                if isinstance(tokenizer_vocab_size, int)
                and token_id >= tokenizer_vocab_size
            ]
            if isinstance(tokenizer_vocab_size, int)
            else []
        )
        message = (
            "Preview translation decode failed. "
            f"source_text={source_text!r} "
            f"src_ids={encoded_source_ids} "
            f"predicted_ids={predicted_ids} "
            f"predicted_len={len(predicted_ids)} "
            f"max_predicted_id={max(predicted_ids) if predicted_ids else None} "
            f"invalid_predicted_ids={invalid_ids} "
            f"tokenizer_vocab_size={tokenizer_vocab_size} "
            f"tokenizer_bos_id={getattr(tokenizer, 'bos_token_id', None)} "
            f"tokenizer_eos_id={getattr(tokenizer, 'eos_token_id', None)} "
            f"tokenizer_pad_id={getattr(tokenizer, 'pad_token_id', None)} "
            f"dataset_tgt_bos_id={dataset_metadata.tgt_bos_id} "
            f"dataset_tgt_eos_id={dataset_metadata.tgt_eos_id} "
            f"dataset_tgt_pad_id={dataset_metadata.tgt_pad_id} "
            f"max_len={max_len} "
            f"cause={cause!r}"
        )
        super().__init__(message)
        object.__setattr__(self, "source_text", source_text)
        object.__setattr__(self, "encoded_source_ids", list(encoded_source_ids))
        object.__setattr__(self, "predicted_ids", list(predicted_ids))
        object.__setattr__(self, "tokenizer_vocab_size", tokenizer_vocab_size)
        object.__setattr__(self, "tokenizer_bos_id", getattr(tokenizer, "bos_token_id", None))
        object.__setattr__(self, "tokenizer_eos_id", getattr(tokenizer, "eos_token_id", None))
        object.__setattr__(self, "tokenizer_pad_id", getattr(tokenizer, "pad_token_id", None))
        object.__setattr__(self, "dataset_tgt_bos_id", dataset_metadata.tgt_bos_id)
        object.__setattr__(self, "dataset_tgt_eos_id", dataset_metadata.tgt_eos_id)
        object.__setattr__(self, "dataset_tgt_pad_id", dataset_metadata.tgt_pad_id)
        object.__setattr__(self, "max_len", max_len)
        object.__setattr__(self, "__cause__", cause)


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
            try:
                translated_text = tokenizer.decode(predicted_ids)
            except Exception as exc:
                raise PreviewTranslationFailure(
                    source_text=text,
                    encoded_source_ids=encoded_source_ids,
                    predicted_ids=predicted_ids,
                    tokenizer=tokenizer,
                    dataset_metadata=dataset_metadata,
                    max_len=max_len,
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
