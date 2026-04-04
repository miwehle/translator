from __future__ import annotations

from collections.abc import Sequence
from math import ceil
from pathlib import Path

import torch
import yaml

from ..model import Seq2Seq
from ..training.checkpointing import checkpoint_manifest_path
from ..training.config import ModelConfig
from .tokenizer import TokenizerProtocol, create_tokenizer


def _estimate_target_token_count(source_token_count: int) -> int:
    """Estimate a safe DE->EN target token count for greedy decoding."""
    if source_token_count < 0:
        raise ValueError("source_token_count must be non-negative.")
    return ceil(2.8 * source_token_count) + 5


class TranslationFailure(RuntimeError):
    pass


def _invalid_predicted_ids(
    predicted_ids: Sequence[int], tokenizer: TokenizerProtocol
) -> list[int]:
    tokenizer_vocab_size = getattr(tokenizer, "vocab_size", None)
    if not isinstance(tokenizer_vocab_size, int):
        return []
    return [token_id for token_id in predicted_ids if token_id >= tokenizer_vocab_size]


def _build_translation_failure(
    source_text: str, encoded_source_ids: Sequence[int], predicted_ids: Sequence[int],
    tokenizer: TokenizerProtocol, tgt_bos_id: int | None, cause: Exception,
) -> TranslationFailure:
    return TranslationFailure(
        "Translation decode failed. "
        f"source_text={source_text!r} "
        f"src_ids={list(encoded_source_ids)} "
        f"predicted_ids={list(predicted_ids)} "
        f"invalid_predicted_ids={_invalid_predicted_ids(predicted_ids, tokenizer)} "
        f"tokenizer_vocab_size={getattr(tokenizer, 'vocab_size', None)} "
        f"tgt_bos_id={tgt_bos_id} "
        f"cause={cause!r}"
    )


def _load_manifest(checkpoint_path: str | Path) -> dict[str, object]:
    manifest_path = checkpoint_manifest_path(Path(checkpoint_path).parent)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Checkpoint manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle) or {}
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid checkpoint manifest: {manifest_path}")
    return manifest


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def _create_model(
    model_config: ModelConfig, checkpoint_dataset: dict[str, object], device: torch.device
) -> Seq2Seq:
    return Seq2Seq(
        src_vocab_size=int(checkpoint_dataset["src_vocab_size"]),
        tgt_vocab_size=int(checkpoint_dataset["tgt_vocab_size"]),
        d_model=model_config.d_model,
        ff_dim=model_config.ff_dim,
        num_heads=model_config.num_heads,
        num_layers=model_config.num_layers,
        src_pad_idx=int(checkpoint_dataset["src_pad_id"]),
        tgt_pad_idx=int(checkpoint_dataset["tgt_pad_id"]),
        tgt_sos_idx=int(checkpoint_dataset["tgt_bos_id"]),
        dropout=model_config.dropout,
        max_len=model_config.max_seq_len,
        attention=model_config.attention,
    ).to(device)


class Translator:
    def __init__(
        self, model: Seq2Seq, tokenizer: TokenizerProtocol, device: torch.device,
        tgt_bos_id: int | None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.tgt_bos_id = tgt_bos_id

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str | Path, device: str | torch.device | None = None
    ) -> "Translator":
        checkpoint_file = Path(checkpoint_path)
        manifest = _load_manifest(checkpoint_file)
        checkpoint_tokenizer = manifest["tokenizer"]
        resolved_device = _resolve_device(device)
        model = _create_model(
            ModelConfig(**manifest["model_config"]), checkpoint_tokenizer, resolved_device
        )
        payload = torch.load(checkpoint_file, map_location=resolved_device)
        model.load_state_dict(payload["model_state_dict"])
        tokenizer = create_tokenizer("hf", [], checkpoint_tokenizer["model_name"])
        return cls(model, tokenizer, resolved_device, int(checkpoint_tokenizer["tgt_bos_id"]))

    def translate(self, text: str) -> str:
        eos_idx = self.tokenizer.eos_token_id
        if eos_idx is None:
            raise ValueError("Tokenizer has no eos_token_id for translation.")
        encoded_source_ids = self.tokenizer.encode(text)
        predicted_ids = self.model.translate(
            encoded_source_ids,
            max_len=_estimate_target_token_count(len(encoded_source_ids)),
            device=self.device, eos_idx=eos_idx
        )
        try:
            return self.tokenizer.decode(predicted_ids)
        except Exception as exc:
            raise _build_translation_failure(
                text, encoded_source_ids, predicted_ids, self.tokenizer,
                self.tgt_bos_id, exc
            ) from exc

    def translate_many(self, texts: Sequence[str]) -> list[str]:
        was_training = self.model.training
        self.model.eval()
        try:
            return [self.translate(text) for text in texts]
        finally:
            self.model.train(was_training)
