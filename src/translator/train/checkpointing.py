import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from ..data import deserialize_tokenizer, serialize_tokenizer
from ..data.factory import TokenizerProtocol
from ..model import Seq2Seq

BuildModelFn = Callable[
    [
        argparse.Namespace,
        TokenizerProtocol,
        TokenizerProtocol,
        torch.device,
        str | None,
    ],
    Seq2Seq,
]


def save_checkpoint(
    path: str,
    model: Seq2Seq,
    src_tokenizer: TokenizerProtocol,
    tgt_tokenizer: TokenizerProtocol,
    args: argparse.Namespace,
) -> None:
    checkpoint_dir = Path(path).parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "src_tokenizer": serialize_tokenizer(src_tokenizer),
        "tgt_tokenizer": serialize_tokenizer(tgt_tokenizer),
        "hparams": {
            "emb_dim": args.emb_dim,
            "hidden_dim": args.hidden_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "attention": getattr(args, "attention", "torch"),
            "tokenizer": getattr(args, "tokenizer", "custom"),
        },
    }
    torch.save(payload, path)
    print(f"\nCheckpoint gespeichert: {path}")


def load_checkpoint(path: str, device: torch.device) -> dict[str, Any]:
    ckpt = torch.load(path, map_location=device)
    return ckpt


def _load_tokenizers_from_checkpoint_payload(
    ckpt: dict[str, Any],
) -> tuple[TokenizerProtocol, TokenizerProtocol]:
    src_data = ckpt.get("src_tokenizer")
    tgt_data = ckpt.get("tgt_tokenizer")
    if not isinstance(src_data, dict) or not isinstance(tgt_data, dict):
        raise ValueError("Checkpoint enthaelt keine gueltigen Tokenizer-Daten.")
    return deserialize_tokenizer(src_data), deserialize_tokenizer(tgt_data)


def _load_model_from_checkpoint_payload(
    ckpt: dict[str, Any],
    src_tokenizer: TokenizerProtocol,
    tgt_tokenizer: TokenizerProtocol,
    device: torch.device,
    args: argparse.Namespace,
    build_model_fn: BuildModelFn,
) -> Seq2Seq:
    hparams = ckpt["hparams"]
    trained_attention = hparams.get("attention", "torch")
    requested_attention = getattr(args, "attention", "torch")
    if requested_attention != trained_attention:
        raise ValueError(
            f"Attention-Mismatch: Checkpoint nutzt '{trained_attention}', "
            f"CLI fordert '{requested_attention}'."
        )
    trained_tokenizer = hparams.get("tokenizer", "custom")
    requested_tokenizer = getattr(args, "tokenizer", "custom")
    if requested_tokenizer != trained_tokenizer:
        raise ValueError(
            f"Tokenizer-Mismatch: Checkpoint nutzt '{trained_tokenizer}', "
            f"CLI fordert '{requested_tokenizer}'."
        )

    model_args = argparse.Namespace(
        emb_dim=hparams["emb_dim"],
        hidden_dim=hparams["hidden_dim"],
        num_heads=hparams["num_heads"],
        num_layers=hparams.get("num_layers", 2),
        dropout=hparams.get("dropout", 0.1),
    )
    model = build_model_fn(
        model_args,
        src_tokenizer,
        tgt_tokenizer,
        device,
        requested_attention,
    )
    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint passt nicht zur aktuellen Modellarchitektur. "
            "Bitte mit der aktuellen Transformer-Version neu trainieren."
        ) from exc
    model.eval()
    return model


def load_inference_components(
    path: str,
    device: torch.device,
    args: argparse.Namespace,
    build_model_fn: BuildModelFn | None = None,
) -> tuple[Seq2Seq, TokenizerProtocol, TokenizerProtocol]:
    if build_model_fn is None:
        # Late import to avoid circular import at module load time.
        from .training import build_model as build_model_fn

    ckpt = load_checkpoint(path, device)
    src_tokenizer, tgt_tokenizer = _load_tokenizers_from_checkpoint_payload(ckpt)
    model = _load_model_from_checkpoint_payload(
        ckpt, src_tokenizer, tgt_tokenizer, device, args, build_model_fn
    )
    return model, src_tokenizer, tgt_tokenizer
