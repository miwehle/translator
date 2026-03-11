import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..data import (
    TranslationDataset,
    collate_fn,
    set_seed,
    tiny_parallel_corpus,
)
from ..data import factory as data_factory
from ..data.factory import TOKENIZER_CHOICES, TokenizerProtocol
from ..model import Seq2Seq
from ..model.factory import ATTENTION_CHOICES
from .checkpointing import load_inference_components, save_checkpoint


def build_model(
    args: argparse.Namespace,
    src_tokenizer: TokenizerProtocol,
    tgt_tokenizer: TokenizerProtocol,
    device: torch.device,
    attention: str | None = None,
) -> Seq2Seq:
    tgt_sos_idx = tgt_tokenizer.bos_token_id
    if tgt_sos_idx is None:
        tgt_sos_idx = getattr(tgt_tokenizer, "sos_token_id", None)
    if tgt_sos_idx is None:
        raise ValueError(
            "Tokenizer hat keinen Start-Token (bos_token_id/sos_token_id)."
        )
    src_pad_idx = src_tokenizer.pad_token_id
    tgt_pad_idx = tgt_tokenizer.pad_token_id
    if src_pad_idx is None or tgt_pad_idx is None:
        raise ValueError("Tokenizer hat keinen pad_token_id.")

    attention_name = (
        attention if attention is not None else str(getattr(args, "attention", "torch"))
    )
    model = Seq2Seq(
        src_vocab_size=src_tokenizer.vocab_size,
        tgt_vocab_size=tgt_tokenizer.vocab_size,
        d_model=args.emb_dim,
        ff_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        src_pad_idx=src_pad_idx,
        tgt_pad_idx=tgt_pad_idx,
        tgt_sos_idx=tgt_sos_idx,
        dropout=args.dropout,
        attention=attention_name,
    ).to(device)
    return model


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    def create_data_loader(
        pairs, src_tokenizer, tgt_tokenizer, src_pad_idx: int, tgt_pad_idx: int
    ):
        dataset = TranslationDataset(pairs, src_tokenizer, tgt_tokenizer)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, src_pad_idx, tgt_pad_idx),
        )
        return loader

    def create_tokenizer(texts):
        return data_factory.create_tokenizer(
            str(getattr(args, "tokenizer", "custom")),
            texts,
            getattr(args, "hf_tokenizer_name", "bert-base-multilingual-cased"),
        )

    def print_sample_translations(model, pairs, src_tokenizer, tgt_tokenizer):
        model.eval()
        print("\nBeispiele:")
        for src_text, _ in pairs[:5]:
            src_ids = src_tokenizer.encode(src_text)
            pred_ids = model.translate(
                src_ids, max_len=20, device=device, eos_idx=tgt_tokenizer.eos_token_id
            )
            print(f"{src_text:20s} -> {tgt_tokenizer.decode(pred_ids)}")

    pairs = tiny_parallel_corpus()
    src_tokenizer = create_tokenizer([p[0] for p in pairs])
    tgt_tokenizer = create_tokenizer([p[1] for p in pairs])
    src_pad_idx = src_tokenizer.pad_token_id
    tgt_pad_idx = tgt_tokenizer.pad_token_id
    if src_pad_idx is None or tgt_pad_idx is None:
        raise ValueError("Tokenizer hat keinen pad_token_id.")

    loader = create_data_loader(
        pairs, src_tokenizer, tgt_tokenizer, src_pad_idx, tgt_pad_idx
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        args,
        src_tokenizer,
        tgt_tokenizer,
        device,
        attention=args.attention,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for src, tgt in loader:
            src = src.to(device)
            tgt = tgt.to(device)

            optim.zero_grad()
            logits = model(src, tgt)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt[:, 1:].reshape(-1),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += float(loss.item())

        print(f"Epoch {epoch:03d} | loss={total_loss / len(loader):.4f}")

    print_sample_translations(model, pairs, src_tokenizer, tgt_tokenizer)

    save_checkpoint(args.checkpoint_path, model, src_tokenizer, tgt_tokenizer, args)


def run_translate(args: argparse.Namespace, interactive: bool) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, src_tokenizer, tgt_tokenizer = load_inference_components(
        args.checkpoint_path, device, args
    )
    tgt_eos_idx = tgt_tokenizer.eos_token_id
    if tgt_eos_idx is None:
        raise ValueError("Tokenizer hat keinen eos_token_id.")

    def translate_text(text: str) -> str:
        src_ids = src_tokenizer.encode(text)
        pred_ids = model.translate(
            src_ids, max_len=args.max_len, device=device, eos_idx=tgt_eos_idx
        )
        return tgt_tokenizer.decode(pred_ids)

    if interactive:
        print("Interaktiver Modus. Mit 'exit' beenden.")
        while True:
            try:
                text = input("de> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text:
                continue
            if text.lower() == "exit":
                break
            print(f"en> {translate_text(text)}")
        return

    print(translate_text(args.translate))


def main() -> None:
    def parse_args() -> argparse.Namespace:
        p = argparse.ArgumentParser(
            description="Minimaler Transformer-Translator (Pre-Norm)"
        )
        p.add_argument("--epochs", type=int, default=200)
        p.add_argument("--batch-size", type=int, default=4)
        p.add_argument("--emb-dim", type=int, default=64)
        p.add_argument("--hidden-dim", type=int, default=64)
        p.add_argument("--lr", type=float, default=1e-3)
        p.add_argument("--num-heads", type=int, default=4)
        p.add_argument("--num-layers", type=int, default=2)
        p.add_argument("--dropout", type=float, default=0.1)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument(
            "--checkpoint-path", type=str, default="checkpoints/translator.pt"
        )
        p.add_argument(
            "--attention", type=str, choices=ATTENTION_CHOICES, default="torch"
        )
        p.add_argument(
            "--tokenizer", type=str, choices=TOKENIZER_CHOICES, default="custom"
        )
        p.add_argument(
            "--hf-tokenizer-name", type=str, default="bert-base-multilingual-cased"
        )
        p.add_argument("--translate", type=str, default=None)
        p.add_argument("--interactive", action="store_true")
        p.add_argument("--max-len", type=int, default=30)
        return p.parse_args()

    args = parse_args()
    if args.translate and args.interactive:
        raise ValueError("Nutze entweder --translate oder --interactive, nicht beides.")

    if args.translate or args.interactive:
        run_translate(args, interactive=args.interactive)
        return

    train(args)
