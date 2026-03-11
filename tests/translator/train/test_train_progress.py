import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from translator.data import (
    Tokenizer,
    TranslationDataset,
    collate_fn,
    set_seed,
    tiny_parallel_corpus,
)
from translator.train import build_model


def make_args() -> argparse.Namespace:
    return argparse.Namespace(
        emb_dim=32,
        hidden_dim=64,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
        attention="torch",
        lr=1e-3,
        batch_size=4,
        seed=42,
    )


def build_training_objects():
    args = make_args()
    set_seed(args.seed)
    device = torch.device("cpu")

    pairs = tiny_parallel_corpus()
    src_tokenizer = Tokenizer.build([p[0] for p in pairs])
    tgt_tokenizer = Tokenizer.build([p[1] for p in pairs])

    dataset = TranslationDataset(pairs, src_tokenizer, tgt_tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(
            batch, src_tokenizer.pad_token_id, tgt_tokenizer.pad_token_id
        ),
    )

    model = build_model(args, src_tokenizer, tgt_tokenizer, device)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    return (
        model,
        loader,
        criterion,
        optimizer,
        src_tokenizer,
        tgt_tokenizer,
        device,
        pairs,
    )


def batch_loss(model, criterion, batch, device):
    src, tgt = batch
    src = src.to(device)
    tgt = tgt.to(device)
    logits = model(src, tgt)
    return criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))


def translation_match_count(
    model, pairs, src_tokenizer, tgt_tokenizer, device, n_samples=5
):
    count = 0
    for src_text, tgt_text in pairs[:n_samples]:
        pred_ids = model.translate(
            src_tokenizer.encode(src_text),
            max_len=20,
            device=device,
            eos_idx=tgt_tokenizer.eos_token_id,
        )
        if tgt_tokenizer.decode(pred_ids) == tgt_text:
            count += 1
    return count


def test_loss_decreases_over_updates():
    model, loader, criterion, optimizer, *_ = build_training_objects()
    first_batch = next(iter(loader))

    model.eval()
    with torch.no_grad():
        initial_loss = float(
            batch_loss(model, criterion, first_batch, torch.device("cpu")).item()
        )

    model.train()
    for _ in range(20):
        for batch in loader:
            optimizer.zero_grad()
            loss = batch_loss(model, criterion, batch, torch.device("cpu"))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        final_loss = float(
            batch_loss(model, criterion, first_batch, torch.device("cpu")).item()
        )

    min_abs_drop = 0.2
    assert (initial_loss - final_loss) >= min_abs_drop


def test_translation_quality_improves_after_training():
    model, loader, criterion, optimizer, src_tokenizer, tgt_tokenizer, device, pairs = (
        build_training_objects()
    )

    model.eval()
    before = translation_match_count(model, pairs, src_tokenizer, tgt_tokenizer, device)

    model.train()
    for _ in range(90):
        for batch in loader:
            optimizer.zero_grad()
            loss = batch_loss(model, criterion, batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    after = translation_match_count(model, pairs, src_tokenizer, tgt_tokenizer, device)

    assert (after - before) == 5
