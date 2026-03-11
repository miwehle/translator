import argparse
from pathlib import Path
from uuid import uuid4

import torch

from translator.data import Tokenizer, tiny_parallel_corpus
from translator.model import Seq2Seq
from translator.train import (
    build_model,
    load_checkpoint,
    load_inference_components,
    run_translate,
    save_checkpoint,
    train,
)


def make_args(checkpoint_path: Path, epochs: int = 1) -> argparse.Namespace:
    return argparse.Namespace(
        epochs=epochs,
        batch_size=4,
        emb_dim=32,
        hidden_dim=64,
        lr=1e-3,
        num_heads=4,
        num_layers=1,
        dropout=0.1,
        seed=42,
        checkpoint_path=str(checkpoint_path),
        attention="torch",
        translate=None,
        interactive=False,
        max_len=20,
    )


def make_vocabs():
    pairs = tiny_parallel_corpus()
    src_tokenizer = Tokenizer.build([p[0] for p in pairs])
    tgt_tokenizer = Tokenizer.build([p[1] for p in pairs])
    return src_tokenizer, tgt_tokenizer


def make_ckpt_path() -> Path:
    artifacts = Path("tests/.test_artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)
    return artifacts / f"{uuid4().hex}.pt"


def test_build_model_returns_seq2seq():
    src_tokenizer, tgt_tokenizer = make_vocabs()
    args = make_args(Path("dummy.pt"))
    device = torch.device("cpu")

    model = build_model(args, src_tokenizer, tgt_tokenizer, device)

    assert isinstance(model, Seq2Seq)


def test_save_and_load_checkpoint_roundtrip():
    src_tokenizer, tgt_tokenizer = make_vocabs()
    ckpt_path = make_ckpt_path()
    args = make_args(ckpt_path)
    model = build_model(args, src_tokenizer, tgt_tokenizer, torch.device("cpu"))

    save_checkpoint(str(ckpt_path), model, src_tokenizer, tgt_tokenizer, args)
    ckpt = load_checkpoint(str(ckpt_path), torch.device("cpu"))

    assert ckpt_path.exists()
    assert "model_state_dict" in ckpt
    assert "src_tokenizer" in ckpt
    assert "tgt_tokenizer" in ckpt
    assert ckpt["src_tokenizer"]["provider"] == "custom"
    assert ckpt["tgt_tokenizer"]["provider"] == "custom"
    assert ckpt["hparams"]["num_heads"] == 4
    assert ckpt["hparams"]["num_layers"] == 1


def test_train_writes_checkpoint():
    ckpt_path = make_ckpt_path()
    args = make_args(ckpt_path, epochs=1)

    train(args)

    assert ckpt_path.exists()
    assert ckpt_path.stat().st_size > 0


def test_run_translate_prints_output(monkeypatch, capsys):
    src_tokenizer, tgt_tokenizer = make_vocabs()
    ckpt_path = make_ckpt_path()
    args = make_args(ckpt_path)
    model = build_model(args, src_tokenizer, tgt_tokenizer, torch.device("cpu"))
    save_checkpoint(str(ckpt_path), model, src_tokenizer, tgt_tokenizer, args)

    def fake_translate(self, src_ids, max_len, device, eos_idx):
        return [self.tgt_sos_idx, self.tgt_sos_idx + 3, eos_idx]

    monkeypatch.setattr(Seq2Seq, "translate", fake_translate)

    infer_args = make_args(ckpt_path)
    infer_args.translate = "ich bin muede"
    run_translate(infer_args, interactive=False)
    out = capsys.readouterr().out.strip()

    assert out != ""


def test_run_translate_interactive_exit(monkeypatch, capsys):
    src_tokenizer, tgt_tokenizer = make_vocabs()
    ckpt_path = make_ckpt_path()
    args = make_args(ckpt_path)
    model = build_model(args, src_tokenizer, tgt_tokenizer, torch.device("cpu"))
    save_checkpoint(str(ckpt_path), model, src_tokenizer, tgt_tokenizer, args)

    inputs = iter(["exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    run_translate(args, interactive=True)
    out = capsys.readouterr().out

    assert "Interaktiver Modus" in out


def test_run_translate_raises_on_attention_mismatch():
    src_tokenizer, tgt_tokenizer = make_vocabs()
    ckpt_path = make_ckpt_path()
    args = make_args(ckpt_path)
    model = build_model(args, src_tokenizer, tgt_tokenizer, torch.device("cpu"))
    save_checkpoint(str(ckpt_path), model, src_tokenizer, tgt_tokenizer, args)

    infer_args = make_args(ckpt_path)
    infer_args.translate = "ich bin muede"
    infer_args.attention = "simple_sdp"

    try:
        run_translate(infer_args, interactive=False)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "Attention-Mismatch" in str(exc)


def test_load_inference_components_returns_model_and_tokenizers():
    src_tokenizer, tgt_tokenizer = make_vocabs()
    ckpt_path = make_ckpt_path()
    args = make_args(ckpt_path)
    model = build_model(args, src_tokenizer, tgt_tokenizer, torch.device("cpu"))
    save_checkpoint(str(ckpt_path), model, src_tokenizer, tgt_tokenizer, args)

    loaded_model, loaded_src_tokenizer, loaded_tgt_tokenizer = (
        load_inference_components(str(ckpt_path), torch.device("cpu"), args)
    )

    assert isinstance(loaded_model, Seq2Seq)
    assert isinstance(loaded_src_tokenizer, Tokenizer)
    assert isinstance(loaded_tgt_tokenizer, Tokenizer)
