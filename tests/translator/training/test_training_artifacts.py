from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import cast

import pytest

from tests.translator.training.support import create_valid_mapped_dataset, train_config_for_test
from translator.training import DataLoaderConfig, Example, ModelConfig, Trainer, check_dataset
from translator.training.dataset import load_arrow_records
from translator.training.internal.factory import Factory


def test_trainer_writes_checkpoint_and_summary(tmp_path: Path) -> None:
    dataset_path = create_valid_mapped_dataset(tmp_path / "valid_training.mapped")
    ds = cast(Iterable[Example], load_arrow_records(dataset_path))
    run_dir = tmp_path / "test_run_root" / "training_runs" / "artifacts_run"
    checkpoint_path = run_dir / "checkpoint.pt"
    checkpoint_manifest_path = run_dir / "checkpoint_manifest.yaml"
    log_path = run_dir / "training.log"
    translation_examples_path = run_dir / "translation_examples.txt"
    train_config = train_config_for_test(
        str(tmp_path / "test_run_root"),
        seed=7,
        device="cpu",
        lr=1e-3,
        epochs=1,
        log_every=1000,
        run_name="artifacts_run",
    )

    check_result = check_dataset(ds)
    factory = Factory(
        dataset_metadata=type(
            "_DatasetMetadata",
            (),
            {
                "src_vocab_size": check_result["src_vocab_size"],
                "tgt_vocab_size": check_result["tgt_vocab_size"],
                "src_pad_id": check_result["src_pad_idx"],
                "tgt_pad_id": check_result["tgt_pad_idx"],
                "tgt_bos_id": check_result["tgt_sos_idx"],
                "tokenizer_model_name": "test-tokenizer",
                "id_field": check_result["id_field"],
                "src_field": check_result["src_field"],
                "tgt_field": check_result["tgt_field"],
            },
        )()
    )
    out = Trainer(
        factory,
        train_config,
        DataLoaderConfig(batch_size=32, shuffle=False),
        model_config=ModelConfig(d_model=32, ff_dim=64, num_heads=4, num_layers=2, dropout=0.0),
    ).train(ds)

    assert checkpoint_path.is_file()
    assert checkpoint_manifest_path.is_file()
    assert log_path.is_file()
    assert not translation_examples_path.exists()
    assert out.checkpoint_path == str(checkpoint_path)


def test_trainer_writes_translation_examples_to_separate_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path = create_valid_mapped_dataset(tmp_path / "valid_training.mapped")
    ds = cast(Iterable[Example], load_arrow_records(dataset_path))
    run_dir = tmp_path / "test_run_root" / "training_runs" / "artifacts_run"
    log_path = run_dir / "training.log"
    translation_examples_path = run_dir / "translation_examples.txt"
    train_config = train_config_for_test(
        str(tmp_path / "test_run_root"),
        seed=7,
        device="cpu",
        lr=1e-3,
        epochs=1,
        log_every=1000,
        run_name="artifacts_run",
        translate_every=1000,
        translate_examples=("Hallo Welt.", "Ich mag Kaffee."),
    )

    check_result = check_dataset(ds)
    factory = Factory(
        dataset_metadata=type(
            "_DatasetMetadata",
            (),
            {
                "src_vocab_size": check_result["src_vocab_size"],
                "tgt_vocab_size": check_result["tgt_vocab_size"],
                "src_pad_id": check_result["src_pad_idx"],
                "tgt_pad_id": check_result["tgt_pad_idx"],
                "tgt_bos_id": check_result["tgt_sos_idx"],
                "tokenizer_model_name": "test-tokenizer",
                "id_field": check_result["id_field"],
                "src_field": check_result["src_field"],
                "tgt_field": check_result["tgt_field"],
            },
        )()
    )
    monkeypatch.setattr("translator.training.trainer.create_tokenizer", lambda *args: object())
    monkeypatch.setattr(
        "translator.training.trainer.Translator",
        lambda *args: type(
            "_FakeTranslator", (), {"translate_many": lambda self, texts: ["Hello world.", "I like coffee."]}
        )(),
    )
    trainer = Trainer(
        factory,
        train_config,
        DataLoaderConfig(batch_size=32, shuffle=False),
        model_config=ModelConfig(d_model=32, ff_dim=64, num_heads=4, num_layers=2, dropout=0.0),
    )
    trainer.train(ds)

    translation_examples = translation_examples_path.read_text(encoding="utf-8")
    training_log = log_path.read_text(encoding="utf-8")

    assert translation_examples_path.is_file()
    assert (
        translation_examples == "step=1 ep=1 loss=3.4921\n"
        "---\n"
        "src: Hallo Welt.\n"
        "pred: Hello world.\n"
        "\n"
        "src: Ich mag Kaffee.\n"
        "pred: I like coffee.\n\n"
    )
    assert "src: Hallo Welt." not in training_log
    assert "src: Ich mag Kaffee." not in training_log
