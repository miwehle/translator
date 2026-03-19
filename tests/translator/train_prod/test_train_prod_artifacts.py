from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import cast

from tests.translator.train_prod.support import create_valid_mapped_dataset
from translator.data_prod import load_arrow_records
from translator.train_prod import Example, Trainer, TrainerConfig, build_model, check_dataset


def test_trainer_writes_checkpoint_and_summary(tmp_path: Path) -> None:
    dataset_path = create_valid_mapped_dataset(tmp_path / "valid_train_prod.mapped")
    ds = cast(Iterable[Example], load_arrow_records(dataset_path))
    checkpoint_path = tmp_path / "translator_train_prod.pt"
    summary_path = tmp_path / "translator_train_prod.json"

    check_result = check_dataset(ds)
    model = build_model(
        src_vocab_size=check_result["src_vocab_size"],
        tgt_vocab_size=check_result["tgt_vocab_size"],
        src_pad_idx=check_result["src_pad_idx"],
        tgt_pad_idx=check_result["tgt_pad_idx"],
        tgt_sos_idx=check_result["tgt_sos_idx"],
        emb_dim=32,
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
        device="cpu",
        seed=7,
    )

    trainer_config = TrainerConfig(
        id_field=check_result["id_field"],
        src_field=check_result["src_field"],
        tgt_field=check_result["tgt_field"],
        batch_size=32,
        seed=7,
        shuffle=False,
        device="cpu",
    )
    out = Trainer(model, trainer_config).train(
        ds,
        lr=1e-3,
        epochs=1,
        log_every=1000,
        checkpoint_path=checkpoint_path,
        summary_path=summary_path,
    )

    assert checkpoint_path.is_file()
    assert summary_path.is_file()
    assert out["checkpoint_path"] == str(checkpoint_path)
    assert out["summary_path"] == str(summary_path)
