from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import cast

from tests.translator.train_prod.support import (
    create_valid_mapped_dataset,
    train_config_for_test,
)
from translator.data_prod import load_arrow_records
from translator.train_prod import Example, Trainer, check_dataset
from translator.train_prod.factory import Factory
from translator.train_prod.training import DataLoaderConfig, ModelConfig


def test_trainer_writes_checkpoint_and_summary(tmp_path: Path) -> None:
    dataset_path = create_valid_mapped_dataset(tmp_path / "valid_train_prod.mapped")
    ds = cast(Iterable[Example], load_arrow_records(dataset_path))
    run_dir = tmp_path / "test_run_root" / "artifacts_run"
    checkpoint_path = run_dir / "checkpoint.pt"
    log_path = run_dir / "training.log"

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
                "id_field": check_result["id_field"],
                "src_field": check_result["src_field"],
                "tgt_field": check_result["tgt_field"],
            },
        )()
    )
    out = Trainer(factory).train(
        ds,
        train_config=train_config_for_test(
            str(tmp_path / "test_run_root"),
            seed=7,
            device="cpu",
            lr=1e-3,
            epochs=1,
            log_every=1000,
            run_name="artifacts_run",
        ),
        model_config=ModelConfig(
            emb_dim=32,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            dropout=0.0,
        ),
        data_loader_config=DataLoaderConfig(
            batch_size=32,
            shuffle=False,
        ),
    )

    assert checkpoint_path.is_file()
    assert log_path.is_file()
    assert out["checkpoint_path"] == str(checkpoint_path)
