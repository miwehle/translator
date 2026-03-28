from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import cast

import yaml

from tests.translator.training.support import (
    create_valid_mapped_dataset,
    train_config_for_test,
)
from translator.training import (
    DataLoaderConfig,
    Example,
    ModelConfig,
    Trainer,
    check_dataset,
)
from translator.training.dataset import load_arrow_records
from translator.training.factory import Factory


def _create_factory(ds: Iterable[Example]) -> Factory:
    check_result = check_dataset(ds)
    return Factory(
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


class TestTrainer:
    def test_train_resumes_from_checkpoint(self, tmp_path: Path) -> None:
        dataset_path = create_valid_mapped_dataset(tmp_path / "valid_training.mapped")
        ds = cast(Iterable[Example], load_arrow_records(dataset_path))
        factory = _create_factory(ds)
        train_root = tmp_path / "test_run_root"

        first_summary = Trainer(
            factory,
            train_config_for_test(
                str(train_root),
                seed=7,
                device="cpu",
                lr=1e-3,
                epochs=1,
                log_every=1000,
                run_name="first_run",
            ),
            model_config=ModelConfig(
                d_model=32,
                ff_dim=64,
                num_heads=4,
                num_layers=2,
                dropout=0.0,
            ),
        ).train(
            ds,
            DataLoaderConfig(
                batch_size=32,
                shuffle=False,
            ),
        )

        second_summary = Trainer(
            factory,
            train_config_for_test(
                str(train_root),
                seed=7,
                device="cpu",
                lr=5e-4,
                epochs=1,
                log_every=1000,
                run_name="second_run",
            ),
            checkpoint_path=first_summary.checkpoint_path,
        ).train(
            ds,
            DataLoaderConfig(
                batch_size=32,
                shuffle=False,
            ),
        )

        second_run_dir = train_root / "second_run"
        manifest = yaml.safe_load(
            second_run_dir.joinpath("checkpoint_manifest.yaml").read_text(encoding="utf-8")
        )

        assert Path(second_summary.checkpoint_path) == second_run_dir / "checkpoint.pt"
        assert manifest["optimizer"]["type"] == "adam"
        assert manifest["checkpoint_file"] == "checkpoint.pt"
