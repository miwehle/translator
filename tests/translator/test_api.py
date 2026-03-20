from __future__ import annotations

from pathlib import Path

import yaml

from tests.translator.train_prod.support import create_valid_mapped_dataset, train_config_for_test
from translator.api import train
from translator.train_prod.training import DataLoaderConfig, ModelConfig


def _write_dataset_manifest(dataset_dir: Path) -> None:
    manifest = {
        "schema_version": 1,
        "tokenizer_model_name": "test-tokenizer",
        "src_lang": "de",
        "tgt_lang": "en",
        "id_field": "id",
        "src_field": "src_ids",
        "tgt_field": "tgt_ids",
        "base_vocab_size": 128,
        "src_vocab_size": 128,
        "tgt_vocab_size": 128,
        "src_pad_id": 0,
        "tgt_pad_id": 1,
        "tgt_bos_id": 2,
        "tgt_eos_id": 3,
        "num_examples": 512,
    }
    (dataset_dir / "dataset_manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )


def test_train_avoids_run_dir_name_collisions(tmp_path: Path, monkeypatch) -> None:
    dataset_dir = create_valid_mapped_dataset(tmp_path / "dataset.mapped")
    _write_dataset_manifest(dataset_dir)

    run_root = tmp_path / "training_runs"
    existing_run_dir = run_root / "run1"
    existing_run_dir.mkdir(parents=True)
    (existing_run_dir / "checkpoint.pt").write_text("existing checkpoint", encoding="utf-8")

    monkeypatch.setattr("translator.api._git_head", lambda _: "test-commit")

    summary = train(
        dataset_path=dataset_dir,
        train_config=train_config_for_test(
            str(run_root),
            run_name="run1",
            device="cpu",
            epochs=1,
            log_every=1000,
            lr=1e-3,
            seed=7,
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
        repo_root=tmp_path,
    )

    new_run_dir = run_root / "run1 (1)"

    assert existing_run_dir.joinpath("checkpoint.pt").read_text(encoding="utf-8") == (
        "existing checkpoint"
    )
    assert new_run_dir.is_dir()
    assert new_run_dir.joinpath("config.json").is_file()
    assert new_run_dir.joinpath("checkpoint.pt").is_file()
    assert new_run_dir.joinpath("summary.json").is_file()
    assert new_run_dir.joinpath("training.log").is_file()
    assert Path(summary["checkpoint_path"]) == new_run_dir / "checkpoint.pt"
