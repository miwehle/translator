from __future__ import annotations

import csv
from pathlib import Path
from typing import cast

import pytest
import yaml
from lab_infrastructure.run_config import read_run_config

from tests.translator.training.support import create_valid_mapped_dataset, train_config_for_test
from translator.api import check_dataset, comet_score, train
from translator.evaluation import CometScoreConfig
from translator.training import DataLoaderConfig, ModelConfig

_MODEL_CONFIG = ModelConfig(d_model=32, ff_dim=64, num_heads=4, num_layers=2, dropout=0.0)


def _write_dataset_manifest(dataset_dir: Path, **overrides: object) -> None:
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
        **overrides,
    }
    (dataset_dir / "dataset_manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )


def test_train_avoids_run_dir_name_collisions(tmp_path: Path, monkeypatch) -> None:
    artifacts_dir = tmp_path / "artifacts"
    dataset_dir = create_valid_mapped_dataset(artifacts_dir / "datasets" / "dataset.mapped")
    validation_dir = create_valid_mapped_dataset(artifacts_dir / "datasets" / "validation.mapped")
    _write_dataset_manifest(dataset_dir)
    _write_dataset_manifest(validation_dir)

    run_root = artifacts_dir / "training_runs"
    existing_run_dir = run_root / "run1"
    existing_run_dir.mkdir(parents=True)
    (existing_run_dir / "checkpoint.pt").write_text("existing checkpoint", encoding="utf-8")

    monkeypatch.setattr("translator.api.git_head_commit", lambda _: "test-commit")

    summary = train(
        train_config_for_test(
            str(artifacts_dir), run_name="run1", device="cpu", epochs=1, log_every=1000, lr=1e-3, seed=7
        ),
        DataLoaderConfig(batch_size=32, shuffle=False),
        tmp_path,
        model_config=_MODEL_CONFIG,
    )

    new_run_dir = run_root / "run1 (1)"
    training_summary = yaml.safe_load(
        new_run_dir.joinpath("training_summary.yaml").read_text(encoding="utf-8")
    )
    with run_root.joinpath("checkpoint_register.csv").open("r", encoding="utf-8", newline="") as handle:
        register_rows = list(csv.DictReader(handle, delimiter=";"))

    assert existing_run_dir.joinpath("checkpoint.pt").read_text(encoding="utf-8") == ("existing checkpoint")
    assert new_run_dir.is_dir()
    assert new_run_dir.joinpath("training_config.yaml").is_file()
    assert new_run_dir.joinpath("training_summary.yaml").is_file()
    assert new_run_dir.joinpath("checkpoint.pt").is_file()
    assert new_run_dir.joinpath("checkpoint_manifest.yaml").is_file()
    assert new_run_dir.joinpath("training.log").is_file()
    assert Path(summary.checkpoint_path) == new_run_dir / "checkpoint.pt"
    assert summary.validation_loss is not None
    assert training_summary["validation_loss"] == summary.validation_loss
    assert register_rows[0]["validation_loss"] == str(summary.validation_loss).replace(".", ",")


def test_train_resumes_from_checkpoint(tmp_path: Path, monkeypatch) -> None:
    artifacts_dir = tmp_path / "artifacts"
    dataset_dir = create_valid_mapped_dataset(artifacts_dir / "datasets" / "dataset.mapped")
    validation_dir = create_valid_mapped_dataset(artifacts_dir / "datasets" / "validation.mapped")
    _write_dataset_manifest(dataset_dir)
    _write_dataset_manifest(validation_dir)
    run_root = artifacts_dir / "training_runs"

    monkeypatch.setattr("translator.api.git_head_commit", lambda _: "test-commit")

    train(
        train_config_for_test(
            str(artifacts_dir), run_name="run1", device="cpu", epochs=1, log_every=1000, lr=1e-3, seed=7
        ),
        DataLoaderConfig(batch_size=32, shuffle=False),
        tmp_path,
        model_config=_MODEL_CONFIG,
    )

    second_summary = train(
        train_config_for_test(
            str(artifacts_dir), run_name="run2", device="cpu", epochs=1, log_every=1000, lr=5e-4, seed=7
        ),
        DataLoaderConfig(batch_size=32, shuffle=False),
        tmp_path,
        resume_run="run1",
    )

    second_run_dir = run_root / "run2"
    manifest = yaml.safe_load(second_run_dir.joinpath("checkpoint_manifest.yaml").read_text(encoding="utf-8"))
    train_cfg = read_run_config(second_run_dir / "training_config.yaml")
    training_summary = yaml.safe_load(
        second_run_dir.joinpath("training_summary.yaml").read_text(encoding="utf-8")
    )
    with run_root.joinpath("checkpoint_register.csv").open("r", encoding="utf-8", newline="") as handle:
        register_rows = list(csv.DictReader(handle, delimiter=";"))

    assert Path(second_summary.checkpoint_path) == second_run_dir / "checkpoint.pt"
    assert manifest["optimizer"]["type"] == "adam"
    assert manifest["checkpoint_file"] == "checkpoint.pt"
    assert manifest["tokenizer"]["model_name"] == "test-tokenizer"
    assert "summary" not in manifest
    assert train_cfg["model_config"] is None
    assert train_cfg["resume_run"] == "run1"
    assert training_summary["checkpoint_path"] == second_summary.checkpoint_path
    assert training_summary["final_loss"] == second_summary.final_loss
    assert training_summary["validation_loss"] == second_summary.validation_loss
    assert len(register_rows) == 2
    assert register_rows[1]["input_ckpt"] == "run1"
    assert register_rows[1]["dataset_path"] == "dataset.mapped"
    assert register_rows[1]["git_commit"] == "test-commit"
    assert register_rows[1]["output_ckpt"] == "run2"
    assert register_rows[1]["validation_loss"] == str(second_summary.validation_loss).replace(".", ",")


def test_train_rejects_incompatible_validation_dataset(tmp_path: Path, monkeypatch) -> None:
    artifacts_dir = tmp_path / "artifacts"
    dataset_dir = create_valid_mapped_dataset(artifacts_dir / "datasets" / "dataset.mapped")
    validation_dir = create_valid_mapped_dataset(artifacts_dir / "datasets" / "validation.mapped")
    _write_dataset_manifest(dataset_dir)
    _write_dataset_manifest(validation_dir, src_pad_id=999)

    monkeypatch.setattr("translator.api.git_head_commit", lambda _: "test-commit")

    with pytest.raises(ValueError, match="Validation dataset metadata mismatch"):
        train(
            train_config_for_test(
                str(artifacts_dir), device="cpu", epochs=1, log_every=1000, lr=1e-3, seed=7
            ),
            DataLoaderConfig(batch_size=32, shuffle=False),
            tmp_path,
            model_config=_MODEL_CONFIG,
        )


def test_train_rejects_validate_every_without_validation_dataset(tmp_path: Path, monkeypatch) -> None:
    artifacts_dir = tmp_path / "artifacts"
    dataset_dir = create_valid_mapped_dataset(artifacts_dir / "datasets" / "dataset.mapped")
    _write_dataset_manifest(dataset_dir)

    monkeypatch.setattr("translator.api.git_head_commit", lambda _: "test-commit")

    with pytest.raises(ValueError, match="validate_every requires validation_dataset"):
        train(
            train_config_for_test(
                str(artifacts_dir),
                device="cpu",
                epochs=1,
                log_every=1000,
                lr=1e-3,
                seed=7,
                validation_dataset=None,
                validate_every=10,
            ),
            DataLoaderConfig(batch_size=32, shuffle=False),
            tmp_path,
            model_config=_MODEL_CONFIG,
        )


def test_check_dataset_uses_dataset_manifest_defaults(tmp_path: Path) -> None:
    dataset_dir = create_valid_mapped_dataset(tmp_path / "dataset.mapped")
    _write_dataset_manifest(dataset_dir)

    result = check_dataset(dataset_path=dataset_dir, require_unique_ids=True, min_seq_len=2)

    assert result["id_field"] == "id"
    assert result["src_field"] == "src_ids"
    assert result["tgt_field"] == "tgt_ids"
    assert result["src_pad_idx"] == 0
    assert result["tgt_pad_idx"] == 1


def test_comet_score_uses_convention_checkpoint_path(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}
    checkpoint_dir = tmp_path / "training_runs" / "ttc10-lr1"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    class _FakeScorer:
        def __init__(self, **kwargs: object) -> None:
            captured["kwargs"] = kwargs

        def score_checkpoint(self, checkpoint: str | Path) -> float:
            captured["checkpoint"] = checkpoint
            return 0.88

    monkeypatch.setattr("translator.evaluation.CometScorer", _FakeScorer)
    monkeypatch.setattr(CometScoreConfig, "checkpoint_file", property(lambda self: checkpoint_dir / "checkpoint.pt"))

    score = comet_score(
        CometScoreConfig(
            checkpoint="ttc10-lr1",
            dataset_config={"path": "IWSLT/iwslt2017", "name": "iwslt2017-de-en", "split": "validation"},
            mapping_config={"src": "translation.de", "ref": "translation.en"},
        )
    )

    assert score == 0.88
    expected_checkpoint = checkpoint_dir / "checkpoint.pt"
    assert Path(captured["checkpoint"]) == expected_checkpoint
    scorer_kwargs = cast(dict[str, object], captured["kwargs"])
    assert scorer_kwargs["comet_model"] == "Unbabel/wmt22-comet-da"
    assert scorer_kwargs["output_path"] is None
    assert scorer_kwargs["mapping"].src == "translation.de"
    assert scorer_kwargs["mapping"].ref == "translation.en"
    comet_score_summary = yaml.safe_load((checkpoint_dir / "comet_score.yaml").read_text(encoding="utf-8"))
    with checkpoint_dir.parent.joinpath("comet_score_register.csv").open("r", encoding="utf-8", newline="") as handle:
        register_rows = list(csv.DictReader(handle, delimiter=";"))

    assert comet_score_summary == {
        "score": 0.88,
        "config": {
            "checkpoint": "ttc10-lr1",
            "dataset_config": {
                "datasets_dir": "/content/drive/MyDrive/nmt_lab/artifacts/datasets",
                "path": "IWSLT/iwslt2017",
                "name": "iwslt2017-de-en",
                "split": "validation",
                "data_files": None,
            },
            "mapping_config": {"src": "translation.de", "ref": "translation.en"},
            "model": "Unbabel/wmt22-comet-da",
            "output_path": None,
        },
    }
    assert register_rows == [
        {
            "timestamp": register_rows[0]["timestamp"],
            "checkpoint": "ttc10-lr1",
            "eval_dataset": "IWSLT/iwslt2017",
            "comet_model": "Unbabel/wmt22-comet-da",
            "comet_score": "0,88",
        }
    ]
