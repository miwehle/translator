from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from tests.translator.training.support import create_valid_mapped_dataset, train_config_for_test
from translator.training import Example, ModelConfig, Trainer, check_dataset
from translator.training.dataset import DatasetMetadata, load_arrow_records
from translator.training.internal.factory import Factory


def _create_factory(ds: Iterable[Example], *, configured_max_seq_len: int | None = None) -> Factory:
    check_result = check_dataset(ds)
    return Factory(
        dataset_metadata=DatasetMetadata(
            schema_version=1,
            src_vocab_size=check_result["src_vocab_size"],
            tgt_vocab_size=check_result["tgt_vocab_size"],
            src_pad_id=check_result["src_pad_idx"],
            tgt_pad_id=check_result["tgt_pad_idx"],
            tgt_bos_id=check_result["tgt_sos_idx"],
            tgt_eos_id=0,
            num_examples=int(check_result["num_examples"]),
            tokenizer_model_name="test-tokenizer",
            src_lang="de",
            tgt_lang="en",
            base_vocab_size=0,
            id_field=check_result["id_field"],
            src_field=check_result["src_field"],
            tgt_field=check_result["tgt_field"],
            configured_max_seq_len=configured_max_seq_len,
        )
    )


def _assert_init_handles_configured_seq_len_above_model_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path = create_valid_mapped_dataset(tmp_path / "valid_training.mapped")
    ds = cast(Iterable[Example], load_arrow_records(dataset_path))
    factory = _create_factory(ds, configured_max_seq_len=5)
    model_config = ModelConfig(max_seq_len=4)
    warning_messages: list[str] = []

    with pytest.raises(ValueError, match="configured_max_seq_len"):
        Trainer(factory, train_config_for_test(str(tmp_path), device="cpu"), model_config=model_config)

    monkeypatch.setattr(
        "translator.training.trainer.logger.warning",
        lambda message, *args: warning_messages.append(message % args),
    )
    Trainer(
        factory, train_config_for_test(str(tmp_path), device="cpu", force=True), model_config=model_config
    )

    assert any("configured_max_seq_len" in message for message in warning_messages)


def _assert_resume_rejects_configured_seq_len_above_model_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path = create_valid_mapped_dataset(tmp_path / "valid_training.mapped")
    ds = cast(Iterable[Example], load_arrow_records(dataset_path))
    factory = _create_factory(ds, configured_max_seq_len=5)
    monkeypatch.setattr(
        "translator.training.trainer.load_checkpoint",
        lambda checkpoint_path, factory, device: SimpleNamespace(
            model=object(), optimizer=object(), model_config=ModelConfig(max_seq_len=4)
        ),
    )

    with pytest.raises(ValueError, match="configured_max_seq_len"):
        Trainer(factory, train_config_for_test(str(tmp_path), device="cpu"), resume_run="first_run")


class TestTrainer:
    def test_init_handles_configured_seq_len_above_model_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _assert_init_handles_configured_seq_len_above_model_limit(tmp_path, monkeypatch)

    def test_train_resumes_from_checkpoint(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _assert_resume_rejects_configured_seq_len_above_model_limit(tmp_path, monkeypatch)

    def test_train_runs_periodic_evaluation(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        dataset_path = create_valid_mapped_dataset(tmp_path / "valid_training.mapped")
        ds = cast(Iterable[Example], load_arrow_records(dataset_path))
        factory = _create_factory(ds)
        train_config = train_config_for_test(
            str(tmp_path), device="cpu", epochs=1, log_every=1000, eval_every=4, seed=7
        )
        trainer = Trainer(factory, train_config, model_config=ModelConfig(d_model=32, ff_dim=64, num_heads=4, num_layers=2))
        evaluate_call_count = 0
        observed_eval_steps: list[int] = []

        def fake_evaluate(examples: Iterable[Example]) -> float:
            nonlocal evaluate_call_count
            evaluate_call_count += 1
            return 0.25

        monkeypatch.setattr(trainer, "evaluate", fake_evaluate)
        monkeypatch.setattr(
            "translator.training.internal.training_observer.TrainingObserver.on_evaluation",
            lambda self, step, epoch, eval_loss: observed_eval_steps.append(step),
        )

        summary = trainer.train(ds, ds)

        assert summary.num_examples == len(ds)
        assert evaluate_call_count == 2
        assert observed_eval_steps == [4, 8]

    def test_evaluate_restores_training_mode(self, tmp_path: Path) -> None:
        dataset_path = create_valid_mapped_dataset(tmp_path / "valid_training.mapped")
        ds = cast(Iterable[Example], load_arrow_records(dataset_path))
        trainer = Trainer(
            _create_factory(ds),
            train_config_for_test(str(tmp_path), device="cpu", seed=7),
            model_config=ModelConfig(d_model=32, ff_dim=64, num_heads=4, num_layers=2),
        )

        trainer._model.train()
        trainer.evaluate(ds)

        assert trainer._model.training is True
