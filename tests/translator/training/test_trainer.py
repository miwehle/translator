from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from tests.translator.training.support import create_valid_mapped_dataset, train_config
from translator.training import Example, ModelConfig, Trainer, check_dataset
from translator.training.dataset import DatasetMetadata, load_arrow_records
from translator.training.internal.factory import Factory

_MODEL_CFG = ModelConfig(d_model=32, ff_dim=64, num_heads=4, num_layers=2)


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
        Trainer(factory, train_config(str(tmp_path), model_config=model_config))

    monkeypatch.setattr(
        "translator.training.trainer.logger.warning",
        lambda message, *args: warning_messages.append(message % args),
    )
    Trainer(factory, train_config(str(tmp_path), force=True, model_config=model_config))

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
        Trainer(factory, train_config(str(tmp_path), parent_checkpoint="first_run"))


class TestTrainer:
    def test_init_handles_configured_seq_len_above_model_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _assert_init_handles_configured_seq_len_above_model_limit(tmp_path, monkeypatch)

    def test_train_resumes_from_checkpoint(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _assert_resume_rejects_configured_seq_len_above_model_limit(tmp_path, monkeypatch)

    def test_init_rejects_validate_every_without_validation_dataset(self, tmp_path: Path) -> None:
        dataset_path = create_valid_mapped_dataset(tmp_path / "valid_training.mapped")
        ds = cast(Iterable[Example], load_arrow_records(dataset_path))
        cfg = train_config(str(tmp_path), validate_every=10, validation_dataset=None, model_config=_MODEL_CFG)

        with pytest.raises(ValueError, match="validate_every requires validation_dataset"):
            Trainer(_create_factory(ds), cfg)

    def test_train_runs_periodic_evaluation(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        dataset_path = create_valid_mapped_dataset(tmp_path / "valid_training.mapped")
        ds = cast(Iterable[Example], load_arrow_records(dataset_path))
        factory = _create_factory(ds)
        cfg = train_config(
            str(tmp_path), epochs=1, log_every=1000, validate_every=4, seed=7, model_config=_MODEL_CFG
        )
        trainer = Trainer(factory, cfg)
        evaluate_call_count = 0
        observed_eval_steps: list[int] = []

        def fake_validate(examples: Iterable[Example]) -> float:
            nonlocal evaluate_call_count
            evaluate_call_count += 1
            return 0.25

        monkeypatch.setattr(trainer, "validate", fake_validate)

        def fake_log(
            self,
            step: int,
            epoch: int,
            loss: float,
            median_loss: float | None,
            *,
            validation_loss: float | None = None,
            label: str | None = None,
            level: int = 20,
            grad_norm: float | None = None,
            lr: float | None = None,
            batch_ids: object | None = None,
        ) -> str:
            if validation_loss is not None:
                observed_eval_steps.append(step)
            return ""

        monkeypatch.setattr("translator.training.internal.logging.TrainingLogger.log", fake_log)

        summary = trainer.train(ds, ds)

        assert summary.num_examples == len(ds)
        assert evaluate_call_count == 2
        assert observed_eval_steps == [4, 8]

    def test_evaluate_restores_training_mode(self, tmp_path: Path) -> None:
        dataset_path = create_valid_mapped_dataset(tmp_path / "valid_training.mapped")
        ds = cast(Iterable[Example], load_arrow_records(dataset_path))
        trainer = Trainer(
            _create_factory(ds),
            train_config(str(tmp_path), seed=7, model_config=_MODEL_CFG),
        )

        trainer._model.train()
        trainer.validate(ds)

        assert trainer._model.training is True

    def test_train_rejects_missing_validation_examples_for_eval(self, tmp_path: Path) -> None:
        dataset_path = create_valid_mapped_dataset(tmp_path / "valid_training.mapped")
        ds = cast(Iterable[Example], load_arrow_records(dataset_path))
        trainer = Trainer(
            _create_factory(ds),
            train_config(str(tmp_path), validate_every=10, model_config=_MODEL_CFG),
        )

        with pytest.raises(ValueError, match="validate_every requires validation_examples"):
            trainer.train(ds)

    def test_autocast_context_uses_bf16_only_on_cuda_when_enabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dataset_path = create_valid_mapped_dataset(tmp_path / "valid_training.mapped")
        ds = cast(Iterable[Example], load_arrow_records(dataset_path))
        trainer = Trainer(
            _create_factory(ds),
            train_config(str(tmp_path), use_bf16=True, model_config=_MODEL_CFG),
        )
        trainer._device = torch.device("cuda")
        seen_kwargs: dict[str, object] = {}

        def fake_autocast(*, device_type: str, dtype: object):
            seen_kwargs.update(device_type=device_type, dtype=dtype)
            return nullcontext()

        monkeypatch.setattr("translator.training.trainer.torch.autocast", fake_autocast)

        with trainer._autocast_context():
            pass

        assert seen_kwargs == {"device_type": "cuda", "dtype": torch.bfloat16}
