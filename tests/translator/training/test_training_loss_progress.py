from __future__ import annotations

import platform
import re
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import cast

import pytest

from tests.translator.training.support import (
    create_valid_mapped_dataset,
    log,
    pad_index_from_records,
    train_config_for_test,
)
from translator.training import DataLoaderConfig, Example, ModelConfig, Trainer, check_dataset
from translator.training.dataset import load_arrow_records
from translator.training.internal.factory import Factory

STEP_LINE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\s+(?:TRAIN|SPIKE)\s+\S+\s+"
    r"(?P<step>\d+)\s+(?P<epoch>\d+)\s+(?P<loss>\d+(?:\.\d+)?)\b"
)


def _parse_losses(stdout_text: str) -> list[float]:
    losses: list[float] = []
    for line in stdout_text.splitlines():
        match = STEP_LINE_RE.match(line)
        if match is None:
            continue
        losses.append(float(match.group("loss")))
    return losses


def _build_training_log_body(
    *, dataset_path: Path, train_kwargs: Mapping[str, object], summary: Mapping[str, object], stdout_text: str
) -> str:
    step_rows: list[tuple[int, int, float]] = []
    for line in stdout_text.splitlines():
        match = STEP_LINE_RE.match(line.strip())
        if match is None:
            continue
        step_rows.append((int(match.group("step")), int(match.group("epoch")), float(match.group("loss"))))

    lines = [f"dataset_path={dataset_path}", "train_kwargs:"]
    for key in sorted(train_kwargs):
        lines.append(f"  {key}={train_kwargs[key]}")
    lines.append("summary:")
    for key in sorted(summary):
        lines.append(f"  {key}={summary[key]}")
    lines.append("")
    lines.append("loss_curve (step, epoch, loss):")
    for step, epoch, loss in step_rows:
        lines.append(f"{step},{epoch},{loss:.6f}")
    return "\n".join(lines) + "\n"


def test_trainer_loss_trend_decreases_on_synthetic_smoke_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset_path = create_valid_mapped_dataset(tmp_path / "valid_training.mapped")
    ds = cast(Iterable[Example], load_arrow_records(dataset_path))
    src_pad_idx = pad_index_from_records(dataset_path, "src_ids")
    tgt_pad_idx = pad_index_from_records(dataset_path, "tgt_ids")

    train_kwargs: dict[str, object] = {"dataset_path": dataset_path, "device": "cpu"}
    check_result = check_dataset(ds, src_pad_idx=src_pad_idx, tgt_pad_idx=tgt_pad_idx)
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
    train_config = train_config_for_test(
        str(tmp_path), run_name="synthetic_run", device="cpu", seed=7, lr=1e-3, epochs=2, log_every=1
    )
    t0 = time.perf_counter()
    out = Trainer(
        factory,
        train_config,
        DataLoaderConfig(batch_size=32, shuffle=False),
        model_config=ModelConfig(d_model=64, ff_dim=128, num_heads=4, num_layers=2, dropout=0.0),
    ).train(ds)
    training_duration_seconds = time.perf_counter() - t0

    assert out.num_examples > 0

    captured = capsys.readouterr()
    losses = _parse_losses(captured.out)
    assert len(losses) >= 6, "Not enough logged training steps to assess loss trend."

    window = 3
    first_avg = sum(losses[:window]) / window
    last_avg = sum(losses[-window:]) / window

    log(
        module_file=Path(__file__),
        test_name="test_trainer_loss_trend_decreases_on_synthetic_smoke_dataset",
        body=_build_training_log_body(
            dataset_path=dataset_path,
            train_kwargs=train_kwargs,
            summary={
                "num_examples": out.num_examples,
                "final_loss": out.final_loss,
                "first_avg": first_avg,
                "last_avg": last_avg,
                "num_logged_steps": len(losses),
                "training_duration_seconds": round(training_duration_seconds, 3),
                "run_host": platform.node(),
                "run_platform": platform.platform(),
            },
            stdout_text=captured.out,
        ),
    )

    assert last_avg < 0.4 * first_avg, (
        f"Expected the trailing average loss on the real preprocessed dataset "
        f"to drop below 40% of the initial average, "
        f"got first_avg={first_avg:.4f} last_avg={last_avg:.4f}"
    )


# @pytest.mark.slow
def test_trainer_loss_trend_decreases_on_real_preprocessed_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset_path = Path(__file__).resolve().parents[2] / "testdata" / "europarl_de-en_train_300"
    if not dataset_path.is_dir():
        pytest.skip(f"Missing test dataset directory: {dataset_path}")

    ds = cast(Iterable[Example], load_arrow_records(dataset_path))
    src_pad_idx = pad_index_from_records(dataset_path, "src_ids")
    tgt_pad_idx = pad_index_from_records(dataset_path, "tgt_ids")

    train_kwargs: dict[str, object] = {"dataset_path": dataset_path, "device": "cpu"}
    check_result = check_dataset(ds, src_pad_idx=src_pad_idx, tgt_pad_idx=tgt_pad_idx)
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
    train_config = train_config_for_test(
        str(tmp_path), run_name="real_run", device="cpu", seed=7, lr=1e-3, epochs=4, log_every=1
    )
    t0 = time.perf_counter()
    out = Trainer(
        factory,
        train_config,
        DataLoaderConfig(batch_size=32, shuffle=False),
        model_config=ModelConfig(d_model=64, ff_dim=128, num_heads=4, num_layers=2, dropout=0.0),
    ).train(ds)
    training_duration_seconds = time.perf_counter() - t0

    assert out.num_examples > 0

    captured = capsys.readouterr()
    losses = _parse_losses(captured.out)
    assert len(losses) >= 6, "Not enough logged training steps to assess loss trend."

    window = 3
    first_avg = sum(losses[:window]) / window
    last_avg = sum(losses[-window:]) / window

    log(
        module_file=Path(__file__),
        test_name="test_trainer_loss_trend_decreases_on_real_preprocessed_dataset",
        body=_build_training_log_body(
            dataset_path=dataset_path,
            train_kwargs=train_kwargs,
            summary={
                "num_examples": out.num_examples,
                "final_loss": out.final_loss,
                "first_avg": first_avg,
                "last_avg": last_avg,
                "num_logged_steps": len(losses),
                "training_duration_seconds": round(training_duration_seconds, 3),
                "run_host": platform.node(),
                "run_platform": platform.platform(),
            },
            stdout_text=captured.out,
        ),
    )

    assert last_avg < 0.8 * first_avg, (
        f"Expected the trailing average loss on the real preprocessed dataset "
        f"to drop below 80% of the initial average, "
        f"got first_avg={first_avg:.4f} last_avg={last_avg:.4f}"
    )
