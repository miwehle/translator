from __future__ import annotations

import platform
import re
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import cast

import pytest

from tests.translator.train_prod.support import (
    create_valid_mapped_dataset,
    log,
    pad_index_from_records,
)
from translator.data_prod import load_arrow_records
from translator.train_prod import (
    Example,
    Trainer,
    TrainerConfig,
    build_model,
    check_dataset,
)

LOSS_LINE_RE = re.compile(r"\bloss=(?P<loss>\d+(?:\.\d+)?)\b")
STEP_LINE_RE = re.compile(
    r"^(?:\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} )?"
    r"(?:SPIKE )?step=(?P<step>\d+)\s+ep=(?P<epoch>\d+)\s+"
    r"loss=(?P<loss>\d+(?:\.\d+)?)\b"
)


def _parse_losses(stdout_text: str) -> list[float]:
    losses: list[float] = []
    for line in stdout_text.splitlines():
        if STEP_LINE_RE.match(line) is None:
            continue
        match = LOSS_LINE_RE.search(line)
        if match is None:
            continue
        losses.append(float(match.group("loss")))
    return losses


def _build_training_log_body(
    *,
    dataset_path: Path,
    train_kwargs: Mapping[str, object],
    summary: Mapping[str, object],
    stdout_text: str,
) -> str:
    step_rows: list[tuple[int, int, float]] = []
    for line in stdout_text.splitlines():
        match = STEP_LINE_RE.match(line.strip())
        if match is None:
            continue
        step_rows.append(
            (
                int(match.group("step")),
                int(match.group("epoch")),
                float(match.group("loss")),
            )
        )

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
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_path = create_valid_mapped_dataset(tmp_path / "valid_train_prod.mapped")
    ds = cast(Iterable[Example], load_arrow_records(dataset_path))
    src_pad_idx = pad_index_from_records(dataset_path, "src_ids")
    tgt_pad_idx = pad_index_from_records(dataset_path, "tgt_ids")

    train_kwargs: dict[str, object] = {
        "dataset_path": dataset_path,
        "max_examples": 512,
        "device": "cpu",
    }
    check_result = check_dataset(
        ds,
        src_pad_idx=src_pad_idx,
        tgt_pad_idx=tgt_pad_idx,
        max_examples=512,
    )
    model = build_model(
        src_vocab_size=check_result["src_vocab_size"],
        tgt_vocab_size=check_result["tgt_vocab_size"],
        src_pad_idx=check_result["src_pad_idx"],
        tgt_pad_idx=check_result["tgt_pad_idx"],
        tgt_sos_idx=check_result["tgt_sos_idx"],
        emb_dim=64,
        hidden_dim=128,
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
        shuffle=False,
        max_examples=check_result["max_examples"],
        device="cpu",
        seed=7,
    )
    t0 = time.perf_counter()
    out = Trainer(model, trainer_config).train(
        ds,
        lr=1e-3,
        epochs=2,
        log_every=1,
    )
    training_duration_seconds = time.perf_counter() - t0

    assert out["global_step"] > 0

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
                "global_step": out["global_step"],
                "final_loss": out["final_loss"],
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


@pytest.mark.slow
def test_trainer_loss_trend_decreases_on_real_preprocessed_dataset(
    capsys: pytest.CaptureFixture[str],
) -> None:    
    dataset_path = (
        Path(__file__).resolve().parents[2]
        / "testdata"
        / "europarl_de-en_train_10000"
    )
    if not dataset_path.is_dir():
        pytest.skip(f"Missing test dataset directory: {dataset_path}")

    ds = cast(Iterable[Example], load_arrow_records(dataset_path))
    src_pad_idx = pad_index_from_records(dataset_path, "src_ids")
    tgt_pad_idx = pad_index_from_records(dataset_path, "tgt_ids")

    train_kwargs: dict[str, object] = {
        "dataset_path": dataset_path,
        "max_examples": 256,
        "device": "cpu",
    }
    check_result = check_dataset(
        ds,
        src_pad_idx=src_pad_idx,
        tgt_pad_idx=tgt_pad_idx,
        max_examples=256,
    )
    model = build_model(
        src_vocab_size=check_result["src_vocab_size"],
        tgt_vocab_size=check_result["tgt_vocab_size"],
        src_pad_idx=check_result["src_pad_idx"],
        tgt_pad_idx=check_result["tgt_pad_idx"],
        tgt_sos_idx=check_result["tgt_sos_idx"],
        emb_dim=64,
        hidden_dim=128,
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
        shuffle=False,
        max_examples=check_result["max_examples"],
        device="cpu",
        seed=7,
    )
    t0 = time.perf_counter()
    out = Trainer(model, trainer_config).train(
        ds,
        lr=1e-3,
        epochs=4,
        log_every=1,
    )
    training_duration_seconds = time.perf_counter() - t0

    assert out["global_step"] > 0

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
                "global_step": out["global_step"],
                "final_loss": out["final_loss"],
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
