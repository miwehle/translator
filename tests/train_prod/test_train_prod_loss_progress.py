from __future__ import annotations

import os
import platform
import re
import time
from pathlib import Path
from typing import Mapping, cast

import pytest
from datasets import Dataset

from translator.data_prod.arrow_dataset import load_arrow_records
from translator.train_prod import train_prod

LOSS_LINE_RE = re.compile(r"\bloss=(?P<loss>\d+(?:\.\d+)?)\b")
STEP_LINE_RE = re.compile(r"^step=(?P<step>\d+)\s+epoch=(?P<epoch>\d+)\s+loss=(?P<loss>\d+(?:\.\d+)?)\b")


def _find_mapped_dataset() -> Path:
    configured = os.getenv("TRANSLATOR2_TESTDATA_MAPPED")
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if candidate.is_dir():
            return candidate
        pytest.skip(
            "TRANSLATOR2_TESTDATA_MAPPED is set but does not point to a directory: "
            f"{candidate}"
        )

    testdata_dir = Path(__file__).resolve().parents[1] / "testdata"
    if not testdata_dir.exists():
        pytest.skip(f"Missing test data directory: {testdata_dir}")

    candidates = sorted(p for p in testdata_dir.rglob("*.mapped") if p.is_dir())
    if not candidates:
        pytest.skip(
            "No mapped Arrow dataset found under tests/testdata. "
            "Expected a directory like '.../europarl.mapped'."
        )
    return candidates[0]


def _create_valid_mapped_dataset(dataset_dir: Path) -> Path:
    rows = {
        "id": list(range(512)),
        "src_ids": [[10, 100 + (i % 17), 11] for i in range(512)],
        "tgt_ids": [[2, 20 + (i % 9), 3] for i in range(512)],
    }
    Dataset.from_dict(rows).save_to_disk(str(dataset_dir))
    return dataset_dir


def _parse_losses(stdout_text: str) -> list[float]:
    losses: list[float] = []
    for line in stdout_text.splitlines():
        if not line.startswith("step="):
            continue
        match = LOSS_LINE_RE.search(line)
        if match is None:
            continue
        losses.append(float(match.group("loss")))
    return losses


def _parse_step_rows(stdout_text: str) -> list[tuple[int, int, float]]:
    rows: list[tuple[int, int, float]] = []
    for line in stdout_text.splitlines():
        match = STEP_LINE_RE.match(line.strip())
        if match is None:
            continue
        rows.append(
            (
                int(match.group("step")),
                int(match.group("epoch")),
                float(match.group("loss")),
            )
        )
    return rows


def _pad_index_from_records(dataset_path: Path, field: str) -> int:
    records = load_arrow_records(dataset_path)
    max_token = -1
    for row in records:
        row_map = cast(Mapping[str, object], row)
        values_obj = row_map.get(field)
        if not isinstance(values_obj, list):
            raise ValueError(f"Expected list[int] in field '{field}', got {type(values_obj).__name__}.")
        values = [int(x) for x in values_obj]
        max_token = max(max_token, max(values))
    return max_token + 1


def _write_training_log(
    *,
    dataset_path: Path,
    train_kwargs: Mapping[str, object],
    summary: Mapping[str, object],
    stdout_text: str,
) -> Path:
    testdata_dir = Path(__file__).resolve().parents[1] / "testdata"
    testdata_dir.mkdir(parents=True, exist_ok=True)
    log_path = testdata_dir / "train_prod_loss_progress.log"

    step_rows = _parse_step_rows(stdout_text)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("train_prod loss progress log\n")
        f.write(f"dataset_path={dataset_path}\n")
        f.write("train_kwargs:\n")
        for key in sorted(train_kwargs):
            f.write(f"  {key}={train_kwargs[key]}\n")
        f.write("summary:\n")
        for key in sorted(summary):
            f.write(f"  {key}={summary[key]}\n")
        f.write("\nloss_curve (step, epoch, loss):\n")
        for step, epoch, loss in step_rows:
            f.write(f"{step},{epoch},{loss:.6f}\n")

    return log_path


def test_train_prod_loss_trend_decreases(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_path = _create_valid_mapped_dataset(tmp_path / "valid_train_prod.mapped")
    src_pad_idx = _pad_index_from_records(dataset_path, "src_ids")
    tgt_pad_idx = _pad_index_from_records(dataset_path, "tgt_ids")

    train_kwargs: dict[str, object] = {
        "dataset_path": dataset_path,
        "src_pad_idx": src_pad_idx,
        "tgt_pad_idx": tgt_pad_idx,
        "emb_dim": 64,
        "hidden_dim": 128,
        "num_heads": 4,
        "num_layers": 2,
        "dropout": 0.0,
        "lr": 1e-3,
        "batch_size": 32,
        "epochs": 2,
        "seed": 7,
        "max_examples": 512,
        "shuffle": False,
        "log_every": 1,
        "device": "cpu",
    }
    t0 = time.perf_counter()
    out = train_prod(
        dataset_path=dataset_path,
        src_pad_idx=src_pad_idx,
        tgt_pad_idx=tgt_pad_idx,
        emb_dim=64,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
        lr=1e-3,
        batch_size=32,
        epochs=2,
        seed=7,
        max_examples=512,
        shuffle=False,
        log_every=1,
        device="cpu",
    )
    training_duration_seconds = time.perf_counter() - t0

    assert out["global_step"] > 0

    captured = capsys.readouterr()
    losses = _parse_losses(captured.out)
    assert len(losses) >= 6, "Not enough logged training steps to assess loss trend."

    window = 3
    first_avg = sum(losses[:window]) / window
    last_avg = sum(losses[-window:]) / window

    _write_training_log(
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
    )

    assert last_avg < first_avg, (
        f"Expected decreasing loss trend, got first_avg={first_avg:.4f} "
        f"last_avg={last_avg:.4f}"
    )
