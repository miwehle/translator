from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from translator.data_prod import load_arrow_records
from translator.train_prod import Trainer, TrainerConfig, check_dataset


@dataclass(frozen=True)
class TrainBuildConfig:
    dataset_path: str
    runs_dir: str
    run_name: str
    max_examples: int | None = None
    trainer_config_overrides: dict[str, Any] | None = None
    train_kwargs: dict[str, Any] | None = None


BUILD_CONFIG = TrainBuildConfig(
    dataset_path="/content/drive/MyDrive/translator_data/europarl.preprocessed",
    runs_dir="/content/drive/MyDrive/translator_runs",
    run_name="run1",
    max_examples=None,
    trainer_config_overrides={},
    train_kwargs={
        "epochs": 3,
        "num_workers": 2,
        "log_every": 50,
    },
)


def build_run_dir(runs_dir: Path, run_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = runs_dir / f"{timestamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_config_snapshot(run_dir: Path, config: TrainBuildConfig) -> Path:
    config_path = run_dir / "config.json"
    config_path.write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return config_path


def main(config: TrainBuildConfig = BUILD_CONFIG) -> dict[str, Any]:
    dataset_path = Path(config.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    run_dir = build_run_dir(Path(config.runs_dir), config.run_name)
    config_path = write_config_snapshot(run_dir, config)

    ds = load_arrow_records(dataset_path)
    check_result = check_dataset(ds, max_examples=config.max_examples)

    trainer_config = TrainerConfig(
        src_pad_idx=check_result["src_pad_idx"],
        tgt_pad_idx=check_result["tgt_pad_idx"],
        tgt_sos_idx=check_result["tgt_sos_idx"],
        src_vocab_size=check_result["src_vocab_size"],
        tgt_vocab_size=check_result["tgt_vocab_size"],
        num_examples=check_result["num_examples"],
        max_examples=config.max_examples,
        **(config.trainer_config_overrides or {}),
    )

    summary = Trainer(trainer_config).train(
        ds,
        checkpoint_path=run_dir / "model.pt",
        summary_path=run_dir / "summary.json",
        **(config.train_kwargs or {}),
    )
    summary["config_path"] = str(config_path)
    summary["run_dir"] = str(run_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
