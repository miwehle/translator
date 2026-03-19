from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import checkpoint_register as cr

from translator.data_prod import DatasetMetadata, load_arrow_records
from translator.train_prod import Example, Trainer, check_dataset
from translator.train_prod.factory import Factory
from translator.train_prod.training import DataLoaderConfig, ModelConfig, TrainConfig


@dataclass(frozen=True)
class TrainingRunConfig:
    dataset_path: str
    run_preflight_check: bool = False


def get_git_commit_hash(repo_root: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def write_training_run_config(
    run_dir: Path,
    config: dict[str, object],
    *,
    build_commit: str,
) -> Path:
    config_path = run_dir / "config.json"
    payload = dict(config)
    payload["build_commit"] = build_commit
    config_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return config_path


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/train.py <config-path>")
        return 1

    config_path = Path(sys.argv[1])
    try:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"Failed to load config: {exc}")
        return 1

    try:
        config = TrainingRunConfig(
            dataset_path=cfg["dataset_path"],
            run_preflight_check=cfg.get("run_preflight_check", False),
        )
        dataset_path = Path(config.dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        model_config = ModelConfig(**(cfg.get("model_config") or {}))
        train_config = TrainConfig(**(cfg.get("train_config") or {}))
        data_loader_config = DataLoaderConfig(**(cfg.get("data_loader_config") or {}))
        run_dir = Path(train_config.runs_dir) / train_config.run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        git_commit = get_git_commit_hash(REPO_ROOT)
        write_training_run_config(
            run_dir,
            {
                "dataset_path": config.dataset_path,
                "run_preflight_check": config.run_preflight_check,
                "model_config": asdict(model_config),
                "train_config": asdict(train_config),
                "data_loader_config": asdict(data_loader_config),
            },
            build_commit=git_commit,
        )

        ds = cast(list[Example], load_arrow_records(dataset_path))
        metadata = DatasetMetadata.from_file(dataset_path / "dataset_manifest.yaml")
        if config.run_preflight_check:
            check_dataset(dataset_path)

        summary = Trainer(Factory(metadata)).train(
            ds,
            train_config=train_config,
            model_config=model_config,
            data_loader_config=data_loader_config,
        )
        register_path = Path(train_config.runs_dir) / "checkpoint_register.csv"
        cr.insert(
            register_path=register_path,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            input_ckpt="",
            dataset_path=str(dataset_path),
            git_commit=git_commit,
            output_ckpt=summary["checkpoint_path"],
        )
    except Exception as exc:
        print(f"Training failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
