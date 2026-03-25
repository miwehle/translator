from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main() -> int:
    from translator import DataLoaderConfig, ModelConfig, TrainConfig, train

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
        dataset_path = Path(cfg["dataset_path"])
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        model_config = ModelConfig(**(cfg.get("model_config") or {}))
        train_config = TrainConfig(**(cfg.get("train_config") or {}))
        data_loader_config = DataLoaderConfig(**(cfg.get("data_loader_config") or {}))
        train(
            dataset_path=dataset_path,
            train_config=train_config,
            model_config=model_config,
            data_loader_config=data_loader_config,
            repo_root=REPO_ROOT,
        )
    except Exception as exc:
        print(f"Training failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
