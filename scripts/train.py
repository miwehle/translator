from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SHARED_SRC_DIR = REPO_ROOT.parent / "lab_infrastructure" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SHARED_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_SRC_DIR))


def main() -> int:
    from lab_infrastructure.run_config import read_run_config_as

    from translator import TrainRunConfig, train

    if len(sys.argv) != 2:
        print("Usage: python scripts/train.py <config-path>")
        return 1

    try:
        cfg = read_run_config_as(Path(sys.argv[1]), TrainRunConfig)
        train(cfg)
    except Exception as exc:
        print(f"Training failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
