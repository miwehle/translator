from __future__ import annotations

import sys
from pathlib import Path

from _bootstrap import add_src_dirs

add_src_dirs(__file__)


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
