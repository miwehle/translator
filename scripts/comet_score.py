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
    from lab_infrastructure.run_config import read_run_config

    from translator import CometScoreConfig, comet_score

    if len(sys.argv) != 2:
        print("Usage: python scripts/comet_score.py <config-path>")
        return 1

    try:
        print(comet_score(CometScoreConfig(**read_run_config(Path(sys.argv[1])))))
    except Exception as exc:
        print(f"COMET scoring failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
