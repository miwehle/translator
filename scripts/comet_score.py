from __future__ import annotations

import sys
from pathlib import Path

from _bootstrap import add_src_dirs

add_src_dirs(__file__)


def main() -> int:
    from lab_infrastructure.run_config import read_run_config_as

    from translator import CometScoreConfig, comet_score

    if len(sys.argv) != 2:
        print("Usage: python scripts/comet_score.py <config-path>")
        return 1

    try:
        print(comet_score(read_run_config_as(Path(sys.argv[1]), CometScoreConfig)))
    except Exception as exc:
        print(f"COMET scoring failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
