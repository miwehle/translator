from __future__ import annotations

import json
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

    from translator import PreflightConfig, check_dataset

    if len(sys.argv) != 2:
        print("Usage: python scripts/preflight.py <config-path>")
        return 1

    config_path = Path(sys.argv[1])
    try:
        cfg = read_run_config_as(config_path, PreflightConfig)
    except Exception as exc:
        print(f"Failed to load config: {exc}")
        return 1

    try:
        result = check_dataset(cfg)
    except Exception as exc:
        print(f"Preflight check failed: {exc}")
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
