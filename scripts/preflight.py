from __future__ import annotations

import json
import sys
from pathlib import Path

from _bootstrap import add_src_dirs

add_src_dirs(__file__)


def main() -> int:
    from lab_infrastructure import run

    from translator import preflight_check

    if len(sys.argv) != 2:
        print("Usage: python scripts/preflight.py <config-path>")
        return 1

    config_path = Path(sys.argv[1])
    try:
        result = run(preflight_check, config_path)
    except Exception as exc:
        print(f"Preflight check failed: {exc}")
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
