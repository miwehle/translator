from __future__ import annotations

from _bootstrap import add_src_dirs

add_src_dirs(__file__)


def main() -> int:
    from lab_infrastructure import run_cli

    from translator import comet_score

    print(run_cli(comet_score))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
