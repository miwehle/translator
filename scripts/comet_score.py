from __future__ import annotations

from _bootstrap import add_src_dirs

add_src_dirs(__file__)


def main() -> int:
    from lab_infrastructure import run

    from translator import CometScoreConfig, comet_score

    print(run(comet_score, CometScoreConfig))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
