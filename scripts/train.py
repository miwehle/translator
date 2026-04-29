from __future__ import annotations

from _bootstrap import add_src_dirs

add_src_dirs(__file__)


def main() -> int:
    from lab_infrastructure import run_config_cli

    from translator import TrainRunConfig, train

    run_config_cli(train, TrainRunConfig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
