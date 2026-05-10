from __future__ import annotations

from _bootstrap import add_src_dirs

add_src_dirs(__file__)


def main() -> int:
    from lab_infrastructure import run_cli

    from translator import train

    run_cli(
        train,
        cli_override_map={
            "dataset": "dataset",
            "validation-dataset": "validation_dataset",
            "lr": "lr",
            "epochs": "epochs",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
