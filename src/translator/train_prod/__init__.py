from typing import Any

from .preflight import check_examples

__all__ = [
    "build_train_prod_config",
    "run_train_prod",
    "run_train_prod_2",
    "train_prod",
    "check_examples",
]


def build_train_prod_config(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from .session import build_train_prod_config as _build_train_prod_config

    return _build_train_prod_config(*args, **kwargs)


def run_train_prod(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from .session import run_train_prod as _run_train_prod

    return _run_train_prod(*args, **kwargs)


def run_train_prod_2(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from .session import run_train_prod_2 as _run_train_prod_2

    return _run_train_prod_2(*args, **kwargs)


def train_prod(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from .training import train_prod as _train_prod

    return _train_prod(*args, **kwargs)
