from .preflight import validate_records_contract
from typing import Any

__all__ = [
    "build_train_prod_config",
    "run_train_prod",
    "train_prod",
    "validate_records_contract",
]


def build_train_prod_config(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from .session import build_train_prod_config as _build_train_prod_config

    return _build_train_prod_config(*args, **kwargs)


def run_train_prod(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from .session import run_train_prod as _run_train_prod

    return _run_train_prod(*args, **kwargs)


def train_prod(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from .training import train_prod as _train_prod

    return _train_prod(*args, **kwargs)
