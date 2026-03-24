from __future__ import annotations

import pytest

from translator.shared.logging_utils import close_translator_logging


@pytest.fixture(autouse=True)
def _close_translator_logging_after_test():
    yield
    close_translator_logging()
