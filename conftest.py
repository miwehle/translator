from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"

# Ensure pytest and VS Code test discovery can import modules from src/.
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
