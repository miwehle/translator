from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
SHARED_SRC_PATH = PROJECT_ROOT.parent / "nmt_lab_shared" / "src"

# Ensure pytest and VS Code test discovery can import modules from src/.
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(SHARED_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SHARED_SRC_PATH))
