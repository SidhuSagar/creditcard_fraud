"""FraudShield AI backend package.

Ensures the ``backend`` directory is importable so that ``training.*`` (the
single source of truth for feature order and preprocessing) can be imported by
the API regardless of the working directory uvicorn is launched from.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

__version__ = "1.0.0"
