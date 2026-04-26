import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

CLI = ROOT / "cli"
if str(CLI) not in sys.path:
    sys.path.insert(0, str(CLI))

print("TEST MODULE: tests/conftest.py loaded")
