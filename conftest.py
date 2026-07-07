"""
Repo-root conftest.py — ensures the project root is on sys.path regardless of
where pytest is invoked from, so tests can `from src.xxx import ...` cleanly.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
