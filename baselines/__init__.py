"""Single-round baselines using immutable manuscript training selections."""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_root / "baselines/.deps"), str(_root / "kermut")]