"""Shared setup for the experiment scripts.

Each script is run from the command line, so it needs the repository root on
the import path. Paths come from this file's location, not the working
directory, so the scripts work from anywhere.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
DATA_DIR = RESULTS_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _jsonable(value):
    """Turn numpy scalars, arrays and complex numbers into plain JSON."""
    import numpy as np

    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, complex):
        return {'real': value.real, 'imag': value.imag}
    if isinstance(value, float) and (value != value or value in (float('inf'),
                                                                float('-inf'))):
        return None
    return value


def save_json(data, path):
    """Write results as JSON. Non-finite values are written as null."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), indent=2))
    return path


def checkpoint(name, compute):
    """Compute a sweep cell once and reuse it afterwards.

    The big cells take minutes, so an interrupted run would otherwise start
    over. Initial conditions are drawn by the caller before compute() runs,
    so the random stream advances the same way either way. Delete
    results/data/cache to force a clean recomputation.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.json"

    if path.exists():
        print(f"  (reusing cached {name})")
        return json.loads(path.read_text())

    value = compute()
    path.write_text(json.dumps(value))
    return value


def banner(text):
    print(text)
    print("=" * 55)
