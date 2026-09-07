"""
superposition/aggregation_utils.py
-----------------------------------
Shared bookkeeping: JSON persistence, top-k selection, score averaging.

CRITICAL: JSON cannot store tuple keys like (3, 7).
All save/load functions here handle the conversion transparently.
"""

from __future__ import annotations
import ast
import json
from pathlib import Path


def save_json(data: dict, path: str | Path) -> None:
    """Save dict to JSON. Tuple keys are stringified automatically."""
    path = Path(path)
    serializable = {str(k): v for k, v in data.items()}
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"[aggregation] Saved → {path}")


def load_json(path: str | Path, restore_tuple_keys: bool = False) -> dict:
    """Load JSON. Optionally restore tuple keys from string representation."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path) as f:
        raw = json.load(f)
    if restore_tuple_keys:
        result = {}
        for k, v in raw.items():
            try:
                result[ast.literal_eval(k)] = v
            except Exception:
                result[k] = v
        return result
    return raw


def topk_heads(head_scores: dict, k: int) -> list[tuple[int, int]]:
    """
    Return top-k (layer, head) tuples sorted by score descending.
    head_scores keys must be (layer, head) tuples.
    """
    # Filter out metadata keys like "_meta"
    score_items = {k: v for k, v in head_scores.items()
                   if isinstance(k, tuple)}
    if not score_items:
        raise ValueError("No tuple keys found in head_scores. "
                         "Load with restore_tuple_keys=True.")
    sorted_heads = sorted(score_items.items(), key=lambda x: x[1], reverse=True)
    return [head for head, _ in sorted_heads[:k]]


def mean_abs(values: list[float]) -> float:
    """Mean of absolute values."""
    if not values:
        return 0.0
    return sum(abs(v) for v in values) / len(values)
