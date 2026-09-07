"""
superposition/compute_metrics.py
----------------------------------
Compute and print per-family metrics from superposition JSON results.
Run this immediately after run_superposition.py to sanity-check results.

Usage:
    python superposition/compute_metrics.py --family qwen
    python superposition/compute_metrics.py --all
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

RESULTS_DIR = Path(__file__).parent.parent / "results_superposition"
FAMILIES    = ["qwen", "llama", "gemma"]


def load_result(family: str, variant: str) -> dict | None:
    path = RESULTS_DIR / f"superposition_{family}_{variant}_sub.json"
    if not path.exists():
        print(f"  [missing] {path}")
        return None
    with open(path) as f:
        return json.load(f)


def load_comparison(family: str) -> dict | None:
    path = RESULTS_DIR / f"superposition_{family}_comparison_sub.json"
    if not path.exists():
        print(f"  [missing] {path}")
        return None
    with open(path) as f:
        return json.load(f)


def metrics_for_model(results: dict, label: str):
    """Print role distribution and top/bottom heads for one model."""
    heads = results["heads"]
    roles = Counter(h["role"] for h in heads.values())
    total = len(heads)

    print(f"\n  {label}")
    print(f"    Prompts processed: {results['n_prompts']}")
    print(f"    Architecture:      {results['n_layers']}L × {results['n_heads']}H = {total} heads")
    print(f"    Role distribution:")
    print(f"      context      : {roles['context']:4d}  ({100*roles['context']/total:.1f}%)")
    print(f"      superposition: {roles['superposition']:4d}  ({100*roles['superposition']/total:.1f}%)")
    print(f"      memory       : {roles['memory']:4d}  ({100*roles['memory']/total:.1f}%)")

    # Sanity checks
    if roles["superposition"] == total:
        print(f"    ⚠️  WARNING: ALL heads classified as superposition.")
        print(f"       Check norm_type, W_O indexing, and W_U projection.")
    if results["n_prompts"] < 100:
        print(f"    ⚠️  WARNING: Only {results['n_prompts']} prompts — results may be noisy.")

    # Extreme heads
    sorted_heads = sorted(heads.items(), key=lambda x: x[1]["ratio"])
    print(f"    Top-5 memory heads (lowest ratio):")
    for k, h in sorted_heads[:5]:
        print(f"      {k:8s}  ratio={h['ratio']:+8.3f}  role={h['role']}")
    print(f"    Top-5 context heads (highest ratio):")
    for k, h in sorted_heads[-5:]:
        print(f"      {k:8s}  ratio={h['ratio']:+8.3f}  role={h['role']}")


def metrics_for_comparison(comparison: dict, family: str):
    """Print shift statistics from comparison file."""
    total   = len(comparison)
    changed = [v for v in comparison.values() if v["role_changed"]]
    shifts  = Counter(v["shift_label"] for v in comparison.values())

    print(f"\n  [{family}] Comparison summary")
    print(f"    Total heads compared: {total}")
    print(f"    Role changed: {len(changed)} ({100*len(changed)/total:.1f}%)")
    print(f"    Shift breakdown:")
    for shift_type, count in shifts.most_common():
        pct = 100 * count / total
        marker = "←" if shift_type in ("stable",) else "  "
        print(f"      {shift_type:<30}: {count:4d}  ({pct:.1f}%) {marker}")

    # Average ratio shift
    all_shifts = [v["ratio_shift"] for v in comparison.values()]
    avg_shift  = sum(all_shifts) / len(all_shifts) if all_shifts else 0
    print(f"    Mean ratio shift (instruct - base): {avg_shift:+.4f}")
    print(f"    (positive = instruction tuning pushed heads toward context overall)")


def run_family(family: str):
    print(f"\n{'='*60}")
    print(f"Metrics: {family.upper()}")
    print("="*60)

    base_res  = load_result(family, "base")
    inst_res  = load_result(family, "instruct")
    comp      = load_comparison(family)

    if base_res:
        metrics_for_model(base_res, f"{family}/base")
    if inst_res:
        metrics_for_model(inst_res, f"{family}/instruct")
    if comp:
        metrics_for_comparison(comp, family)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute superposition metrics")
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--all", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.all:
        for fam in FAMILIES:
            run_family(fam)
    elif args.family:
        run_family(args.family)
    else:
        print("Specify --family or --all")
