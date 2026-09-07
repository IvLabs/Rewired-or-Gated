"""
superposition/run_intervention.py
-----------------------------------
CLI runner for logit intervention experiments.

REQUIRES: superposition comparison JSON files to exist first.
Run run_superposition.py before this script.

Usage:
    python superposition/run_intervention.py --family qwen --topk 5
    python superposition/run_intervention.py --family llama --topk 5
    python superposition/run_intervention.py --family gemma --topk 5

Produces:
    results_superposition/intervention_{family}.json
    results_superposition/logs/intervention_{family}_{timestamp}.log
"""

import argparse
import json
import logging
import sys
import time
import torch
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from load_models import FAMILIES, load_tl_model, load_tokenizer
from load_dataset import load_substitution_prompts
from intervention import run_intervention_experiment

RESULTS_DIR = Path(__file__).parent.parent / "results_superposition"
LOGS_DIR    = RESULTS_DIR / "logs"
RESULTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


def setup_logging(family: str) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = LOGS_DIR / f"intervention_{family}_{timestamp}.log"
    logging.basicConfig(
        level    = logging.INFO,
        format   = "%(asctime)s  %(levelname)s  %(message)s",
        handlers = [
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.getLogger(__name__).info(f"Log: {log_path}")


def run_family(family: str, topk: int, device: str):
    """Run intervention experiment for one family."""
    setup_logging(family)
    cfg = FAMILIES[family]

    # Check comparison file exists
    comp_path = RESULTS_DIR / f"superposition_{family}_comparison_sub.json"
    if not comp_path.exists():
        print(f"[ERROR] Missing: {comp_path}")
        print(f"Run first: python superposition/run_superposition.py --family {family}")
        return

    with open(comp_path) as f:
        comparison = json.load(f)

    print(f"\n{'='*65}")
    print(f"Intervention: {family.upper()}  |  top-{topk} heads")
    print("="*65)

    # Load tokenizer and prompts
    tokenizer = load_tokenizer(family)
    prompts   = load_substitution_prompts(tokenizer, verbose=True)
    if not prompts:
        print(f"[ERROR] No valid prompts for {family}.")
        return

    # Load BASE model (interventions run on base model)
    t0    = time.time()
    model = load_tl_model(cfg["base"], family, device)

    # Run all four conditions
    results = run_intervention_experiment(
        model      = model,
        prompts    = prompts,
        comparison = comparison,
        topk       = topk,
        label      = family,
    )

    # Save results
    out_path = RESULTS_DIR / f"intervention_{family}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[intervention/{family}] Saved → {out_path}")

    # Print summary table
    print(f"\n{'─'*55}")
    print(f"  {'Condition':<25}  {'CRR':>6}  {'ΔCRR':>7}")
    print(f"{'─'*55}")
    print(f"  {'baseline':<25}  {results['baseline']['crr']:>6.4f}  {'—':>7}")
    for cond, res in results["conditions"].items():
        direction = "✓" if _correct_direction(cond, res["delta_crr"]) else "✗"
        print(f"  {cond:<25}  {res['crr']:>6.4f}  {res['delta_crr']:>+7.4f}  {direction}")
    print(f"{'─'*55}")
    print(f"  ✓ = ΔCRR in expected direction")
    print(f"  Elapsed: {time.time()-t0:.0f}s")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()


def _correct_direction(cond_name: str, delta_crr: float) -> bool:
    """Check whether ΔCRR is in the predicted direction."""
    expected_positive = {"amplify_context", "suppress_memory"}
    if cond_name in expected_positive:
        return delta_crr > 0
    else:
        return delta_crr < 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Logit intervention experiment"
    )
    parser.add_argument("--family", choices=["qwen", "llama", "gemma"])
    parser.add_argument("--topk",   type=int, default=5)
    parser.add_argument("--all",    action="store_true")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.all:
        for fam in ["qwen", "llama", "gemma"]:
            run_family(fam, args.topk, device)
    elif args.family:
        run_family(args.family, args.topk, device)
    else:
        print("Specify --family or --all")
        print("Example: python superposition/run_intervention.py --family qwen --topk 5")
