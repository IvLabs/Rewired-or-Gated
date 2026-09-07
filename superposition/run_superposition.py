"""
superposition/run_superposition.py
------------------------------------
CLI runner for superposition analysis.

Run order:
    python superposition/run_superposition.py --family qwen
    # verify results, then:
    python superposition/run_superposition.py --family llama
    python superposition/run_superposition.py --family gemma

Each run produces:
    results_superposition/superposition_{family}_base_sub.json
    results_superposition/superposition_{family}_instruct_sub.json
    results_superposition/superposition_{family}_comparison_sub.json
    results_superposition/logs/superposition_{family}_{timestamp}.log

Supports resume: if a result file already exists, skips that model
variant and loads from disk instead of re-running.
"""

import argparse
import json
import logging
import sys
import time
import torch
from datetime import datetime
from pathlib import Path

# Add superposition directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from load_models import FAMILIES, load_tl_model, load_tokenizer
from load_dataset import load_substitution_prompts
from superposition import compute_superposition, compare_models

RESULTS_DIR = Path(__file__).parent.parent / "results_superposition"
LOGS_DIR    = RESULTS_DIR / "logs"
RESULTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


def setup_logging(family: str) -> logging.Logger:
    """Set up file + console logging for this run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = LOGS_DIR / f"superposition_{family}_{timestamp}.log"

    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(levelname)s  %(message)s",
        handlers= [
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger(__name__)
    log.info(f"Log file: {log_path}")
    return log


def print_summary(results: dict, label: str):
    """Print role distribution and extreme heads."""
    roles = [h["role"] for h in results["heads"].values()]
    nc = roles.count("context")
    nm = roles.count("memory")
    ns = roles.count("superposition")

    print(f"\n  [{label}]")
    print(f"    context={nc}  memory={nm}  superposition={ns}  "
          f"(total {len(roles)} heads, {results['n_layers']}L × {results['n_heads']}H)")

    all_heads = sorted(results["heads"].items(), key=lambda x: x[1]["ratio"])
    print("    Top-5 memory-biased heads:")
    for k, h in all_heads[:5]:
        print(f"      {k:8s}  ratio={h['ratio']:+8.3f}  "
              f"mem={h['mem_pull']:.4f}  ctx={h['ctx_pull']:.4f}")
    print("    Top-5 context-biased heads:")
    for k, h in all_heads[-5:]:
        print(f"      {k:8s}  ratio={h['ratio']:+8.3f}  "
              f"mem={h['mem_pull']:.4f}  ctx={h['ctx_pull']:.4f}")


def print_comparison_summary(comparison: dict, family: str):
    """Print shift distribution."""
    from collections import Counter
    changed     = [v for v in comparison.values() if v["role_changed"]]
    shift_counts= Counter(v["shift_label"] for v in comparison.values())

    print(f"\n  [{family}] Role changes: {len(changed)}/{len(comparison)}")
    for shift_type, count in shift_counts.most_common():
        if shift_type != "stable":
            print(f"    {shift_type}: {count}")

    top_shifts = sorted(changed, key=lambda x: abs(x["ratio_shift"]), reverse=True)
    print(f"  Top-10 largest ratio shifts:")
    for v in top_shifts[:10]:
        print(f"    L{v['layer']}H{v['head']:2d}: "
              f"{v['base_role']:14s} → {v['inst_role']:14s}  "
              f"shift={v['ratio_shift']:+.3f}")


def run_family(family: str, device: str):
    """Run full superposition analysis for one model family."""
    log = setup_logging(family)
    cfg = FAMILIES[family]

    print(f"\n{'='*65}")
    print(f"Family: {family.upper()}  |  Substitution prompts")
    print(f"  Base:      {cfg['base']}")
    print(f"  Instruct:  {cfg['instruct']}")
    print(f"  norm_type: {cfg['norm_type']}")
    print("="*65)

    # Load tokenizer
    tokenizer = load_tokenizer(family)

    # Load prompts (filtered to this tokenizer's single-token answers)
    prompts = load_substitution_prompts(tokenizer, verbose=True)
    if not prompts:
        print(f"[ERROR] No valid prompts for {family}. Exiting.")
        return

    saved = {}

    for variant in ["base", "instruct"]:
        out_path = RESULTS_DIR / f"superposition_{family}_{variant}_sub.json"

        # Resume support — skip if already computed
        if out_path.exists():
            print(f"\n[{family}/{variant}] Exists, loading: {out_path}")
            with open(out_path) as f:
                saved[variant] = json.load(f)
            print_summary(saved[variant], f"{family}/{variant}")
            continue

        # Load model
        hf_name = cfg[variant]
        t0      = time.time()
        model   = load_tl_model(hf_name, family, device)

        log.info(f"Model loaded: {hf_name}  "
                 f"n_layers={model.cfg.n_layers} n_heads={model.cfg.n_heads} "
                 f"d_model={model.cfg.d_model}")

        # Run superposition
        print(f"\n[{family}/{variant}] Computing superposition "
              f"({len(prompts)} prompts, "
              f"{model.cfg.n_layers}L × {model.cfg.n_heads}H = "
              f"{model.cfg.n_layers * model.cfg.n_heads} heads)...")

        results = compute_superposition(
            model     = model,
            prompts   = prompts,
            norm_type = cfg["norm_type"],
            device    = device,
            label     = f"{family}/{variant}",
        )
        results["family"]  = family
        results["variant"] = variant
        results["model"]   = hf_name

        saved[variant] = results
        out_path.write_text(json.dumps(results, indent=2))
        print(f"[{family}/{variant}] Saved → {out_path}")
        print_summary(results, f"{family}/{variant}")

        elapsed = time.time() - t0
        log.info(f"Done in {elapsed:.0f}s  ({results['n_prompts']} prompts)")

        # Free GPU memory
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # Comparison
    if "base" in saved and "instruct" in saved:
        comp_path  = RESULTS_DIR / f"superposition_{family}_comparison_sub.json"
        comparison = compare_models(saved["base"], saved["instruct"])
        comp_path.write_text(json.dumps(comparison, indent=2))
        print(f"\n[{family}] Comparison saved → {comp_path}")
        print_comparison_summary(comparison, family)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Superposition analysis — one forward pass per prompt"
    )
    parser.add_argument(
        "--family", choices=["qwen", "llama", "gemma"],
        help="Model family to run."
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all three families: qwen → llama → gemma."
    )
    parser.add_argument(
        "--device", default=None,
        help="Force device (cuda/cpu). Default: auto-detect."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.all:
        for fam in ["qwen", "llama", "gemma"]:
            run_family(fam, device)
    elif args.family:
        run_family(args.family, device)
    else:
        print("Specify --family qwen/llama/gemma or --all")
        print("Example: python superposition/run_superposition.py --family qwen")
