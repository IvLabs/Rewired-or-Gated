"""run_all_lds.py — run LDS (all 3 flavors: gradnorm, gradact, eap) +
behavioral/log-prob CRR across all 3 real model families (Llama-3.2-3B,
Gemma-3-4B, Qwen-2.5-3B), both base and instruct, on a LOCAL GPU.

For a 24GB+ GPU this runs directly -- no Modal needed. Each variant is
loaded once, run through the full EAP screen (both substitution and
coherent prompt types) plus CRR, then freed before the next variant loads,
so peak memory is one model at a time (largest is Gemma-3-4B at ~8.6GB
bf16 weights + activations -- comfortable headroom on 24GB even without
the gradient checkpointing this already enables).

This is LDS-only (+ CRR, which is cheap and needs no second GPU pass) --
run run_all_pathpatching.py afterwards for Path Patching, which needs
this script's lds2_..._eap.json output to select its top-k heads from.

Calls each family's OWN run_variant(..., steps=("lds", "crr")) -- not a
reimplementation -- so this exercises the exact code path
run_llama32.py/run_gemma3.py/run_qwen25.py already have, just orchestrated
across all 6 variants in one command instead of one script per family.

Prereqs:
  - .env in the repo root with HF_TOKEN=<your token> (needed for the
    gated Llama/Gemma repos; Qwen is public).
  - The single-token survival cache already exists for each variant you
    run (results/<family>/single_token_survival_<family>_<variant>.json)
    -- already committed to the repo, nothing to regenerate.
  - pip install -r requirements.txt

Usage:
    # All 3 families, both variants, full dataset (the real run)
    python run_all_lds.py

    # One family, base only, small subset (smoke test before committing to a full run)
    python run_all_lds.py --families llama32_3b --variants base --n-subset 16

    # Rebuild each family's formatted summary from whatever's already on disk (no GPU)
    python run_all_lds.py --summary-only
"""
from __future__ import annotations

import argparse
import importlib
from typing import List, Optional

FAMILY_MODULE = {
    "llama32_3b": "run_llama32",
    "gemma3_4b": "run_gemma3",
    "qwen25_3b": "singleton_fix.run_qwen25",
}
ALL_FAMILIES = list(FAMILY_MODULE.keys())
ALL_VARIANTS = ["base", "instruct"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--families", nargs="+", default=ALL_FAMILIES, choices=ALL_FAMILIES)
    parser.add_argument("--variants", nargs="+", default=ALL_VARIANTS, choices=ALL_VARIANTS)
    parser.add_argument("--n-subset", type=int, default=None, dest="n_subset",
                         help="Prompts per (variant, prompt_type) cell. Omit for the full dataset.")
    parser.add_argument("--no-shuffle", action="store_true",
                         help="Skip the shuffled-label noise-floor control (faster).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heldout-domain", default=None, dest="heldout_domain")
    parser.add_argument("--split", default="all", choices=["train", "heldout", "all"])
    parser.add_argument("--summary-only", action="store_true", dest="summary_only",
                         help="Skip all GPU work; just rebuild each family's formatted summary JSON.")
    args = parser.parse_args()

    modules = {fam: importlib.import_module(FAMILY_MODULE[fam]) for fam in args.families}

    if args.summary_only:
        for fam, mod in modules.items():
            mod.build_summary()
        return

    for fam in args.families:
        mod = modules[fam]
        print(f"\n{'#'*70}\n# FAMILY: {fam}  (LDS + CRR)\n{'#'*70}")
        for variant in args.variants:
            mod.run_variant(
                variant=variant,
                n_subset=args.n_subset,
                no_shuffle=args.no_shuffle,
                seed=args.seed,
                heldout_domain=args.heldout_domain,
                split=args.split,
                steps=("lds", "crr"),
            )
        # NOT calling build_summary() here: it needs BOTH the eap file (just
        # written) AND pp_topheads (not written yet -- that's the next
        # script) to build a cell, so it would just print "missing files --
        # skip" for everything right now. run_all_pathpatching.py calls it
        # once PP output exists too.

    print("\n\n=== DONE ===")
    print("LDS + CRR complete for:", ", ".join(f"{fam}/{v}" for fam in args.families for v in args.variants))
    print("Raw per-cell files: results/<family>/lds2_*.json, crr_*.json")
    print("Next: run_all_pathpatching.py to add Path Patching + the formatted LDS-vs-PP triangulation summary.")


if __name__ == "__main__":
    main()
