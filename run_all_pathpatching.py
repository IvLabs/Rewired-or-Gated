"""run_all_pathpatching.py — run Path Patching (causal verification of
LDS's top-k heads, substitution prompts only per spec Part H) across all
3 real model families (Llama-3.2-3B, Gemma-3-4B, Qwen-2.5-3B), both base
and instruct, on a LOCAL GPU.

REQUIRES run_all_lds.py to have already been run for the same
family/variant/split first: PP selects its top-k heads from the EAP
scores in lds2_<tag>_substitution_<set>_eap.json, and errors clearly
(does not crash) if that file is missing for a given variant.

After PP completes, calls each family's build_summary() to produce the
formatted per-family report: results/<family>/<family>_summary_<set_tag>.json,
containing per-(variant, prompt_type) Spearman correlation between PP and
gradact (Merge A -- the "do the methods agree" gate, rho >= 0.5 = pass),
sign agreement, noise-floor baselines, and the top-20 heads with both
their EAP and PP scores side by side.

Calls each family's OWN run_variant(..., steps=("pp",)) -- not a
reimplementation.

Prereqs: same as run_all_lds.py, PLUS run_all_lds.py already run.

Usage:
    # All 3 families, both variants (the real run)
    python run_all_pathpatching.py

    # One family, base only, small subset (smoke test)
    python run_all_pathpatching.py --families llama32_3b --variants base --n-subset 16

    # Rebuild each family's formatted summary from whatever's already on disk (no GPU)
    python run_all_pathpatching.py --summary-only
"""
from __future__ import annotations

import argparse
import importlib

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
                         help="Prompts for the PP pass. Should match what run_all_lds.py used "
                              "for the same variant, or PP's noise-floor comparison mixes subset sizes.")
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
        print(f"\n{'#'*70}\n# FAMILY: {fam}  (Path Patching)\n{'#'*70}")
        for variant in args.variants:
            mod.run_variant(
                variant=variant,
                n_subset=args.n_subset,
                no_shuffle=True,  # irrelevant to PP -- steps=("pp",) never touches the LDS shuffle control
                seed=args.seed,
                heldout_domain=args.heldout_domain,
                split=args.split,
                steps=("pp",),
            )
        mod.build_summary()

    print("\n\n=== DONE ===")
    print("Path Patching complete for:", ", ".join(f"{fam}/{v}" for fam in args.families for v in args.variants))
    print("Formatted per-family summaries (Merge A gate, top heads, noise floor):")
    for fam, mod in modules.items():
        print(f"  results/{fam}/{fam}_summary_{mod.SET_TAG}.json")


if __name__ == "__main__":
    main()
