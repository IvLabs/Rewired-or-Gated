"""run_single_token_filter_all.py — single-token filter + survival stats for
all 3 model families (Llama-3.2-3B, Gemma-3-4B, Qwen-2.5-3B), base + instruct.

For each of the 6 variants: loads the REAL model (not just its tokenizer --
spec A2.2's log-prob refinement needs an actual forward pass), runs
load_single_token_prompts (on "substitution"; survival is identical for
"coherent" since only memory_answer/context_answer decide survival, not
prompt.text), and writes:

    results/<family>/single_token_survival_<family>_<variant>.json

Each file also records, per row, whether A2.2's log-prob-preferred alias
differs from A2.1's naive first-single-token alias -- this quantifies the
alias-selection bias risk (does the filter ever pick a word the model
wouldn't actually say?).

A final aggregate is written to results/single_token_filter_summary.json:
per-family base==instruct check (expected identical -- same tokenizer), and
the cross-family intersection size, reported as a DIAGNOSTIC ONLY. Per the
2026-07-08 team decision: each family's own maximal single-token set is the
primary backbone for the within-family base-vs-instruct comparison; the
cross-family intersection is not used to gate or restrict any real run.

Usage:
    # All 6 variants (default) -- downloads + loads each model in turn,
    # freeing GPU memory between models. Skips any variant already done.
    ../.venv/Scripts/python.exe singleton_fix/run_single_token_filter_all.py

    # Just one family, or one variant
    ../.venv/Scripts/python.exe singleton_fix/run_single_token_filter_all.py --families qwen25_3b
    ../.venv/Scripts/python.exe singleton_fix/run_single_token_filter_all.py --families llama32_3b --variants base

    # Rebuild the aggregate summary from whatever per-variant files already exist
    ../.venv/Scripts/python.exe singleton_fix/run_single_token_filter_all.py --summary-only

Access: Llama-3.2-3B(-Instruct) and google/gemma-3-4b-{pt,it} are gated --
HF_TOKEN in .env must belong to an account that accepted all four licenses.
Qwen/Qwen2.5-3B(-Instruct) is public, no token needed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from contract import ConflictPrompt
from foundation import load_conflict_prompts, load_model
from singleton_fix.single_token_loader import (
    _select_memory_answer,
    load_single_token_prompts,
    load_cached_single_token_prompts,  # re-exported for convenience; canonical home is single_token_loader.py
)

RESULTS_DIR = Path(_ROOT) / "results"
DTYPE = "bfloat16"

FAMILIES: Dict[str, Dict[str, str]] = {
    "llama32_3b": {
        "base": "meta-llama/Llama-3.2-3B",
        "instruct": "meta-llama/Llama-3.2-3B-Instruct",
    },
    "gemma3_4b": {
        "base": "google/gemma-3-4b-pt",
        "instruct": "google/gemma-3-4b-it",
    },
    "qwen25_3b": {
        "base": "Qwen/Qwen2.5-3B",
        "instruct": "Qwen/Qwen2.5-3B-Instruct",
    },
}


def _domain_counts(prompts: List[ConflictPrompt]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for p in prompts:
        counts[p.domain] = counts.get(p.domain, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _out_path(family: str, variant: str) -> Path:
    return RESULTS_DIR / family / f"single_token_survival_{family}_{variant}.json"


def check_variant(model_name: str, family: str, variant: str, force: bool = False) -> Dict:
    """Run (or load a cached) single-token survival check for one variant.

    Writes a REUSABLE cache, not just stats: each surviving row carries a
    stable row_index (position in ParaConflict's deterministic "test" split
    load order -- the same every time, since it's the same HF dataset/split),
    both text variants (substitution + coherent -- confirmed identical
    survival set between them, see module docstring), the full alias list,
    the distractor, the A2.1+A2.2-corrected canonical memory_answer, AND the
    naive A2.1-only answer for comparison -- enough to reconstruct a
    ConflictPrompt directly via load_cached_single_token_prompts() below,
    without reloading the model or rerunning the (GPU-bound) A2.2 oracle.
    """
    out_path = _out_path(family, variant)
    if out_path.exists() and not force:
        print(f"[survival] {family}/{variant} already done -> {out_path} (skip)")
        return json.loads(out_path.read_text(encoding="utf-8"))

    print(f"\n{'='*70}\n[{family}/{variant}] loading {model_name} ...\n{'='*70}")
    model = load_model(model_name, dtype=DTYPE)

    # Full unfiltered substitution list, in the dataset's deterministic load
    # order -- its position IS the stable row_index. Also load the coherent
    # column so each surviving row can carry both text variants without a
    # second model load.
    all_substitution = load_conflict_prompts(prompt_type="substitution")
    all_coherent = load_conflict_prompts(prompt_type="coherent")
    # (domain, clean_text) is untouched by A2 alias correction and unique per
    # row in practice -- a safe natural key back to row_index/coherent text.
    index_by_key = {(p.domain, p.clean_text): idx for idx, p in enumerate(all_substitution)}
    coherent_text_by_key = {(p.domain, p.clean_text): p.text for p in all_coherent}

    prompts = load_single_token_prompts(model, prompt_type="substitution")

    n_a22_changed = 0
    rows: List[Dict] = []
    for p in prompts:
        naive = _select_memory_answer(model, p.memory_aliases, use_logprob_refinement=False)
        changed = naive != p.memory_answer
        if changed:
            n_a22_changed += 1
        key = (p.domain, p.clean_text)
        rows.append({
            "row_index": index_by_key.get(key),
            "domain": p.domain,
            "clean_text": p.clean_text,
            "substitution_text": p.text,
            "coherent_text": coherent_text_by_key.get(key),
            "memory_aliases": p.memory_aliases,
            "context_answer": p.context_answer,
            "memory_answer": p.memory_answer,
            "a21_naive_answer": naive,
            "a22_changed": changed,
        })

    result = {
        "family": family,
        "variant": variant,
        "model": model_name,
        "n_survived": len(prompts),
        "domain_counts": _domain_counts(prompts),
        "n_a22_changed_vs_a21": n_a22_changed,
        "rows": rows,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[survival] saved -> {out_path}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def build_summary() -> Dict:
    """Rebuild the aggregate summary from whatever per-variant files already
    exist on disk (tolerates missing variants -- reports what it has)."""
    by_family: Dict[str, Dict[str, Dict]] = {}
    for family, variants in FAMILIES.items():
        for variant in variants:
            path = _out_path(family, variant)
            if path.exists():
                by_family.setdefault(family, {})[variant] = json.loads(path.read_text(encoding="utf-8"))

    def _key_set(variant_data: Dict) -> set:
        return {(row["row_index"]) for row in variant_data["rows"]}

    families_summary = {}
    all_row_key_sets: List[set] = []
    for family, variants in by_family.items():
        base = variants.get("base")
        instruct = variants.get("instruct")
        base_instruct_identical: Optional[bool] = None
        if base is not None and instruct is not None:
            base_instruct_identical = _key_set(base) == _key_set(instruct)
        families_summary[family] = {
            "base_n_survived": base["n_survived"] if base else None,
            "instruct_n_survived": instruct["n_survived"] if instruct else None,
            "base_instruct_identical_set": base_instruct_identical,
        }
        for v in variants.values():
            all_row_key_sets.append(_key_set(v))

    intersection_indices = sorted(set.intersection(*all_row_key_sets)) if all_row_key_sets else []
    intersection_size = len(intersection_indices)
    n_variants_found = sum(len(v) for v in by_family.values())

    diagnostic_note = (
        "DIAGNOSTIC / OPTIONAL SECONDARY ROBUSTNESS CHECK ONLY -- NOT the "
        "primary backbone. Each family's own maximal single-token set is "
        "the primary backbone for the within-family base-vs-instruct "
        "comparison; intersecting across all 3 tokenizers loses power and "
        "skews further toward famous-entity rows. Team decision, 2026-07-08."
    )

    summary = {
        "n_variants_found": n_variants_found,
        "n_variants_expected": sum(len(v) for v in FAMILIES.values()),
        "families": families_summary,
        "cross_family_intersection_size": intersection_size,
        "note": diagnostic_note,
    }
    out_path = RESULTS_DIR / "single_token_filter_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[summary] saved -> {out_path} ({n_variants_found}/{summary['n_variants_expected']} variants found)")

    # Persist the actual intersecting row_index list (not just the count) so
    # it can be reused as an optional shared-prompt-set robustness check --
    # see load_intersection_row_indices() below.
    intersection_path = RESULTS_DIR / "single_token_intersection.json"
    intersection_path.write_text(json.dumps({
        "n": intersection_size,
        "row_indices": intersection_indices,
        "computed_from_variants": sorted(
            f"{family}/{variant}" for family, variants in by_family.items() for variant in variants
        ),
        "note": diagnostic_note,
    }, indent=2), encoding="utf-8")
    print(f"[summary] saved -> {intersection_path} ({intersection_size} rows survive every variant checked)")

    return summary


def load_intersection_row_indices() -> List[int]:
    """Read the row_index list from single_token_intersection.json (must run
    build_summary()/--summary-only at least once after all desired variants
    are checked). Raises FileNotFoundError with a clear message otherwise."""
    path = RESULTS_DIR / "single_token_intersection.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist yet -- run "
            f"'singleton_fix/run_single_token_filter_all.py --summary-only' "
            f"(or the full run) after all desired variants have a survival file."
        )
    return json.loads(path.read_text(encoding="utf-8"))["row_indices"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--families", nargs="+", default=list(FAMILIES.keys()), choices=list(FAMILIES.keys()))
    parser.add_argument("--variants", nargs="+", default=["base", "instruct"], choices=["base", "instruct"])
    parser.add_argument("--force", action="store_true", help="Recompute even if a result file already exists.")
    parser.add_argument("--summary-only", action="store_true", dest="summary_only",
                         help="Skip all model loading; just rebuild the aggregate summary from existing files.")
    args = parser.parse_args()

    if args.summary_only:
        build_summary()
        return

    results: List[Dict] = []
    for family in args.families:
        for variant in args.variants:
            model_name = FAMILIES[family][variant]
            results.append(check_variant(model_name, family, variant, force=args.force))

    print("\n\n=== per-variant survival ===")
    for r in results:
        print(f"{r['family']}/{r['variant']}: {r['n_survived']} survived, "
              f"{r['n_a22_changed_vs_a21']} rows where A2.2 changed the alias vs A2.1")

    build_summary()


if __name__ == "__main__":
    main()
