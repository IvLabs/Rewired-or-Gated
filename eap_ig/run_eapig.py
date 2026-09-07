"""Single-token EAP-IG runner for all 3 model families x base/instruct.

Substitution only, single-token cache only (with a live-filter fallback if a
cache is ever missing). One model resident at a time. Also the orchestrator:
`--family all` loops all three families.

Usage:
  # Smoke: Qwen (public, no gated login) base, coarse, 8 prompts
  python -m eap_ig.run_eapig --family qwen25_3b --variant base \
      --granularity coarse --n-subset 8

  # Full run, one family both variants (backgrounded recommended)
  python -m eap_ig.run_eapig --family qwen25_3b --variant both

  # Everything (gated Llama/Gemma need HF_TOKEN accepted)
  python -m eap_ig.run_eapig --family all --variant both
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from eap_ig.eap_ig import EdgeAttributionConfig
from eap_ig.eap_ig_single_token import (
    SET_TAG,
    build_result_json_single_token,
    compute_edge_attribution_single_token,
)
from eap_ig.run_eap_ig import load_hooked_model
from singleton_fix.single_token_loader import (
    load_cached_single_token_prompts,
    load_single_token_prompts,
    print_domain_composition,
)

RESULTS_ROOT = Path(_ROOT) / "results"
PROMPT_TYPE = "substitution"

FAMILY_SPECS: Dict[str, dict] = {
    "llama32_3b": {
        "base":     {"tag": "llama32_3b_base",     "repo": "meta-llama/Llama-3.2-3B"},
        "instruct": {"tag": "llama32_3b_instruct", "repo": "meta-llama/Llama-3.2-3B-Instruct"},
    },
    "gemma3_4b": {
        "base":     {"tag": "gemma3_4b_base",     "repo": "google/gemma-3-4b-pt"},
        "instruct": {"tag": "gemma3_4b_instruct", "repo": "google/gemma-3-4b-it"},
    },
    "qwen25_3b": {
        "base":     {"tag": "qwen25_3b_base",     "repo": "Qwen/Qwen2.5-3B"},
        "instruct": {"tag": "qwen25_3b_instruct", "repo": "Qwen/Qwen2.5-3B-Instruct"},
    },
}
DTYPE_DEFAULT = "bfloat16"


def _load_prompts(model, family: str, variant: str) -> List:
    """Cache-first (no model needed), live-filter fallback (needs model)."""
    try:
        return load_cached_single_token_prompts(family, variant, prompt_type=PROMPT_TYPE)
    except FileNotFoundError:
        print(f"[warn] no cache for {family}/{variant} -- live filtering (needs model).")
        return load_single_token_prompts(model, prompt_type=PROMPT_TYPE)


def run_variant(
    family: str,
    variant: str,
    ig_steps: int = 5,
    granularity: str = "full",
    n_subset: Optional[int] = None,
    also_plain: bool = True,
    top_k: int = 1000,
    seed: int = 42,
    force: bool = False,
    dtype: Optional[str] = DTYPE_DEFAULT,
    device: Optional[str] = None,
) -> List[str]:
    spec = FAMILY_SPECS[family][variant]
    tag, repo = spec["tag"], spec["repo"]
    # Granularity is always in the filename so coarse and full never collide; a
    # subset (smoke) run is quarantined under smoke/ so it can't masquerade as the
    # real 764/870-row result. Mirrors eap_ig/modal_run_eapig.py._out_paths.
    if n_subset is not None:
        out_dir = RESULTS_ROOT / family / "smoke"
        suffix = f"_{granularity}_smoke_n{n_subset}"
    else:
        out_dir = RESULTS_ROOT / family
        suffix = f"_{granularity}"
    igp = out_dir / f"eapig_{tag}_{PROMPT_TYPE}_{SET_TAG}{suffix}.json"
    eap = out_dir / f"eap_{tag}_{PROMPT_TYPE}_{SET_TAG}{suffix}.json"
    write_plain = also_plain and ig_steps != 1
    want = [igp] + ([eap] if write_plain else [])
    if not force and all(p.exists() for p in want):
        print(f"[run_eapig] outputs exist for {family}/{variant}; skipping (use --force).")
        return [str(p) for p in want]

    print(f"\n{'='*70}\n[run_eapig] {family}/{variant} tag={tag} repo={repo}\n{'='*70}")
    model = load_hooked_model(repo, dtype=dtype, device=device)

    prompts = _load_prompts(model, family, variant)
    print_domain_composition(prompts)
    if n_subset is not None and n_subset < len(prompts):
        import random as _random
        prompts = _random.Random(seed).sample(prompts, n_subset)
        print(f"[run_eapig] smoke subset -> {len(prompts)} prompts")

    outputs: List[str] = []
    try:
        def _run(steps: int, out_path: Path) -> None:
            config = EdgeAttributionConfig(
                n_steps=steps, granularity=granularity, contract_device="cpu")
            meta: Dict = {}
            scores = compute_edge_attribution_single_token(
                model, prompts, config, _meta_out=meta,
                desc=f"EAP-IG-st[{family}/{variant}]",
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            payload = build_result_json_single_token(scores, meta, tag, tokenizer_name=repo, top_k=top_k)
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[run_eapig] saved -> {out_path}")

        _run(ig_steps, igp)
        outputs.append(str(igp))
        if write_plain:
            _run(1, eap)
            outputs.append(str(eap))
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", choices=list(FAMILY_SPECS) + ["all"], default="all")
    parser.add_argument("--variant", choices=["base", "instruct", "both"], default="both")
    parser.add_argument("--granularity", choices=["full", "coarse"], default="full")
    parser.add_argument("--ig-steps", type=int, default=5, dest="ig_steps")
    parser.add_argument("--n-subset", type=int, default=None, dest="n_subset",
                        help="Smoke only; omit to run ALL cached rows.")
    parser.add_argument("--no-plain", action="store_true")
    parser.add_argument("--top-k", type=int, default=1000, dest="top_k")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default=DTYPE_DEFAULT)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dtype = None if args.dtype.lower() == "float32" else args.dtype
    families = list(FAMILY_SPECS) if args.family == "all" else [args.family]
    variants = ["base", "instruct"] if args.variant == "both" else [args.variant]

    for family in families:
        for variant in variants:
            try:
                run_variant(
                    family, variant, ig_steps=args.ig_steps, granularity=args.granularity,
                    n_subset=args.n_subset, also_plain=not args.no_plain, top_k=args.top_k,
                    seed=args.seed, force=args.force, dtype=dtype, device=args.device,
                )
            except Exception as exc:  # noqa: BLE001 -- keep going across cells
                print(f"[run_eapig] FAILED {family}/{variant}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
