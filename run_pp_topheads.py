"""run_pp_topheads.py — Path Patching on EAP-selected top heads.

Reads EAP scores produced by run_lds.py, selects the top-K context-following
(most positive) and top-K memory-protecting (most negative) heads, then runs
activation patching at hook_z for that subset only.

Because only ~20 heads are patched instead of all n_layers × n_heads, this is
cheap enough to run on large models where full PP would be prohibitive.

Usage:
  # Step 1 (if not already done):
  #   python run_lds.py --model gpt2 --model-tag gpt2_base --prompt-type substitution

  # Step 2:
  python run_pp_topheads.py \
      --model gpt2 --model-tag gpt2_base \
      --prompt-type substitution \
      --k 10

  # With explicit EAP file:
  python run_pp_topheads.py \
      --model gpt2 --model-tag gpt2_base \
      --prompt-type substitution \
      --eap-scores results/lds2_gpt2_base_substitution_full_eap.json

Output:
  results/pp_topheads_<model-tag>_<ptype>.json
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from contract import HeadScores
from foundation import load_model, load_conflict_prompts

# Import from path_patching package
sys.path.insert(0, os.path.join(_ROOT, "path_patching"))
from path_patching import run_path_patching

RESULTS_DIR = Path(_ROOT) / "results"


# ---------------------------------------------------------------------------
# Head selection
# ---------------------------------------------------------------------------

def select_top_heads(
    eap_scores: Dict[Tuple[int, int], float],
    k: int,
) -> List[Tuple[int, int]]:
    """Top-k most-positive (context-following) ∪ top-k most-negative (memory-protecting).

    Returns deduped union with context heads first (descending score), then
    memory heads (ascending score, i.e. most negative first). Stable within
    each group. k is clamped to len(eap_scores); k <= 0 returns empty list.
    """
    if k <= 0 or not eap_scores:
        return []
    k = min(k, len(eap_scores))
    sorted_asc = sorted(eap_scores.items(), key=lambda kv: kv[1])
    sorted_desc = sorted(eap_scores.items(), key=lambda kv: kv[1], reverse=True)
    ctx_heads = [h for h, _ in sorted_desc[:k]]
    mem_heads = [h for h, _ in sorted_asc[:k]]
    seen = set(ctx_heads)
    union = list(ctx_heads)
    for h in mem_heads:
        if h not in seen:
            union.append(h)
            seen.add(h)
    return union


def _load_eap(path: Path) -> Dict[Tuple[int, int], float]:
    """Load EAP JSON {"<l>_<h>": float} into {(l, h): float}."""
    if not path.exists():
        print(f"[pp-topheads] ERROR: EAP file not found: {path}", file=sys.stderr)
        print(
            "[pp-topheads] Run run_lds.py first to produce EAP scores.",
            file=sys.stderr,
        )
        sys.exit(1)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not raw:
        print(f"[pp-topheads] ERROR: EAP file is empty: {path}", file=sys.stderr)
        sys.exit(1)
    out: Dict[Tuple[int, int], float] = {}
    for key, val in raw.items():
        l, h = key.split("_")
        out[(int(l), int(h))] = float(val)
    return out


# ---------------------------------------------------------------------------
# Output builder
# ---------------------------------------------------------------------------

def _build_result_json(
    head_scores: HeadScores,
    meta: Dict,
    model_tag: str,
    prompt_type: str,
    eap_scores: Dict[Tuple[int, int], float],
    top_heads: List[Tuple[int, int]],
    k: int,
    eap_path: Path,
) -> Dict:
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_hash = "unknown"

    try:
        import transformer_lens
        tl_version = transformer_lens.__version__
    except Exception:
        tl_version = "unknown"

    sorted_eap = sorted(eap_scores.items(), key=lambda kv: kv[1], reverse=True)
    eap_rank = {h: i + 1 for i, (h, _) in enumerate(sorted_eap)}

    selected = []
    k_ctx = min(k, len(eap_scores))
    ctx_set = set(
        h for h, _ in sorted(eap_scores.items(), key=lambda kv: kv[1], reverse=True)[:k_ctx]
    )
    for h in top_heads:
        direction = "context" if h in ctx_set else "memory"
        selected.append({
            "layer": h[0],
            "head": h[1],
            "eap_score": eap_scores.get(h, float("nan")),
            "eap_rank": eap_rank.get(h),
            "direction": direction,
            "pp_score": head_scores.get(h, float("nan")),
        })

    return {
        "model": model_tag,
        "prompt_type": prompt_type,
        "k": k,
        "source_eap_file": str(eap_path),
        "selected_heads": selected,
        "head_scores": {f"{l}_{h}": v for (l, h), v in head_scores.items()},
        "n_prompts_input": meta.get("n_prompts_input", 0),
        "n_prompts_used": meta.get("n_prompts_used", 0),
        "noise_floor": meta.get("noise_floor", {}),
        "meta": {
            "n_heads_scored": meta.get("n_heads_scored", len(top_heads)),
            "head_subset": meta.get("head_subset"),
            "n_skipped_degenerate_margin": meta.get("n_skipped_degenerate_margin", 0),
            "n_skipped_length_mismatch": meta.get("n_skipped_length_mismatch", 0),
            "n_skipped_clean_neither": meta.get("n_skipped_clean_neither", 0),
            "L_corrupt_distribution": meta.get("L_corrupt_distribution", {}),
            "L_clean_distribution": meta.get("L_clean_distribution", {}),
            "torch_version": torch.__version__,
            "transformer_lens_version": tl_version,
            "git_hash": git_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run path patching on EAP-selected top heads."
    )
    parser.add_argument("--model", default="gpt2",
                        help="TransformerLens architecture name.")
    parser.add_argument("--hf-model", default=None, dest="hf_model",
                        help="Optional HF repo for a fine-tune sharing the base arch.")
    parser.add_argument("--model-tag", default=None, dest="model_tag",
                        help="Output filename tag. Defaults to --model with '/' → '_'.")
    parser.add_argument("--prompt-type", default="substitution",
                        choices=["substitution", "coherent"], dest="prompt_type",
                        help="One prompt type per invocation. Default: substitution.")
    parser.add_argument("--eap-scores", default=None, dest="eap_scores",
                        help="Path to EAP scores JSON. Default: derived from --model-tag "
                             "and --prompt-type.")
    parser.add_argument("--k", type=int, default=10,
                        help="Per-direction top-k (ctx + mem). ~2k heads sent to PP. Default: 10.")
    parser.add_argument("--n-subset", type=int, default=None, dest="n_subset",
                        help="Optional prompt subset for smoke tests.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype", default="bfloat16",
        help="Model dtype: 'bfloat16' (default, recommended for 3-4B+) or 'float32' (GPT-2).",
    )
    args = parser.parse_args()

    model_tag = args.model_tag or args.model.replace("/", "_")

    eap_path = (
        Path(args.eap_scores) if args.eap_scores
        else RESULTS_DIR / f"lds2_{model_tag}_{args.prompt_type}_full_eap.json"
    )

    print(f"[pp-topheads] Loading EAP scores from: {eap_path}")
    eap_scores = _load_eap(eap_path)

    top_heads = select_top_heads(eap_scores, args.k)
    if not top_heads:
        print("[pp-topheads] ERROR: no heads selected (k=0 or empty EAP file).", file=sys.stderr)
        sys.exit(1)

    n_model_layers = max(l for l, _ in eap_scores) + 1
    n_model_heads = max(h for _, h in eap_scores) + 1
    for l, h in top_heads:
        if l >= n_model_layers or h >= n_model_heads:
            print(
                f"[pp-topheads] ERROR: head ({l},{h}) out of range for "
                f"{n_model_layers}L×{n_model_heads}H model.",
                file=sys.stderr,
            )
            sys.exit(1)

    sorted_by_score = sorted(eap_scores.items(), key=lambda kv: kv[1], reverse=True)
    ctx_heads_display = [(h, v) for h, v in sorted_by_score if h in set(top_heads)][:args.k]
    mem_heads_display = [(h, v) for h, v in reversed(sorted_by_score) if h in set(top_heads)][:args.k]
    print(f"[pp-topheads] Selected {len(top_heads)} heads (k={args.k} per direction):")
    print(f"  ctx (top-{args.k}): {[f'L{l}H{h}={v:.3f}' for (l,h),v in ctx_heads_display]}")
    print(f"  mem (top-{args.k}): {[f'L{l}H{h}={v:.3f}' for (l,h),v in mem_heads_display]}")

    print(f"[pp-topheads] Loading model: arch={args.model!r}, hf_model={args.hf_model!r}, dtype={args.dtype!r}")
    kwargs = {}
    if args.hf_model:
        kwargs["hf_model_name"] = args.hf_model
    model = load_model(args.model, dtype=args.dtype, **kwargs)

    prompts = load_conflict_prompts(prompt_type=args.prompt_type)
    print(f"[pp-topheads] Loaded {len(prompts)} prompts ({args.prompt_type})")

    meta: Dict = {}
    scores = run_path_patching(
        model,
        prompts,
        n_subset=args.n_subset,
        seed=args.seed,
        head_subset=top_heads,
        _meta_out=meta,
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"pp_topheads_{model_tag}_{args.prompt_type}.json"
    result = _build_result_json(
        scores, meta, model_tag, args.prompt_type,
        eap_scores, top_heads, args.k, eap_path,
    )
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[pp-topheads] Saved → {out_path}")


if __name__ == "__main__":
    main()
