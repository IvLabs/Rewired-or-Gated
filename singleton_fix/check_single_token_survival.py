"""check_single_token_survival.py — how many ParaConflict prompts survive the
single-token filter, and for which domains?

Uses GPT-2 (small, fast, already cached from earlier experiments in this
repo) as a stand-in tokenizer to answer "which prompts can we actually use"
before spending real compute on Gemma/Llama/Qwen. GPT-2's survival counts
are NOT the counts you'll get for the 3 target models -- "single-token" is a
property of a (word, tokenizer) pair, and GPT-2's BPE vocabulary differs from
Llama's BPE, Qwen's BPE, and Gemma's SentencePiece. This script exists to let
you sanity-check the loader and get an order-of-magnitude feel for the
survival rate cheaply; run the same check with each real model's tokenizer
(swap MODEL_NAME below, or import check_survival() directly) before trusting
a specific model's number.

Usage:
    ../.venv/Scripts/python.exe singleton_fix/check_single_token_survival.py
    ../.venv/Scripts/python.exe singleton_fix/check_single_token_survival.py --model gpt2

Output: prints survival counts + domain composition for both prompt types,
and writes results/single_token_survival_<model_tag>.json (new filename,
never overwrites an existing result).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from foundation import load_model
from singleton_fix.single_token_loader import load_single_token_prompts

RESULTS_DIR = Path(_ROOT) / "results"


def check_survival(model_name: str = "gpt2", model_tag: str = "gpt2") -> Dict:
    """Run the single-token filter for both prompt types under `model_name`'s
    tokenizer and return a summary dict (also written to results/)."""
    model = load_model(model_name)

    summary: Dict = {"model": model_name, "model_tag": model_tag, "prompt_types": {}}

    for ptype in ("substitution", "coherent"):
        prompts = load_single_token_prompts(model, prompt_type=ptype, use_logprob_refinement=False)
        counts: Dict[str, int] = {}
        for p in prompts:
            counts[p.domain] = counts.get(p.domain, 0) + 1
        summary["prompt_types"][ptype] = {
            "n_survived": len(prompts),
            "domain_counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        }

    out_path = RESULTS_DIR / f"single_token_survival_{model_tag}.json"
    RESULTS_DIR.mkdir(exist_ok=True)
    if out_path.exists():
        print(f"[survival] {out_path} already exists -- not overwriting. "
              f"Delete it manually first if you want to recompute.")
    else:
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[survival] saved -> {out_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model", default="gpt2",
        help="TransformerLens architecture name to load as the tokenizer stand-in. "
             "Default 'gpt2' (small, fast, no HF_TOKEN needed). Pass a real target "
             "model (e.g. 'Qwen/Qwen2.5-3B') to get that model's actual survival count.",
    )
    parser.add_argument(
        "--model-tag", default=None, dest="model_tag",
        help="Output filename tag. Defaults to --model with '/' -> '_'.",
    )
    args = parser.parse_args()
    model_tag = args.model_tag or args.model.replace("/", "_")

    summary = check_survival(model_name=args.model, model_tag=model_tag)

    print(f"\n=== single-token survival ({args.model}) ===")
    for ptype, data in summary["prompt_types"].items():
        print(f"\n[{ptype}] survived: {data['n_survived']}")
        for domain, n in data["domain_counts"].items():
            print(f"  {domain!r}: {n}")


if __name__ == "__main__":
    main()
