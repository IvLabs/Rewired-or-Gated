"""Dump N examples per prompt_type to a text file for manual inspection:
calls build_contrast_pair to generate the conflict text and its substitution-based
counterfactual, then reports the prefix+suffix window (common tokens before and
after the substituted span) that score_prompt uses for delta/gradient contraction.

Usage:
    cd BBNLP
    python -m eap_ig.dump_examples --prompt_type substitution --n 10
    python -m eap_ig.dump_examples --prompt_type coherent --n 10
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from eap_ig.eap_ig import _common_prefix_len, _common_suffix_len, build_contrast_pair
from foundation import load_conflict_prompts, load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_type", choices=["substitution", "coherent"], required=True)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out_path = args.out or os.path.join(_ROOT, "results", f"dump_{args.prompt_type}.txt")

    print(f"[dump] loading gpt2 tokenizer...")
    model = load_model("gpt2")

    print(f"[dump] loading {args.prompt_type} prompts...")
    prompts = load_conflict_prompts(prompt_type=args.prompt_type)[: args.n]

    lines = []
    n_zero = 0
    for i, p in enumerate(prompts):
        pair = build_contrast_pair(model, p)
        if pair is None:
            n_zero += 1
            continue
        main_tokens, base_tokens = pair[0][0], pair[1][0]
        L_main, L_base = main_tokens.shape[0], base_tokens.shape[0]
        L_pre = _common_prefix_len(main_tokens, base_tokens)
        L_suf = _common_suffix_len(main_tokens, base_tokens)
        # Apply the same overlap guard as score_prompt in eap_ig.py
        shorter = min(L_main, L_base)
        if L_pre + L_suf > shorter:
            L_suf = shorter - L_pre
        L_real = L_pre + L_suf
        if L_real == 0:
            n_zero += 1

        main_toks_str = [model.tokenizer.decode([t]) for t in main_tokens.tolist()]
        base_toks_str = [model.tokenizer.decode([t]) for t in base_tokens.tolist()]

        lines.append(f"=== example {i} (domain={p.domain}) ===")
        lines.append(f"TEXT (conflict):  {p.text!r}")
        lines.append(f"CLEAN_TEXT:       {p.clean_text!r}")
        lines.append(f"memory_answer={p.memory_answer!r}  context_answer={p.context_answer!r}")
        lines.append(f"L_main={L_main}  L_base={L_base}")
        lines.append(f"main tokens : {main_toks_str}")
        lines.append(f"base tokens : {base_toks_str}")
        suffix_part = main_toks_str[-L_suf:] if L_suf > 0 else []
        lines.append(f"REAL window used by score_prompt (prefix_len={L_pre} + suffix_len={L_suf} = {L_real}): "
                     f"{main_toks_str[:L_pre] + suffix_part!r}")
        lines.append("")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[dump] wrote {len(prompts)} examples -> {out_path}")
    print(f"[dump] prompts where the real common-suffix window is EMPTY "
          f"(would be skipped by score_prompt): {n_zero}/{len(prompts)}")


if __name__ == "__main__":
    main()
