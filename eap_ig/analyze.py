"""Analyse an EAP-IG result JSON: top edges, score distribution, and the
late-layer concentration of edges feeding the logits (the key sanity signal).

Usage:
    python -m eap_ig.analyze [path]
    python -m eap_ig.analyze results/eapig_gpt2_substitution.json
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Dict, Optional, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_LAYER_RE = re.compile(r"blocks\.(\d+)\.")


def _layer_of(label: str) -> Optional[int]:
    """Parse the layer index out of a node label, or None for input/logits."""
    m = _LAYER_RE.search(label)
    return int(m.group(1)) if m else None


def _src_kind(label: str) -> str:
    if label == "input":
        return "input"
    if "attn.hook_result" in label:
        return "attn_head"
    if "hook_mlp_out" in label:
        return "mlp"
    return "other"


def _parse_edges(d: Dict) -> Dict[Tuple[str, str], float]:
    """Reconstruct {(src, dst): score} from the serialised 'A -> B' keys."""
    out: Dict[Tuple[str, str], float] = {}
    for k, v in d["edge_scores"].items():
        src, dst = k.split(" -> ")
        out[(src, dst)] = v
    return out


def analyze(path: str) -> None:
    with open(path) as f:
        d = json.load(f)

    print("=" * 70)
    print(f"EAP-IG ANALYSIS  -  {os.path.basename(path)}")
    print("=" * 70)
    print(f"model={d['model']}  prompt_type={d['prompt_type']}  "
          f"method={d['method']}  granularity={d.get('granularity')}  "
          f"n_steps={d.get('n_ig_steps')}")
    print(f"prompts used={d.get('n_prompts_used')}  "
          f"pair_construction={d.get('pair_construction')}  "
          f"edges={d.get('n_edges')}")

    edges = _parse_edges(d)
    scores = list(edges.values())
    n = len(scores)
    pos = sum(1 for s in scores if s > 0)
    neg = sum(1 for s in scores if s < 0)
    mx = max(scores, key=abs)
    abs_sorted = sorted(edges.items(), key=lambda kv: abs(kv[1]), reverse=True)

    # --- score distribution ---
    print("\n--- score distribution ---")
    print(f"n_edges={n}  positive={pos} ({100*pos/n:.1f}%)  "
          f"negative={neg} ({100*neg/n:.1f}%)")
    print(f"max |score|={abs(mx):.4f}  mean|score|={sum(abs(s) for s in scores)/n:.4f}")

    # --- top edges overall ---
    print("\n--- top 15 edges by |score| ---")
    for (s, dd), v in abs_sorted[:15]:
        print(f"  {v:+8.4f}   {s:34s} -> {dd}")

    # --- edges into logits: late-layer concentration (the sanity signal) ---
    logit_edges = [((s, dd), v) for (s, dd), v in abs_sorted if dd == "logits"]
    print(f"\n--- edges into LOGITS (top {min(15, len(logit_edges))} of "
          f"{len(logit_edges)}) ---")
    n_layers = 12
    upper = lower = 0
    for (s, dd), v in logit_edges[:15]:
        l = _layer_of(s)
        print(f"  {v:+8.4f}   {s}" + (f"   [layer {l}]" if l is not None else ""))
    for (s, dd), v in logit_edges:
        l = _layer_of(s)
        if l is None:
            continue
        if l >= n_layers // 2:
            upper += abs(v)
        else:
            lower += abs(v)
    tot = upper + lower
    if tot > 0:
        print(f"\n  late-layer concentration (sources -> logits): "
              f"upper-half |score| = {100*upper/tot:.1f}%  "
              f"lower-half = {100*lower/tot:.1f}%")
        print("  (Jin et al. 2024: conflict-resolution heads live in upper layers - "
              "expect upper-half dominance.)")

    # --- breakdown by source kind ---
    print("\n--- total |score| by source kind ---")
    by_kind: Dict[str, float] = {}
    for (s, dd), v in edges.items():
        by_kind[_src_kind(s)] = by_kind.get(_src_kind(s), 0.0) + abs(v)
    for kind, tot in sorted(by_kind.items(), key=lambda kv: -kv[1]):
        print(f"  {kind:10s} {tot:10.2f}")
    print("=" * 70)


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "results/eapig_gpt2_substitution.json"
    if not os.path.isabs(path):
        path = os.path.join(_ROOT, path)
    if not os.path.exists(path):
        print(f"[analyze] not found: {path}")
        print("Run a result first, e.g.: python -m eap_ig.eap_ig --model gpt2 "
              "--prompt_type substitution")
        sys.exit(1)
    analyze(path)


if __name__ == "__main__":
    main()
