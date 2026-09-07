#!/usr/bin/env python3
"""Independently validate the base-vs-instruct node-overlap table (eap scorer).

Re-derives every cell from the committed LDS JSON and checks it against the
claimed table, reimplementing lds_attribution.top() exactly:

    sorted(d.items(), key=lambda kv: -abs(kv[1]))[:k]

Result (2026-07-17): all 6 rows x 5 columns reproduce exactly.

IMPORTANT -- two Spearman conventions are in play, and they differ:
  * signed  : spearman(base_score,    inst_score)    -> 0.65-0.84
  * |score|  : spearman(|base_score|,  |inst_score|)  -> 0.82-0.91
The claimed table's Spearman column is the SIGNED one. The paper's Sec. 5.2 quotes
the |score| one. Both are printed below so the distinction stays visible.

Run: python analysis/validate_node_overlap.py
"""
import json
import os
from math import comb

from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")

FAMS = [("qwen25_3b", "Qwen"), ("llama32_3b", "Llama"), ("gemma3_4b", "Gemma")]
FORMS = ["substitution", "coherent"]

# Claimed table: (overlap@10, J@10, hypergeom p@10, J@20, Spearman[signed])
CLAIM = {
    ("Qwen", "substitution"):  (9, 0.818, 5.5e-18, 0.538, 0.654),
    ("Qwen", "coherent"):      (10, 1.000, 9.8e-22, 0.818, 0.777),
    ("Llama", "substitution"): (8, 0.667, 2.0e-15, 0.667, 0.841),
    ("Llama", "coherent"):     (8, 0.667, 2.0e-15, 0.667, 0.740),
    ("Gemma", "substitution"): (9, 0.818, 5.1e-15, 0.818, 0.769),
    ("Gemma", "coherent"):     (8, 0.667, 3.0e-12, 0.905, 0.703),
}


def load_scores(tag, var, form):
    p = os.path.join(RES, tag, f"lds2_{tag}_{var}_{form}_st_eap.json")
    return json.load(open(p))["scores"]


def top(d, k=10):
    """Exactly lds_attribution.top(): rank by |score|, take top-k as a set."""
    return {h for h, _ in sorted(d.items(), key=lambda kv: -abs(kv[1]))[:k]}


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 0.0


def hypergeom_sf(obs, N, k):
    """P(X >= obs) where X = |top-k A intersect top-k B| for random k-subsets of N."""
    return sum(comb(k, i) * comb(N - k, k - i) for i in range(obs, k + 1)) / comb(N, k)


def main():
    hdr = (f"{'Family':6} {'Form':13} {'ov@10':>7} {'J@10':>7} {'p@10':>10} "
           f"{'J@20':>7} {'signed':>7} {'|score|':>8} {'N':>5}  status")
    print(hdr)
    print("-" * len(hdr))

    all_ok = True
    for tag, name in FAMS:
        for form in FORMS:
            b = load_scores(tag, "base", form)
            i = load_scores(tag, "instruct", form)
            assert set(b) == set(i), f"head universe mismatch: {name}/{form}"
            N = len(b)
            heads = sorted(b)

            ov = len(top(b, 10) & top(i, 10))
            j10 = jaccard(top(b, 10), top(i, 10))
            j20 = jaccard(top(b, 20), top(i, 20))
            p10 = hypergeom_sf(ov, N, 10)
            signed, _ = spearmanr([b[h] for h in heads], [i[h] for h in heads])
            absol, _ = spearmanr([abs(b[h]) for h in heads], [abs(i[h]) for h in heads])

            c_ov, c_j10, c_p, c_j20, c_sp = CLAIM[(name, form)]
            ok = (ov == c_ov
                  and abs(j10 - c_j10) < 5e-4
                  and abs(j20 - c_j20) < 5e-4
                  and abs(signed - c_sp) < 5e-4
                  and abs(p10 - c_p) / c_p < 0.05)
            all_ok &= ok
            print(f"{name:6} {form:13} {ov:>4}/10 {j10:>7.3f} {p10:>10.1e} "
                  f"{j20:>7.3f} {signed:>7.3f} {absol:>8.3f} {N:>5}  "
                  f"{'OK' if ok else 'MISMATCH vs ' + str(CLAIM[(name, form)])}")

    print("-" * len(hdr))
    print("TABLE VALIDATED (all 6 rows x 5 cols reproduce)" if all_ok
          else "DISCREPANCIES FOUND")
    print("\nNote: the claimed Spearman column matches the 'signed' column, not '|score|'.")
    print("      Sec. 5.2 of the paper quotes the '|score|' range (0.82-0.91) instead.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
