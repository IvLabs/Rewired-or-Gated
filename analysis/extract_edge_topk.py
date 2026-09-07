#!/usr/bin/env python3
"""Extract slim top-k edge summaries from the full EAP-IG edge-attribution files.

The raw ``eapig_*_substitution_st_full.json`` graphs are ~10^5 edges each
(~213 MB total across families) and are impractical to ship in a review
artifact. Every edge-level number the analysis reports — the top-{50,100,500}
and full-top-1000 Jaccard overlaps and the top-1000 Spearman correlation
(analysis/compute_metrics.py section 3) — is computed only from the ``top_k_edges``
list (the 1000 highest-magnitude edges) plus the scalar ``n_edges``. This script
copies just those fields into ``*_edges_topk.json`` (~90 KB each), preserving
exact reproducibility of the edge metrics without the bulk.

Run: python3 analysis/extract_edge_topk.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FAMS = ["llama32_3b", "qwen25_3b", "gemma3_4b"]
KEEP = ["model", "prompt_type", "method", "n_edges", "top_k_edges"]


def main():
    for fam in FAMS:
        for var in ("base", "instruct"):
            src = os.path.join(RES, fam,
                               f"eapig_{fam}_{var}_substitution_st_full.json")
            dst = os.path.join(RES, fam,
                               f"eapig_{fam}_{var}_substitution_st_edges_topk.json")
            with open(src) as f:
                d = json.load(f)
            slim = {k: d[k] for k in KEEP if k in d}
            with open(dst, "w") as f:
                json.dump(slim, f)
            kb = os.path.getsize(dst) / 1024
            print(f"{fam:11s} {var:8s} -> {os.path.basename(dst)}  ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
