#!/usr/bin/env python3
"""Generate publication figures for the BlackboxNLP paper from committed results.

Reads per-family JSON in results/ and results_superposition_v2/ and writes PDF
figures to drafts/paper/figures/. All numbers are read from committed artifacts;
the single exception is the top-50 edge Jaccard (Fig 3), which requires the
cross-model edge recompute documented in analysis/EAP_IG_analysis.md §A.1 and is
therefore taken from that (verified) table rather than recomputed from the
~10^5-edge files here. Those values are clearly flagged below.

Run: python3 analysis/make_figures.py
"""
import json
import math
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
SUP = os.path.join(ROOT, "results_superposition_v2")
OUT = os.path.join(ROOT, "drafts", "paper", "figures")
os.makedirs(OUT, exist_ok=True)

# Optional: also dump a PNG copy of each figure to $FIG_PNG_OUT for visual QA.
PNG_OUT = os.environ.get("FIG_PNG_OUT")
if PNG_OUT:
    os.makedirs(PNG_OUT, exist_ok=True)


def save(fig, name):
    """Write <name>.pdf to OUT and, if FIG_PNG_OUT is set, <name>.png for QA."""
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    if PNG_OUT:
        fig.savefig(os.path.join(PNG_OUT, name + ".png"), dpi=160)
    plt.close(fig)

# family -> (results tag, superposition dir tag, display name)
FAMS = [
    ("llama32_3b", "llama", "Llama-3.2-3B"),
    ("qwen25_3b", "qwen", "Qwen-2.5-3B"),
    ("gemma3_4b", "gemma", "Gemma-3-4B"),
]

# Colorblind-safe. Base vs instruct are one consistent hue pair everywhere.
C_BASE = "#4477AA"      # blue
C_INST = "#EE6677"      # red
ROLE_COLORS = {"context": "#4477AA", "memory": "#CCBB44", "superposition": "#AA3377"}

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
})


def load(path):
    with open(path) as f:
        return json.load(f)


def crr(tag, var, ptype):
    p = os.path.join(RES, {"llama32_3b": "llama32_3b", "qwen25_3b": "qwen25_3b",
                            "gemma3_4b": "gemma3_4b"}[tag],
                     f"crr_{tag}_{var}_{ptype}_st.json")
    return load(p)["crr_behavioral"]


def wilson_ci(p, n, z=1.96):
    """95% Wilson score interval for a binomial proportion.

    CRR is a fraction of n independent conflict prompts, so its sampling
    uncertainty is binomial. Wilson is used instead of the normal (Wald)
    interval because CRR sits near 0 or 1, where Wald over/undershoots [0,1].
    Returns (lo, hi) bounds.
    """
    if n == 0:
        return p, p
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


# ---------------------------------------------------------------------------
# Fig 2 — CRR base vs instruct x family x prompt-type (the behavioral hook)
# ---------------------------------------------------------------------------
def _crr_with_ci(tag, var, ptype):
    """Return (proportion, [dist_below, dist_above]) 95% Wilson error bars."""
    d = crr(tag, var, ptype)
    p, n = d["crr"], d["total"]
    lo, hi = wilson_ci(p, n)
    return p, (max(0.0, p - lo), max(0.0, hi - p))


def fig_crr():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
    for ax, (tag, _s, name) in zip(axes, FAMS):
        ptypes = ["substitution", "coherent"]
        x = np.arange(len(ptypes))
        w = 0.38
        base, base_err = zip(*(_crr_with_ci(tag, "base", pt) for pt in ptypes))
        inst, inst_err = zip(*(_crr_with_ci(tag, "instruct", pt) for pt in ptypes))
        base_err = np.array(base_err).T   # shape (2, n): [below; above]
        inst_err = np.array(inst_err).T
        ekw = dict(ecolor="0.25", elinewidth=0.9, capsize=2.5, capthick=0.9)
        ax.bar(x - w / 2, base, w, yerr=base_err, label="base", color=C_BASE,
               error_kw=ekw)
        ax.bar(x + w / 2, inst, w, yerr=inst_err, label="instruct", color=C_INST,
               error_kw=ekw)
        for xi in range(len(ptypes)):
            # Place value labels above the upper CI cap so they never overlap it.
            ax.text(xi - w / 2, base[xi] + base_err[1, xi] + 0.02,
                    f"{base[xi]:.2f}", ha="center", fontsize=7.5)
            ax.text(xi + w / 2, inst[xi] + inst_err[1, xi] + 0.02,
                    f"{inst[xi]:.2f}", ha="center", fontsize=7.5)
        ax.set_title(name, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(["subst.", "coherent"], fontsize=9.5)
        ax.tick_params(axis="y", labelsize=9)
        ax.set_ylim(0, 1.12)
    axes[0].set_ylabel("CRR (context-following rate)", fontsize=10)
    # Legend upper-left: the substitution bars top out well below 1.0, so this
    # corner is empty; lower-left previously collided with the base bar.
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    save(fig, "fig2_crr")


# ---------------------------------------------------------------------------
# Fig 3 — Node vs edge overlap with gating / rewiring bands
# node mean-J computed from comparison files; edge J@50 from EAP_IG §A.1 (verified)
# ---------------------------------------------------------------------------
EDGE_J50 = {"llama": 0.613, "qwen": 0.389, "gemma": 0.539}  # EAP_IG_analysis.md §A.1


def fig_node_edge():
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    names, node_j, edge_j = [], [], []
    for _tag, s, name in FAMS:
        c = load(os.path.join(SUP, s, f"comparison_{s}.json"))["_node_overlap"]
        node_j.append((c["jaccard_context_topk"] + c["jaccard_memory_topk"]) / 2)
        edge_j.append(EDGE_J50[s])
        names.append(name.replace("-3B", "").replace("-4B", ""))
    x = np.arange(len(names))
    w = 0.36
    # bands
    ax.axhspan(0.6, 1.0, color="#DDEECC", alpha=0.6, zorder=0)
    ax.axhspan(0.3, 0.6, color="#FFF3CC", alpha=0.6, zorder=0)
    ax.axhspan(0.0, 0.3, color="#F8D7DA", alpha=0.6, zorder=0)
    ax.text(2.55, 0.80, "stable", fontsize=7, color="#557733", va="center")
    ax.text(2.55, 0.45, "partial", fontsize=7, color="#997700", va="center")
    ax.text(2.55, 0.15, "rewired", fontsize=7, color="#992222", va="center")
    ax.bar(x - w / 2, node_j, w, label="node (heads)", color="#332288")
    ax.bar(x + w / 2, edge_j, w, label="edge (wiring)", color="#88CCEE")
    for xi, (n, e) in enumerate(zip(node_j, edge_j)):
        ax.text(xi - w / 2, n + 0.015, f"{n:.2f}", ha="center", fontsize=6.5)
        ax.text(xi + w / 2, e + 0.015, f"{e:.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Jaccard overlap (base vs instruct)")
    ax.set_xlim(-0.6, 3.2)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    save(fig, "fig3_node_edge")


# ---------------------------------------------------------------------------
# Fig 4 — Late-layer localization: top-20 LDS head layers, base vs instruct
# ---------------------------------------------------------------------------
def top_layers(tag, var, k=20):
    p = os.path.join(RES, tag, f"lds2_{tag}_{var}_substitution_st_eap.json")
    scores = load(p)["scores"]
    items = sorted(scores.items(), key=lambda kv: abs(kv[1]), reverse=True)[:k]
    layers = [int(h.split("_")[0]) for h, _ in items]
    depth = max(int(h.split("_")[0]) for h in scores) + 1
    return np.array(layers), depth


def fig_localization():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    for ax, (tag, _s, name) in zip(axes, FAMS):
        lb, depth = top_layers(tag, "base")
        li, _ = top_layers(tag, "instruct")
        ax.scatter(np.full_like(lb, 0, dtype=float) + np.random.uniform(-0.08, 0.08, len(lb)),
                   lb / depth, color=C_BASE, s=14, alpha=0.8, label="base")
        ax.scatter(np.full_like(li, 1, dtype=float) + np.random.uniform(-0.08, 0.08, len(li)),
                   li / depth, color=C_INST, s=14, alpha=0.8, label="instruct")
        ax.axhline(0.5, ls="--", color="gray", lw=0.8)
        ax.set_title(f"{name}\n(L={depth})", fontsize=8.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["base", "inst"])
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("normalized layer depth")
    axes[0].text(-0.45, 0.52, "L/2", fontsize=7, color="gray")
    fig.tight_layout()
    save(fig, "fig4_localization")


# ---------------------------------------------------------------------------
# Fig 5 — Causal ablation directional grid (36/36 sign matches)
# ---------------------------------------------------------------------------
COND_ORDER = ["amplify_context", "suppress_context", "suppress_context_zero",
              "amplify_memory", "suppress_memory", "suppress_memory_zero"]
# Two-line labels: top line = intervention on the head activations, bottom line
# = which head set it targets. "mean" = replace with mean activation (soft
# ablation); "zero" = zero out the heads (harsher ablation).
COND_LABEL = ["amplify\ncontext", "mean-ablate\ncontext", "zero-ablate\ncontext",
              "amplify\nmemory", "mean-ablate\nmemory", "zero-ablate\nmemory"]


def fig_ablation():
    rows, rowlabels = [], []
    matched = []
    for _tag, s, name in FAMS:
        for var in ("base", "instruct"):
            d = load(os.path.join(SUP, s, f"ablation_{s}_{var}.json"))
            cond = d["conditions"]
            rows.append([cond[c]["delta_crr"] for c in COND_ORDER])
            matched.append([cond[c]["matched_expectation"] for c in COND_ORDER])
            rowlabels.append(f"{name.split('-')[0]} {var[:4]}")
    M = np.array(rows)
    # Taller, larger cells and text so the grid stays legible at column width.
    fig, ax = plt.subplots(figsize=(5.9, 3.8))
    vmax = np.abs(M).max()
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            mark = "✓" if matched[i][j] else "✗"
            ax.text(j, i, f"{M[i, j]:+.2f}\n{mark}", ha="center", va="center",
                    fontsize=7, color="black")
    ax.set_xticks(range(len(COND_LABEL)))
    ax.set_xticklabels(COND_LABEL, rotation=0, ha="center", fontsize=7.5)
    ax.set_yticks(range(len(rowlabels)))
    ax.set_yticklabels(rowlabels, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("ΔCRR", fontsize=8)
    fig.tight_layout()
    save(fig, "fig5_ablation")


# ---------------------------------------------------------------------------
# Fig 6 — Superposition context-index scatter, base vs instruct
# ---------------------------------------------------------------------------
def fig_superposition():
    # Sized to span both columns (\textwidth); larger panels than the old
    # single-column version so points and the diagonal are readable.
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharex=True, sharey=True)
    for ax, (_tag, s, name) in zip(axes, FAMS):
        heads = load(os.path.join(SUP, s, f"comparison_{s}.json"))["heads"]
        bi = [h["base_index"] for h in heads.values()]
        ii = [h["inst_index"] for h in heads.values()]
        col = [ROLE_COLORS.get(h["base_role"], "gray") for h in heads.values()]
        flipped = [h["role_changed"] for h in heads.values()]
        ax.plot([-1, 1], [-1, 1], ls="--", color="gray", lw=0.7, zorder=0)
        ax.scatter(bi, ii, c=col, s=20, alpha=0.85, edgecolors="none")
        # ring role-flips
        fb = [b for b, f in zip(bi, flipped) if f]
        fi = [i for i, f in zip(ii, flipped) if f]
        ax.scatter(fb, fi, facecolors="none", edgecolors="black", s=52, lw=0.9)
        n_flip = sum(flipped)
        ax.set_title(f"{name}  ({n_flip} role flips)", fontsize=9)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("base index")
    axes[0].set_ylabel("instruct index")
    handles = [Patch(color=c, label=r) for r, c in ROLE_COLORS.items()]
    handles.append(plt.Line2D([], [], marker="o", ls="", mfc="none",
                              mec="black", label="role flip"))
    # Figure-level legend below the panels: never overlaps the scatter.
    fig.legend(handles=handles, frameon=False, fontsize=8, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "fig6_superposition")


if __name__ == "__main__":
    np.random.seed(0)
    fig_crr(); print("wrote fig2_crr.pdf")
    fig_node_edge(); print("wrote fig3_node_edge.pdf")
    fig_localization(); print("wrote fig4_localization.pdf")
    fig_ablation(); print("wrote fig5_ablation.pdf")
    fig_superposition(); print("wrote fig6_superposition.pdf")
    print("all figures ->", OUT)
