"""
superposition/visualize.py
----------------------------
Generates all paper figures from superposition and intervention JSON results.

Figures produced:
    results_superposition/plots/scatter_{family}.png          — base vs instruct ratio scatter
    results_superposition/plots/heatmap_{family}_base.png     — layer×head ratio heatmap
    results_superposition/plots/heatmap_{family}_instruct.png
    results_superposition/plots/role_distribution_all.png     — stacked bars across all families
    results_superposition/plots/shift_bar_all_families.png    — role shift counts per family
    results_superposition/plots/intervention_crr_{family}.png — ΔCRR under intervention

Usage:
    python superposition/visualize.py --family qwen
    python superposition/visualize.py --all
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))

RESULTS_DIR = Path(__file__).parent.parent / "results_superposition"
PLOTS_DIR   = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
FAMILIES    = ["qwen", "llama", "gemma"]

# Consistent colors across figures
ROLE_COLORS = {
    "context":      "#2196F3",   # blue
    "superposition":"#FF9800",   # orange
    "memory":       "#F44336",   # red
}
FAMILY_COLORS = {
    "qwen":  "#7C4DFF",
    "llama": "#00BCD4",
    "gemma": "#4CAF50",
}


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict | None:
    if not path.exists():
        print(f"  [missing] {path}")
        return None
    with open(path) as f:
        return json.load(f)


# ── Figure 1: Scatter (base ratio vs instruct ratio) ──────────────────────────

def plot_scatter(family: str):
    """
    Scatter plot: base ratio vs instruct ratio, one point per head.
    Color by base model role. Diagonal = no change line.
    Points above diagonal = shifted toward context after tuning.
    """
    comp = load_json(RESULTS_DIR / f"superposition_{family}_comparison_sub.json")
    if comp is None:
        return

    base_ratios = []
    inst_ratios = []
    colors      = []
    labels      = []

    for k, v in comp.items():
        br = v["base_ratio"]
        ir = v["inst_ratio"]
        # Clip extreme ratios for readability
        br = max(min(br, 10.0), -2.0)
        ir = max(min(ir, 10.0), -2.0)
        base_ratios.append(br)
        inst_ratios.append(ir)
        colors.append(ROLE_COLORS.get(v["base_role"], "#999"))
        labels.append(f"L{v['layer']}H{v['head']}")

    fig, ax = plt.subplots(figsize=(7, 7))

    # Diagonal reference line
    lim = max(max(base_ratios), max(inst_ratios)) + 0.5
    lo  = min(min(base_ratios), min(inst_ratios)) - 0.5
    ax.plot([lo, lim], [lo, lim], "k--", lw=1, alpha=0.4, label="no change (y=x)")
    ax.axhline(2.0, color="#2196F3", lw=0.8, ls=":", alpha=0.5, label="context threshold")
    ax.axhline(0.5, color="#F44336", lw=0.8, ls=":", alpha=0.5, label="memory threshold")
    ax.axvline(2.0, color="#2196F3", lw=0.8, ls=":", alpha=0.5)
    ax.axvline(0.5, color="#F44336", lw=0.8, ls=":", alpha=0.5)

    ax.scatter(base_ratios, inst_ratios, c=colors, alpha=0.75, s=60, edgecolors="white", lw=0.4)

    # Legend for roles
    for role, color in ROLE_COLORS.items():
        ax.scatter([], [], c=color, label=f"Base: {role}", s=60)
    ax.legend(fontsize=9, loc="upper left")

    ax.set_xlabel("Base model ratio (ctx_pull / mem_pull)", fontsize=11)
    ax.set_ylabel("Instruct model ratio", fontsize=11)
    ax.set_title(f"{family.upper()} — Superposition ratio: base vs instruct\n"
                 f"(points above diagonal = shifted toward context after tuning)", fontsize=11)
    ax.set_xlim(lo, lim)
    ax.set_ylim(lo, lim)

    out = PLOTS_DIR / f"scatter_{family}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 2: Heatmap (layer × head ratio) ────────────────────────────────────

def plot_heatmap(family: str, variant: str):
    """
    Layer × head heatmap of ratio values.
    Diverging colormap: red=memory, white=superposition(~1), blue=context.
    """
    res = load_json(RESULTS_DIR / f"superposition_{family}_{variant}_sub.json")
    if res is None:
        return

    n_layers = res["n_layers"]
    n_heads  = res["n_heads"]

    ratio_grid = np.zeros((n_layers, n_heads))
    for k, h in res["heads"].items():
        ratio_grid[h["layer"], h["head"]] = min(max(h["ratio"], -2.0), 8.0)

    fig, ax = plt.subplots(figsize=(max(8, n_heads * 0.7), max(6, n_layers * 0.3)))

    # Diverging colormap centered at 1.0 (superposition)
    cmap   = plt.cm.RdBu
    vmin   = 0.0
    vmax   = 4.0
    vcenter= 1.0
    norm   = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    im = ax.imshow(ratio_grid, cmap=cmap, norm=norm, aspect="auto")
    plt.colorbar(im, ax=ax, label="ratio (ctx/mem)", shrink=0.8)

    ax.set_xlabel("Head index", fontsize=11)
    ax.set_ylabel("Layer index", fontsize=11)
    ax.set_title(f"{family.upper()} {variant} — Superposition ratio per head\n"
                 f"Blue=context  White=superposition(~1)  Red=memory", fontsize=10)
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(0, n_layers, max(1, n_layers // 10)))

    out = PLOTS_DIR / f"heatmap_{family}_{variant}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 3: Role distribution across all families ──────────────────────────

def plot_role_distribution():
    """
    Stacked bar chart showing context/superposition/memory counts
    for all families × variants.
    """
    labels  = []
    ctx_counts  = []
    sup_counts  = []
    mem_counts  = []

    for family in FAMILIES:
        for variant in ["base", "instruct"]:
            res = load_json(RESULTS_DIR / f"superposition_{family}_{variant}_sub.json")
            if res is None:
                continue
            roles = Counter(h["role"] for h in res["heads"].values())
            labels.append(f"{family}\n{variant}")
            ctx_counts.append(roles["context"])
            sup_counts.append(roles["superposition"])
            mem_counts.append(roles["memory"])

    if not labels:
        print("  No data for role distribution plot.")
        return

    x   = np.arange(len(labels))
    w   = 0.6
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))

    ax.bar(x, mem_counts, w, label="memory",       color=ROLE_COLORS["memory"],       alpha=0.85)
    ax.bar(x, sup_counts, w, label="superposition",color=ROLE_COLORS["superposition"],alpha=0.85,
           bottom=mem_counts)
    top = [m + s for m, s in zip(mem_counts, sup_counts)]
    ax.bar(x, ctx_counts, w, label="context",      color=ROLE_COLORS["context"],      alpha=0.85,
           bottom=top)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Number of heads", fontsize=11)
    ax.set_title("Role distribution across model families\n"
                 "(base vs instruct — substitution prompts)", fontsize=11)
    ax.legend(fontsize=10)

    out = PLOTS_DIR / "role_distribution_all.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 4: Shift bar chart ─────────────────────────────────────────────────

def plot_shift_bars():
    """
    Grouped bar chart of role shift counts per family.
    Shows how many heads shifted memory→context, super→context, etc.
    """
    shift_types = [
        "memory→context",
        "memory→superposition",
        "superposition→context",
        "superposition→memory",
        "context→superposition",
        "context→memory",
    ]
    shift_colors = ["#4CAF50","#8BC34A","#2196F3","#FF9800","#FF5722","#F44336"]

    fam_data = {}
    for family in FAMILIES:
        comp = load_json(RESULTS_DIR / f"superposition_{family}_comparison_sub.json")
        if comp is None:
            continue
        counts = Counter(v["shift_label"] for v in comp.values())
        fam_data[family] = {st: counts.get(st, 0) for st in shift_types}

    if not fam_data:
        print("  No comparison data for shift bar plot.")
        return

    n_fam    = len(fam_data)
    n_shifts = len(shift_types)
    x        = np.arange(n_shifts)
    width    = 0.25

    fig, ax = plt.subplots(figsize=(13, 5))

    for i, (family, counts) in enumerate(fam_data.items()):
        vals   = [counts[st] for st in shift_types]
        offset = (i - n_fam / 2 + 0.5) * width
        ax.bar(x + offset, vals, width,
               label=family, color=FAMILY_COLORS.get(family, "#999"), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([st.replace("→", " →\n") for st in shift_types], fontsize=9)
    ax.set_ylabel("Number of heads", fontsize=11)
    ax.set_title("Role shifts base → instruct across model families", fontsize=11)
    ax.legend(fontsize=10)

    out = PLOTS_DIR / "shift_bar_all_families.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 5: Intervention CRR ────────────────────────────────────────────────

def plot_intervention_crr(family: str):
    """
    Bar chart of ΔCRR for the four intervention conditions.
    Green = positive (more context-following), Red = negative.
    """
    res = load_json(RESULTS_DIR / f"intervention_{family}.json")
    if res is None:
        return

    baseline = res["baseline"]["crr"]
    conds    = res["conditions"]

    names   = list(conds.keys())
    deltas  = [conds[c]["delta_crr"] for c in names]
    colors  = ["#4CAF50" if d > 0 else "#F44336" for d in deltas]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.axhline(0, color="black", lw=1.0)
    bars = ax.bar(names, deltas, color=colors, alpha=0.85, edgecolor="white", lw=0.5)

    for bar, delta in zip(bars, deltas):
        ax.text(bar.get_x() + bar.get_width() / 2,
                delta + (0.003 if delta >= 0 else -0.006),
                f"{delta:+.3f}", ha="center", va="bottom" if delta >= 0 else "top",
                fontsize=10, fontweight="bold")

    ax.set_ylabel("ΔCRR (vs baseline)", fontsize=11)
    ax.set_title(f"{family.upper()} — Intervention effect on CRR\n"
                 f"Baseline CRR = {baseline:.4f}  |  Green=expected direction",
                 fontsize=11)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=10)

    out = PLOTS_DIR / f"intervention_crr_{family}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def run_family(family: str):
    print(f"\n[visualize] {family.upper()}")
    plot_scatter(family)
    plot_heatmap(family, "base")
    plot_heatmap(family, "instruct")
    plot_intervention_crr(family)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--all", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.all:
        for fam in FAMILIES:
            run_family(fam)
        print("\n[visualize] Cross-family plots...")
        plot_role_distribution()
        plot_shift_bars()
    elif args.family:
        run_family(args.family)
        plot_role_distribution()
        plot_shift_bars()
    else:
        print("Specify --family or --all")
