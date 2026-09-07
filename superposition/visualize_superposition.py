#!/usr/bin/env python3
"""
visualize_superposition.py
===========================
Generates all paper figures from results_superposition/ JSON files.

Run AFTER superposition_analysis.py has completed.

Figures produced in results_superposition/plots/:
  scatter_{family}.png              — base vs instruct ratio scatter
  heatmap_{family}_{variant}.png    — ratio heatmap over target heads
  role_distribution_all.png         — stacked bars across families
  shift_bar_all_families.png        — role shift counts
  intervention_crr_{family}.png     — ΔCRR under intervention

Usage:
  python visualize_superposition.py --family qwen
  python visualize_superposition.py --all
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

RESULTS_DIR = Path("results_superposition")
PLOTS_DIR   = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

FAMILIES = ["qwen", "llama", "gemma"]

ROLE_COLORS = {
    "context":       "#2196F3",
    "superposition": "#FF9800",
    "memory":        "#F44336",
}
FAMILY_COLORS = {
    "qwen":  "#7C4DFF",
    "llama": "#00BCD4",
    "gemma": "#4CAF50",
}

SHIFT_TYPES = [
    "memory→context", "memory→superposition",
    "superposition→context", "superposition→memory",
    "context→superposition", "context→memory",
]


def load_json(path: Path) -> dict | None:
    if not path.exists():
        print(f"  [missing] {path}")
        return None
    with open(path) as f:
        return json.load(f)


# ── Figure 1: Scatter ──────────────────────────────────────────────────────────

def plot_scatter(family: str):
    comp = load_json(RESULTS_DIR / family / f"superposition_{family}_comparison_substitution.json")
    if comp is None: return

    base_r, inst_r, colors, annots = [], [], [], []
    for k, v in comp.items():
        if k == "_meta": continue
        br = max(min(v["base_ratio"], 12.0), -3.0)
        ir = max(min(v["inst_ratio"], 12.0), -3.0)
        base_r.append(br); inst_r.append(ir)
        colors.append(ROLE_COLORS.get(v["base_role"], "#999"))
        annots.append(k)

    fig, ax = plt.subplots(figsize=(7, 7))
    lo  = min(min(base_r), min(inst_r)) - 0.5
    hi  = max(max(base_r), max(inst_r)) + 0.5

    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.4, label="no change (y=x)")
    ax.axhline(2.0, color="#2196F3", lw=0.8, ls=":", alpha=0.5)
    ax.axhline(0.5, color="#F44336", lw=0.8, ls=":", alpha=0.5)
    ax.axvline(2.0, color="#2196F3", lw=0.8, ls=":", alpha=0.5)
    ax.axvline(0.5, color="#F44336", lw=0.8, ls=":", alpha=0.5)

    ax.scatter(base_r, inst_r, c=colors, alpha=0.8, s=80,
               edgecolors="white", lw=0.5, zorder=5)

    for role, col in ROLE_COLORS.items():
        ax.scatter([], [], c=col, s=60, label=f"Base: {role}")
    ax.legend(fontsize=9, loc="upper left")

    ax.set_xlabel("Base model ratio (ctx_pull / mem_pull)", fontsize=11)
    ax.set_ylabel("Instruct model ratio", fontsize=11)
    ax.set_title(f"{family.upper()} — Superposition ratio: base vs instruct\n"
                 f"Points above diagonal = more context-biased after tuning", fontsize=10)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

    out = PLOTS_DIR / f"scatter_{family}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 2: Ratio bar per head ───────────────────────────────────────────────

def plot_ratio_bars(family: str, variant: str):
    """
    Bar chart of ratio per head, colored by role.
    More informative than a heatmap when scoring only 20 specific heads.
    """
    res = load_json(RESULTS_DIR / family / f"superposition_{family}_{variant}_substitution.json")
    if res is None: return

    heads   = res["heads"]
    sorted_heads = sorted(heads.items(), key=lambda x: x[1]["ratio"], reverse=True)
    keys    = [k for k, _ in sorted_heads]
    ratios  = [v["ratio"] for _, v in sorted_heads]
    cols    = [ROLE_COLORS[v["role"]] for _, v in sorted_heads]

    fig, ax = plt.subplots(figsize=(max(10, len(keys)*0.6), 5))
    ax.bar(keys, ratios, color=cols, alpha=0.85, edgecolor="white", lw=0.5)
    ax.axhline(2.0, color="#2196F3", lw=1, ls="--", alpha=0.6, label="context threshold (2.0)")
    ax.axhline(0.5, color="#F44336", lw=1, ls="--", alpha=0.6, label="memory threshold (0.5)")
    ax.axhline(0.0, color="black",   lw=0.5)

    for role, col in ROLE_COLORS.items():
        ax.bar([], [], color=col, label=role)
    ax.legend(fontsize=9)

    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("ratio = ctx_pull / mem_pull", fontsize=11)
    ax.set_title(f"{family.upper()} {variant} — Superposition ratio per head\n"
                 f"(Blue=context  Orange=superposition  Red=memory)", fontsize=10)

    out = PLOTS_DIR / f"ratio_bars_{family}_{variant}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 3: Role distribution ────────────────────────────────────────────────

def plot_role_distribution():
    labels, ctx_c, sup_c, mem_c = [], [], [], []
    for family in FAMILIES:
        for variant in ["base", "instruct"]:
            res = load_json(RESULTS_DIR / family / f"superposition_{family}_{variant}_substitution.json")
            if res is None: continue
            heads = res["heads"]
            roles = Counter(h["role"] for h in heads.values())
            labels.append(f"{family}\n{variant}")
            ctx_c.append(roles["context"])
            sup_c.append(roles["superposition"])
            mem_c.append(roles["memory"])

    if not labels: return

    x, w = np.arange(len(labels)), 0.6
    fig, ax = plt.subplots(figsize=(max(8, len(labels)*1.3), 5))
    ax.bar(x, mem_c, w, label="memory",       color=ROLE_COLORS["memory"],       alpha=0.85)
    ax.bar(x, sup_c, w, label="superposition",color=ROLE_COLORS["superposition"],alpha=0.85,
           bottom=mem_c)
    ax.bar(x, ctx_c, w, label="context",      color=ROLE_COLORS["context"],      alpha=0.85,
           bottom=[m+s for m,s in zip(mem_c, sup_c)])

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Heads (top-20 per model)", fontsize=11)
    ax.set_title("Role distribution: base vs instruct — top-20 LDS heads", fontsize=11)
    ax.legend(fontsize=10)

    out = PLOTS_DIR / "role_distribution_all.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 4: Shift bars ───────────────────────────────────────────────────────

def plot_shift_bars():
    fam_data = {}
    for family in FAMILIES:
        comp = load_json(RESULTS_DIR / family / f"superposition_{family}_comparison_substitution.json")
        if comp is None: continue
        counts = Counter(v["shift_label"] for k,v in comp.items() if k != "_meta")
        fam_data[family] = {st: counts.get(st, 0) for st in SHIFT_TYPES}

    if not fam_data: return
    n_fam, n_shifts = len(fam_data), len(SHIFT_TYPES)
    x, w = np.arange(n_shifts), 0.25

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (family, counts) in enumerate(fam_data.items()):
        vals   = [counts[st] for st in SHIFT_TYPES]
        offset = (i - n_fam/2 + 0.5) * w
        ax.bar(x + offset, vals, w,
               label=family, color=FAMILY_COLORS.get(family,"#999"), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([st.replace("→", " →\n") for st in SHIFT_TYPES], fontsize=9)
    ax.set_ylabel("Number of heads", fontsize=11)
    ax.set_title("Role shifts base→instruct across model families", fontsize=11)
    ax.legend(fontsize=10)

    out = PLOTS_DIR / "shift_bar_all_families.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 5: Intervention CRR ────────────────────────────────────────────────

def plot_intervention_crr(family: str):
    res = load_json(RESULTS_DIR / family / f"intervention_{family}_substitution.json")
    if res is None: return

    baseline = res["baseline_crr"]
    conds    = res["conditions"]
    names    = list(conds.keys())
    deltas   = [conds[c]["delta_crr"] for c in names]
    correct  = [conds[c]["expected_positive"] for c in names]
    cols     = ["#4CAF50" if c else "#FF5722" for c in correct]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0, color="black", lw=1.0)
    bars = ax.bar(names, deltas, color=cols, alpha=0.85, edgecolor="white", lw=0.5)

    for bar, delta in zip(bars, deltas):
        ypos = delta + (0.003 if delta >= 0 else -0.006)
        va   = "bottom" if delta >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f"{delta:+.3f}", ha="center", va=va, fontsize=10, fontweight="bold")

    ax.set_ylabel("ΔCRR (vs baseline)", fontsize=11)
    ax.set_title(f"{family.upper()} — Intervention effect on CRR\n"
                 f"Baseline CRR={baseline:.4f}  |  Green=expected direction", fontsize=10)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=10)

    out = PLOTS_DIR / f"intervention_crr_{family}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 6: Delta ratio per head (base→instruct shift magnitude) ─────────────

def plot_delta_ratio(family: str):
    comp = load_json(RESULTS_DIR / family / f"superposition_{family}_comparison_substitution.json")
    if comp is None: return

    items = [(k, v) for k, v in comp.items() if k != "_meta"]
    items.sort(key=lambda x: x[1]["ratio_shift"], reverse=True)

    keys   = [k for k, _ in items]
    shifts = [v["ratio_shift"] for _, v in items]
    cols   = ["#2196F3" if s > 0 else "#F44336" for s in shifts]

    fig, ax = plt.subplots(figsize=(max(10, len(keys)*0.6), 5))
    ax.bar(keys, shifts, color=cols, alpha=0.85, edgecolor="white", lw=0.5)
    ax.axhline(0, color="black", lw=1.0)
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Δ ratio (instruct − base)", fontsize=11)
    ax.set_title(f"{family.upper()} — Ratio shift per head (base → instruct)\n"
                 f"Blue=more context-biased after tuning  Red=more memory-biased", fontsize=10)

    out = PLOTS_DIR / f"delta_ratio_{family}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out}")


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_family(family: str):
    print(f"\n[visualize] {family.upper()}")
    plot_scatter(family)
    plot_ratio_bars(family, "base")
    plot_ratio_bars(family, "instruct")
    plot_delta_ratio(family)
    plot_intervention_crr(family)


def parse_args():
    p = argparse.ArgumentParser(description="Visualize superposition results")
    p.add_argument("--family", choices=FAMILIES)
    p.add_argument("--all", action="store_true")
    return p.parse_args()


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
