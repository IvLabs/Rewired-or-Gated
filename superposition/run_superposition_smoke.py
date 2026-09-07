"""
superposition/run_superposition_smoke.py
-----------------------------------------
Smoke test runner for superposition analysis.
Runs superposition calculations on:
1. All heads (all L x H heads) for base and instruct.
2. Top-20 context and top-20 memory heads extracted from results.

Runs on a tiny subset of prompts (default 8) to complete quickly.
Saves JSONs and plots under results_superposition/{family}/smoke/.
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime

# Set logging levels
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Set paths
sys.path.insert(0, str(Path(__file__).parent))
import gc
from load_models import FAMILIES, load_tl_model, load_tokenizer
from load_dataset import load_substitution_prompts
from logit_utils import get_ln_scale
from superposition import compare_models
from transformer_lens import HookedTransformer
from intervention import make_scale_hooks, compute_crr

# Setup HF Token
from huggingface_hub import login as hf_login
_hf_token = os.environ.get("HF_TOKEN", "")
if _hf_token:
    try:
        hf_login(token=_hf_token, add_to_git_credential=False)
    except Exception as e:
        log.warning(f"HF login failed: {e}")

RESULTS_DIR = Path(__file__).parent.parent / "results_superposition"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Extractor for Top Heads from results ───────────────────────────────────────

def get_top_heads_from_results(family: str, variant: str, k: int = 20) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Extracts the top-k context heads (most positive EAP scores) and
    top-k memory/parametric heads (most negative EAP scores) from the EAP file in results.
    """
    # Mapping family names to results folder names
    fam_map = {
        "qwen": "qwen25_3b",
        "llama": "llama32_3b",
        "gemma": "gemma3_4b"
    }
    tag_map = {
        "qwen": "qwen25_3b",
        "llama": "llama32_3b",
        "gemma": "gemma3_4b"
    }
    
    fam_dir = fam_map.get(family, family)
    tag = tag_map.get(family, family)
    
    eap_file = Path(__file__).parent.parent / "results" / fam_dir / f"lds2_{tag}_{variant}_substitution_st_eap.json"
    
    if not eap_file.exists():
        log.warning(f"EAP file not found: {eap_file}. Falling back to default registry.")
        from superposition_analysis import HEAD_REGISTRY
        ctx = HEAD_REGISTRY.get((family, variant, "context"), [])[:k]
        mem = HEAD_REGISTRY.get((family, variant, "memory"), [])[:k]
        return ctx, mem

    try:
        with open(eap_file) as f:
            data = json.load(f)
        scores = data["scores"]
        # Sort items by EAP score descending
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Context heads: top positive EAP scores
        ctx_heads = []
        for key, val in sorted_items:
            l, h = map(int, key.split("_"))
            ctx_heads.append((l, h))
            if len(ctx_heads) >= k:
                break
                
        # Memory heads: bottom negative EAP scores
        mem_heads = []
        for key, val in reversed(sorted_items):
            l, h = map(int, key.split("_"))
            mem_heads.append((l, h))
            if len(mem_heads) >= k:
                break
                
        log.info(f"Loaded top-{k} ctx/mem heads from EAP: {eap_file}")
        return ctx_heads, mem_heads
    except Exception as e:
        log.error(f"Error reading EAP file: {e}. Falling back to default registry.")
        from superposition_analysis import HEAD_REGISTRY
        ctx = HEAD_REGISTRY.get((family, variant, "context"), [])[:k]
        mem = HEAD_REGISTRY.get((family, variant, "memory"), [])[:k]
        return ctx, mem


# ── Superposition Computation ──────────────────────────────────────────────────

def compute_superposition_for_heads(
    model:        any,
    prompts:      list,
    target_heads: list[tuple[int, int]],
    norm_type:    str,
    device:       str,
    label:        str = ""
) -> dict:
    """Computes superposition metrics for the specified list of target heads."""
    n_layers = model.cfg.n_layers
    n_heads  = model.cfg.n_heads
    W_O      = model.W_O
    W_U      = model.W_U

    ctx_acc = {h: 0.0 for h in target_heads}
    mem_acc = {h: 0.0 for h in target_heads}
    n_valid = 0

    needed_layers = set(l for l, h in target_heads)
    needed_hooks  = {f"blocks.{l}.attn.hook_z" for l in needed_layers}
    needed_hooks.add(f"blocks.{n_layers-1}.hook_resid_post")
    names_filter = lambda name: name in needed_hooks

    for row in tqdm(prompts, desc=f"  Superposition {label}", unit="prompt"):
        tokens = model.to_tokens(row.prompt_conflict)
        if tokens.shape[1] > model.cfg.n_ctx:
            tokens = tokens[:, -model.cfg.n_ctx:]

        try:
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens,
                    names_filter=names_filter,
                    return_type=None,
                )
        except Exception as e:
            log.warning(f"Forward pass failed: {e}")
            continue

        ln_scale = get_ln_scale(model, cache, norm_type)
        mem_tok  = row.memory_token_id
        ctx_tok  = row.context_token_id

        for (l, h) in target_heads:
            hook_name = f"blocks.{l}.attn.hook_z"
            z         = cache[hook_name][0, -1, h, :]
            out_h     = z @ W_O[l, h]
            scaled    = out_h * ln_scale.squeeze()

            ctx_acc[(l, h)] += (scaled @ W_U[:, ctx_tok]).item()
            mem_acc[(l, h)] += (scaled @ W_U[:, mem_tok]).item()

        del cache
        n_valid += 1

    if n_valid == 0:
        raise RuntimeError("No prompts were successfully processed.")

    # Average over prompts
    for h in target_heads:
        ctx_acc[h] /= n_valid
        mem_acc[h] /= n_valid

    # Build output
    heads_out = {}
    for (l, h) in target_heads:
        mc = mem_acc[(l, h)]
        cc = ctx_acc[(l, h)]
        if abs(mc) > 1e-8:
            ratio = cc / mc
        else:
            ratio = 1e6 if cc > 0 else -1e6
            
        role = "superposition"
        if ratio > 2.0:
            role = "context"
        elif ratio < 0.5:
            role = "memory"

        heads_out[f"L{l}H{h}"] = {
            "layer": l,
            "head": h,
            "mem_pull": float(mc),
            "ctx_pull": float(cc),
            "ratio": float(ratio),
            "role": role,
        }

    return {
        "n_prompts": n_valid,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "thresholds": {"context": 2.0, "memory": 0.5},
        "heads": heads_out,
    }


# ── Intervention Experiment ────────────────────────────────────────────────────

def run_smoke_intervention(
    model: HookedTransformer,
    prompts: list,
    ctx_heads: list[tuple[int, int]],
    mem_heads: list[tuple[int, int]],
    family: str,
    variant: str
) -> dict:
    """Runs a 4-condition intervention experiment using a subset of EAP context/memory heads."""
    # Take top 5 from our top-20 lists
    c_heads = ctx_heads[:5]
    m_heads = mem_heads[:5]

    log.info(f"Intervention heads for smoke test: ctx={c_heads}, mem={m_heads}")

    # Compute baseline
    baseline = compute_crr(model, prompts, hooks=None, label=f"baseline")
    
    conditions = [
        ("amplify_context", c_heads, 2.0),
        ("suppress_context", c_heads, 0.0),
        ("amplify_memory", m_heads, 2.0),
        ("suppress_memory", m_heads, 0.0),
    ]

    cond_results = {}
    for cond_name, heads, scale in conditions:
        log.info(f"  Condition: {cond_name} (scale={scale})")
        hooks = make_scale_hooks(heads, scale)
        res = compute_crr(model, prompts, hooks=hooks, label=cond_name)
        delta = res["crr"] - baseline["crr"]
        
        # Check expected direction
        expected_pos = cond_name in ("amplify_context", "suppress_memory")
        correct = (delta > 0) if expected_pos else (delta < 0)
        if delta == 0:
            correct = True  # baseline is same (common in small subsets)

        cond_results[cond_name] = {
            "scale": scale,
            "crr": res["crr"],
            "delta_crr": float(delta),
            "expected_positive": correct,
        }

    return {
        "family": family,
        "variant": variant,
        "baseline_crr": baseline["crr"],
        "conditions": cond_results,
    }


# ── Plotting Utilities ─────────────────────────────────────────────────────────

def save_plots(family: str, smoke_dir: Path):
    """Adapt the plotting logic from visualize_superposition.py to the smoke directory."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    ROLE_COLORS = {"context": "#2196F3", "superposition": "#FF9800", "memory": "#F44336"}
    plots_dir = smoke_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # 1. Scatter Plot (Top 40 vs All Heads)
    for suffix in ["all_heads", "top40"]:
        comp_file = smoke_dir / f"superposition_{family}_comparison_{suffix}.json"
        if not comp_file.exists():
            continue
        try:
            with open(comp_file) as f:
                comp = json.load(f)
            
            base_r, inst_r, colors = [], [], []
            for k, v in comp.items():
                if k == "_meta": continue
                br = max(min(v["base_ratio"], 12.0), -3.0)
                ir = max(min(v["inst_ratio"], 12.0), -3.0)
                base_r.append(br)
                inst_r.append(ir)
                colors.append(ROLE_COLORS.get(v["base_role"], "#999"))
            
            fig, ax = plt.subplots(figsize=(6, 6))
            lo, hi = -3.5, 12.5
            ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.4, label="y=x")
            ax.axhline(2.0, color="#2196F3", lw=0.8, ls=":", alpha=0.5)
            ax.axhline(0.5, color="#F44336", lw=0.8, ls=":", alpha=0.5)
            ax.axvline(2.0, color="#2196F3", lw=0.8, ls=":", alpha=0.5)
            ax.axvline(0.5, color="#F44336", lw=0.8, ls=":", alpha=0.5)
            
            ax.scatter(base_r, inst_r, c=colors, alpha=0.8, s=60, edgecolors="white", lw=0.5, zorder=5)
            for role, col in ROLE_COLORS.items():
                ax.scatter([], [], c=col, s=40, label=f"Base: {role}")
            ax.legend(fontsize=8, loc="upper left")
            ax.set_xlabel("Base model ratio")
            ax.set_ylabel("Instruct model ratio")
            ax.set_title(f"{family.upper()} ({suffix}) — Smoke Scatter")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            
            out = plots_dir / f"scatter_{family}_{suffix}.png"
            fig.savefig(out, dpi=120, bbox_inches="tight")
            plt.close(fig)
            log.info(f"Saved plot: {out}")
        except Exception as e:
            log.error(f"Error plotting scatter for {suffix}: {e}")

    # 2. Ratio Bars (Top 40)
    for variant in ["base", "instruct"]:
        res_file = smoke_dir / f"superposition_{family}_{variant}_top40.json"
        if not res_file.exists():
            continue
        try:
            with open(res_file) as f:
                res = json.load(f)
            heads = res["heads"]
            sorted_heads = sorted(heads.items(), key=lambda x: x[1]["ratio"], reverse=True)
            keys = [k for k, _ in sorted_heads]
            ratios = [v["ratio"] for _, v in sorted_heads]
            cols = [ROLE_COLORS[v["role"]] for _, v in sorted_heads]
            
            fig, ax = plt.subplots(figsize=(max(8, len(keys)*0.3), 4))
            ax.bar(keys, ratios, color=cols, alpha=0.85, edgecolor="white", lw=0.5)
            ax.axhline(2.0, color="#2196F3", lw=1, ls="--", alpha=0.6)
            ax.axhline(0.5, color="#F44336", lw=1, ls="--", alpha=0.6)
            ax.axhline(0.0, color="black", lw=0.5)
            
            ax.set_xticklabels(keys, rotation=60, ha="right", fontsize=6)
            ax.set_ylabel("ratio")
            ax.set_title(f"{family.upper()} {variant} — Smoke Top-40 Ratios")
            
            out = plots_dir / f"ratio_bars_{family}_{variant}_top40.png"
            fig.savefig(out, dpi=120, bbox_inches="tight")
            plt.close(fig)
            log.info(f"Saved plot: {out}")
        except Exception as e:
            log.error(f"Error plotting ratio bars: {e}")

    # 3. Intervention CRR (Top 40)
    int_file = smoke_dir / f"intervention_{family}_top40.json"
    if int_file.exists():
        try:
            with open(int_file) as f:
                res = json.load(f)
            baseline = res["baseline_crr"]
            conds = res["conditions"]
            names = list(conds.keys())
            deltas = [conds[c]["delta_crr"] for c in names]
            correct = [conds[c]["expected_positive"] for c in names]
            cols = ["#4CAF50" if c else "#FF5722" for c in correct]
            
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.axhline(0, color="black", lw=1.0)
            bars = ax.bar(names, deltas, color=cols, alpha=0.85, edgecolor="white", lw=0.5)
            
            for bar, delta in zip(bars, deltas):
                ypos = delta + (0.01 if delta >= 0 else -0.02)
                va = "bottom" if delta >= 0 else "top"
                ax.text(bar.get_x() + bar.get_width()/2, ypos,
                        f"{delta:+.2f}", ha="center", va=va, fontsize=8, fontweight="bold")
            
            ax.set_ylabel("ΔCRR")
            ax.set_title(f"{family.upper()} — Smoke Intervention CRR (Baseline={baseline:.2f})")
            ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
            
            out = plots_dir / f"intervention_crr_{family}_top40.png"
            fig.savefig(out, dpi=120, bbox_inches="tight")
            plt.close(fig)
            log.info(f"Saved plot: {out}")
        except Exception as e:
            log.error(f"Error plotting intervention CRR: {e}")


# ── Main Run Family ───────────────────────────────────────────────────────────

def run_smoke_test(family: str, n_subset: int, device: str):
    log.info(f"\n==========================================")
    log.info(f"  RUNNING SMOKE TEST: {family.upper()}")
    log.info(f"==========================================")
    
    cfg = FAMILIES[family]
    smoke_dir = RESULTS_DIR / family / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer and prompts
    tokenizer = load_tokenizer(family)
    gc.collect()
    torch.cuda.empty_cache()
    
    prompts = load_substitution_prompts(tokenizer, verbose=False)
    if not prompts:
        log.error(f"No prompts found for {family}")
        return
        
    prompts = prompts[:n_subset]
    log.info(f"Loaded {len(prompts)} smoke test prompts.")

    saved_sup_all = {}
    saved_sup_top40 = {}

    for variant in ["base", "instruct"]:
        hf_name = cfg[variant]
        log.info(f"\n--- Loading {family} {variant} ---")
        model = load_tl_model(hf_name, family, device)
        n_layers = model.cfg.n_layers
        n_heads = model.cfg.n_heads
        
        # 1. ALL HEADS Analysis
        all_heads_list = [(l, h) for l in range(n_layers) for h in range(n_heads)]
        log.info(f"[{family}/{variant}] Computing superposition for ALL {len(all_heads_list)} heads...")
        t0 = time.time()
        sup_all = compute_superposition_for_heads(model, prompts, all_heads_list, cfg["norm_type"], device, f"{variant}_all_heads")
        saved_sup_all[variant] = sup_all
        
        out_all = smoke_dir / f"superposition_{family}_{variant}_all_heads.json"
        out_all.write_text(json.dumps(sup_all, indent=2))
        log.info(f"Saved all heads JSON in {time.time()-t0:.0f}s to {out_all}")

        # 2. TOP 20/20 HEADS Analysis (extract from results)
        ctx_heads, mem_heads = get_top_heads_from_results(family, variant, k=20)
        top40_list = list(dict.fromkeys(ctx_heads + mem_heads))
        log.info(f"[{family}/{variant}] Computing superposition for TOP {len(top40_list)} heads...")
        t0 = time.time()
        sup_top40 = compute_superposition_for_heads(model, prompts, top40_list, cfg["norm_type"], device, f"{variant}_top40")
        saved_sup_top40[variant] = sup_top40
        
        out_top40 = smoke_dir / f"superposition_{family}_{variant}_top40.json"
        out_top40.write_text(json.dumps(sup_top40, indent=2))
        log.info(f"Saved top40 JSON in {time.time()-t0:.0f}s to {out_top40}")

        # 3. INTERVENTION on top 40 heads (run on base model only, like superposition_analysis.py)
        if variant == "base":
            log.info(f"[{family}/{variant}] Running intervention on top 40 heads...")
            t0 = time.time()
            int_res = run_smoke_intervention(model, prompts, ctx_heads, mem_heads, family, variant)
            out_int = smoke_dir / f"intervention_{family}_top40.json"
            out_int.write_text(json.dumps(int_res, indent=2))
            log.info(f"Saved intervention JSON in {time.time()-t0:.0f}s to {out_int}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # 4. Compare base vs instruct
    if "base" in saved_sup_all and "instruct" in saved_sup_all:
        comp_all = compare_models(saved_sup_all["base"], saved_sup_all["instruct"])
        out_comp_all = smoke_dir / f"superposition_{family}_comparison_all_heads.json"
        out_comp_all.write_text(json.dumps(comp_all, indent=2))
        log.info(f"Saved all heads comparison to {out_comp_all}")

    if "base" in saved_sup_top40 and "instruct" in saved_sup_top40:
        comp_top40 = compare_models(saved_sup_top40["base"], saved_sup_top40["instruct"])
        out_comp_top40 = smoke_dir / f"superposition_{family}_comparison_top40.json"
        out_comp_top40.write_text(json.dumps(comp_top40, indent=2))
        log.info(f"Saved top40 comparison to {out_comp_top40}")

    # 5. Save Plots
    log.info(f"Generating and saving plots for {family} smoke test...")
    save_plots(family, smoke_dir)
    log.info(f"Smoke test for {family.upper()} completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run superposition smoke tests.")
    parser.add_argument("--family", choices=["qwen", "llama", "gemma"], help="Run one family.")
    parser.add_argument("--all", action="store_true", help="Run all three families.")
    parser.add_argument("--n-subset", type=int, default=8, help="Number of prompts to run on.")
    parser.add_argument("--device", default=None, help="Device to use.")
    args = parser.parse_args()
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device detected: {device}")
    
    if args.all:
        for fam in ["qwen", "llama", "gemma"]:
            run_smoke_test(fam, args.n_subset, device)
    elif args.family:
        run_smoke_test(args.family, args.n_subset, device)
    else:
        log.info("Please specify --family qwen/llama/gemma or --all")
