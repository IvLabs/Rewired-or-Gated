#!/usr/bin/env python3
"""
superposition_analysis.py
==========================
Superposition analysis using pre-computed LDS head weights from the
existing results (lds2_*_st_eap.json files in your branch).

WHAT THIS DOES
--------------
For each model family (Qwen2.5-3B, Llama-3.2-3B, Gemma-3-4B):
  1. Loads pre-validated top-k context and memory heads from your LDS results
  2. Runs ONE forward pass per prompt (no dual-run needed for measurement)
  3. Computes ctx_pull and mem_pull for each important head via logit attribution
  4. Computes ratio = ctx_pull / mem_pull per head
  5. Classifies: context (ratio>2) / superposition (0.5≤ratio≤2) / memory (ratio<0.5)
  6. Compares base vs instruct: delta_ratio, role_changed, shift_label
  7. Saves all results to results_superposition/

WHICH WEIGHTS TO USE
--------------------
Source: lds2_{model}_{variant}_substitution_st_eap.json (single-token filtered)
These are the CURRENT correct files. The archive_prefix_clean_text_buggy/ folder
contains OLD buggy runs — do NOT use those.

Top heads per family (from lds2_*_substitution_st_eap.json scores field):

Qwen2.5-3B base context:  L33H15, L32H7, L32H3, L31H5, L33H3, L32H13, L33H1, L27H10, L31H1, L27H9
Qwen2.5-3B base memory:   L31H2, L33H13, L34H10, L32H5, L26H7, L28H15, L20H5, L32H11, L33H9, L34H8
Qwen2.5-3B instruct ctx:  L33H15, L32H7, L32H3, L31H5, L33H3, L33H1, L32H13, L31H7, L31H1, L31H3
Qwen2.5-3B instruct mem:  L31H2, L33H13, L32H5, L26H7, L28H15, L20H5, L24H7, L21H0, L26H4, L34H10

Llama-3.2-3B base ctx:    L24H15, L14H22, L26H9, L21H20, L21H2, L14H2, L27H5, L24H4, L19H11, L27H15
Llama-3.2-3B base mem:    L24H16, L24H1, L14H18, L24H2, L15H20, L23H10, L17H17, L26H18, L18H11, L25H7
Llama-3.2-3B instruct ctx: L24H15, L26H9, L21H2, L27H5, L14H22, L27H15, L27H10, L19H11, L21H20, L14H2
Llama-3.2-3B instruct mem: L24H16, L14H18, L24H1, L24H2, L13H3, L15H20, L17H17, L18H11, L27H17, L23H10

Gemma-3-4B base ctx:      L29H4, L30H0, L23H1, L33H5, L23H3, L26H2, L22H5, L24H1, L30H5, L31H7
Gemma-3-4B base mem:      L30H1, L23H2, L33H4, L32H1, L31H6, L27H6, L22H4, L19H4, L30H2, L17H1
Gemma-3-4B instruct ctx:  L29H4, L30H0, L23H3, L23H1, L33H5, L22H5, L26H2, L30H5, L31H7, L30H6
Gemma-3-4B instruct mem:  L30H1, L23H2, L32H1, L33H4, L22H4, L31H6, L27H6, L19H4, L17H1, L15H1

OUTPUTS (results_superposition/)
---------------------------------
superposition_{family}_{variant}_substitution.json   — per-head pulls and ratios
superposition_{family}_comparison_substitution.json  — base vs instruct delta
intervention_{family}_substitution.json              — CRR under head amplification
logs/superposition_{family}_{timestamp}.log

RUN ORDER
---------
python superposition_analysis.py --family qwen
# verify → check results_superposition/superposition_qwen_comparison_substitution.json
python superposition_analysis.py --family llama
python superposition_analysis.py --family gemma
python superposition_analysis.py --all   # runs all three
"""

import argparse
import ast
import json
import logging
import os
import sys
import time
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import Counter

from datasets import load_dataset
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
from huggingface_hub import login as hf_login

load_dotenv()

# Register HF token with huggingface_hub so TransformerLens auto-reads it
_hf_token = os.environ.get("HF_TOKEN", None)
if _hf_token:
    hf_login(token=_hf_token, add_to_git_credential=False)

# ── Output directory ───────────────────────────────────────────────────────────
RESULTS_DIR = Path("results_superposition")
LOGS_DIR    = RESULTS_DIR / "logs"
RESULTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ── Superposition thresholds ───────────────────────────────────────────────────
CONTEXT_THRESH = 2.0   # ratio > 2.0  → context head
MEMORY_THRESH  = 0.5   # ratio < 0.5  → memory head
                       # 0.5–2.0      → superposition


# ══════════════════════════════════════════════════════════════════════════════
# HEAD REGISTRY
# Pre-computed from lds2_*_substitution_st_eap.json (single-token filtered)
# These are the TOP-10 context heads and TOP-10 memory heads per model×variant.
# Format: (layer, head)
# Source: your results/{family}/lds2_{model}_{variant}_substitution_st_eap.json
# ══════════════════════════════════════════════════════════════════════════════

HEAD_REGISTRY = {
    # ── Qwen2.5-3B ────────────────────────────────────────────────────────────
    ("qwen", "base", "context"): [
        (33,15),(32,7),(32,3),(31,5),(33,3),(32,13),(33,1),(27,10),(31,1),(27,9)
    ],
    ("qwen", "base", "memory"): [
        (31,2),(33,13),(34,10),(32,5),(26,7),(28,15),(20,5),(32,11),(33,9),(34,8)
    ],
    ("qwen", "instruct", "context"): [
        (33,15),(32,7),(32,3),(31,5),(33,3),(33,1),(32,13),(31,7),(31,1),(31,3)
    ],
    ("qwen", "instruct", "memory"): [
        (31,2),(33,13),(32,5),(26,7),(28,15),(20,5),(24,7),(21,0),(26,4),(34,10)
    ],

    # ── Llama-3.2-3B ──────────────────────────────────────────────────────────
    ("llama", "base", "context"): [
        (24,15),(14,22),(26,9),(21,20),(21,2),(14,2),(27,5),(24,4),(19,11),(27,15)
    ],
    ("llama", "base", "memory"): [
        (24,16),(24,1),(14,18),(24,2),(15,20),(23,10),(17,17),(26,18),(18,11),(25,7)
    ],
    ("llama", "instruct", "context"): [
        (24,15),(26,9),(21,2),(27,5),(14,22),(27,15),(27,10),(19,11),(21,20),(14,2)
    ],
    ("llama", "instruct", "memory"): [
        (24,16),(14,18),(24,1),(24,2),(13,3),(15,20),(17,17),(18,11),(27,17),(23,10)
    ],

    # ── Gemma-3-4B ────────────────────────────────────────────────────────────
    ("gemma", "base", "context"): [
        (29,4),(30,0),(23,1),(33,5),(23,3),(26,2),(22,5),(24,1),(30,5),(31,7)
    ],
    ("gemma", "base", "memory"): [
        (30,1),(23,2),(33,4),(32,1),(31,6),(27,6),(22,4),(19,4),(30,2),(17,1)
    ],
    ("gemma", "instruct", "context"): [
        (29,4),(30,0),(23,3),(23,1),(33,5),(22,5),(26,2),(30,5),(31,7),(30,6)
    ],
    ("gemma", "instruct", "memory"): [
        (30,1),(23,2),(32,1),(33,4),(22,4),(31,6),(27,6),(19,4),(17,1),(15,1)
    ],
}

# ── Model registry ─────────────────────────────────────────────────────────────
FAMILIES = {
    "qwen": {
        "base":           "Qwen/Qwen2.5-3B",
        "instruct":       "Qwen/Qwen2.5-3B-Instruct",
        "norm_type":      "rms",
        "center_unembed": False,
        "gated":          False,
        "n_layers":       36,
        "n_heads":        16,
    },
    "llama": {
        "base":           "meta-llama/Llama-3.2-3B",
        "instruct":       "meta-llama/Llama-3.2-3B-Instruct",
        "norm_type":      "rms",
        "center_unembed": False,
        "gated":          True,
        "n_layers":       28,
        "n_heads":        24,
    },
    "gemma": {
        "base":           "google/gemma-3-4b-pt",
        "instruct":       "google/gemma-3-4b-it",
        "norm_type":      "rms",
        "center_unembed": False,
        "gated":          True,
        "n_layers":       34,
        "n_heads":        8,
    },
}

# ── CRR baselines from your existing results ────────────────────────────────
# Source: results/{family}/crr_{model}_{variant}_substitution_st.json
BASELINE_CRR = {
    ("qwen",  "base"):    0.6453,
    ("qwen",  "instruct"):0.2199,
    ("llama", "base"):    0.6754,
    ("llama", "instruct"):0.2631,
    ("gemma", "base"):    0.8701,
    ("gemma", "instruct"):0.3690,
}


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging(family: str) -> logging.Logger:
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    logpath = LOGS_DIR / f"superposition_{family}_{ts}.log"
    logging.basicConfig(
        level    = logging.INFO,
        format   = "%(asctime)s  %(levelname)s  %(message)s",
        handlers = [
            logging.FileHandler(logpath),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    # Silence noisy HTTP/network libraries — keep only superposition content
    for noisy in (
        "httpx", "httpcore", "httpcore.http11", "httpcore.connection",
        "huggingface_hub.file_download", "huggingface_hub.utils._http",
        "huggingface_hub.utils._headers", "huggingface_hub._commit_api",
        "huggingface_hub", "filelock", "urllib3", "transformers",
    ):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    log = logging.getLogger(__name__)
    log.info(f"Log: {logpath}")
    return log


# ══════════════════════════════════════════════════════════════════════════════
# DATASET LOADER
# ══════════════════════════════════════════════════════════════════════════════

def _single_token_id(text: str, tokenizer) -> int | None:
    for variant in (" " + text.strip(), text.strip()):
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    return None


def _best_token_from_answer(answer_field: str, tokenizer) -> tuple[int, str] | tuple[None, None]:
    try:
        aliases = ast.literal_eval(answer_field)
        if isinstance(aliases, str):
            aliases = [aliases]
    except Exception:
        aliases = [answer_field]
    for alias in aliases:
        tid = _single_token_id(str(alias), tokenizer)
        if tid is not None:
            return tid, str(alias)
    return None, None


def load_substitution_prompts(tokenizer, verbose: bool = True) -> list[dict]:
    """
    Load gaotang/ParaConflict substitution prompts.
    Filters to single-token answers only (matches your LDS pipeline).
    Returns list of dicts: {prompt, prompt_clean, memory_token_id, context_token_id, domain}
    """
    if verbose:
        print("[dataset] Loading gaotang/ParaConflict (split=test)...")
    ds = load_dataset("gaotang/ParaConflict", split="test")

    kept, skipped = [], 0

    for row in ds:
        conflict_text = str(row.get("Substitution Conflict", "") or "").strip()
        clean_text    = str(row.get("Clean Prompt",          "") or "").strip()
        answer_field  = str(row.get("Answer",                "") or "").strip()
        distract_text = str(row.get("Distracted Token",      "") or "").strip()
        domain        = str(row.get("Category",              "") or "").strip()

        if not (conflict_text and clean_text and answer_field and distract_text):
            skipped += 1; continue

        mid, mem_surface = _best_token_from_answer(answer_field, tokenizer)
        did = _single_token_id(distract_text, tokenizer)

        if mid is None or did is None or mid == did:
            skipped += 1; continue

        kept.append({
            "prompt":           conflict_text,
            "prompt_clean":     clean_text,
            "memory_token_id":  mid,
            "context_token_id": did,
            "memory_surface":   mem_surface,
            "context_surface":  distract_text,
            "domain":           domain,
        })

    if verbose:
        domains = Counter(r["domain"] for r in kept)
        print(f"[dataset] Kept={len(kept)}  Skipped={skipped}")
        for d, c in domains.most_common():
            print(f"          {d}: {c}")

    return kept


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_tl_model(hf_name: str, family_key: str, device: str) -> HookedTransformer:
    cfg      = FAMILIES[family_key]
    hf_token = os.environ.get("HF_TOKEN", None)

    if cfg["gated"] and not hf_token:
        raise EnvironmentError(
            f"{hf_name} is gated. Set HF_TOKEN in your .env file.\n"
            f"Accept license: https://huggingface.co/{hf_name}"
        )

    tl_kwargs = dict(
        center_unembed         = cfg["center_unembed"],
        center_writing_weights = False,
        fold_ln                = True,
        refactor_factored_attn_matrices = False,
        dtype                  = torch.float32,
    )
    # Note: token is registered via hf_login() at module load.
    # Do NOT add token= to tl_kwargs — TransformerLens 3.5.1 also passes it
    # internally to AutoConfig, causing "multiple values for keyword argument 'token'".

    print(f"[model] Loading {hf_name}...")
    try:
        model = HookedTransformer.from_pretrained(hf_name, **tl_kwargs)
    except Exception as e:
        print(f"[model] Direct TL load failed ({e}), trying HF fallback...")
        hf_kwargs = {"torch_dtype": torch.float32}
        if hf_token:
            hf_kwargs["token"] = hf_token
        hf_model = AutoModelForCausalLM.from_pretrained(hf_name, **hf_kwargs)
        model = HookedTransformer.from_pretrained(
            cfg["base"], hf_model=hf_model, **tl_kwargs
        )

    model.to(device).eval()
    assert model.cfg.n_layers == cfg["n_layers"], \
        f"Expected {cfg['n_layers']} layers, got {model.cfg.n_layers}"
    assert model.cfg.n_heads == cfg["n_heads"], \
        f"Expected {cfg['n_heads']} heads, got {model.cfg.n_heads}"

    print(f"[model] OK — {model.cfg.n_layers}L × {model.cfg.n_heads}H, "
          f"d_model={model.cfg.d_model}, d_head={model.cfg.d_head}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# NORMALIZATION SCALE
# ══════════════════════════════════════════════════════════════════════════════

def get_ln_scale(model: HookedTransformer, cache, norm_type: str) -> torch.Tensor:
    """
    Scalar normalization scale from final residual stream.
    RMSNorm (Qwen/Llama/Gemma): scale = 1/sqrt(mean(x^2) + eps)
    LayerNorm (GPT-2):          scale = 1/sqrt(var(x) + eps)
    Returns shape [1,1] for broadcasting.
    """
    n_layers = model.cfg.n_layers
    resid    = cache[f"blocks.{n_layers-1}.hook_resid_post"][:, -1, :]
    try:
        eps = model.ln_final.eps
    except AttributeError:
        eps = 1e-5

    if norm_type == "rms":
        rms   = (resid ** 2).mean(dim=-1, keepdim=True)
        scale = 1.0 / (rms + eps).sqrt()
    else:
        mean  = resid.mean(dim=-1, keepdim=True)
        var   = ((resid - mean) ** 2).mean(dim=-1, keepdim=True)
        scale = 1.0 / (var + eps).sqrt()

    return scale  # [1, 1]


# ══════════════════════════════════════════════════════════════════════════════
# CORE SUPERPOSITION COMPUTATION
# ONE forward pass per prompt — no dual-run needed for measurement
# ══════════════════════════════════════════════════════════════════════════════

def compute_superposition(
    model:         HookedTransformer,
    prompts:       list[dict],
    target_heads:  list[tuple[int, int]],   # (layer, head) pairs to score
    norm_type:     str,
    device:        str,
    label:         str = "",
) -> dict:
    """
    Single forward pass per prompt. Computes logit attribution for target_heads only.

    For each head (l, h):
        z_h    = hook_z[0, -1, h, :]              [d_head]
        out_h  = z_h @ W_O[l, h]                  [d_model]
        scaled = out_h * ln_scale                  [d_model] (linear norm approx)
        ctx_pull = scaled @ W_U[:, context_token]  scalar
        mem_pull = scaled @ W_U[:, memory_token]   scalar
        ratio    = ctx_pull / mem_pull

    Averaged over all prompts.
    """
    n_layers = model.cfg.n_layers
    W_O      = model.W_O   # [n_layers, n_heads, d_head, d_model]
    W_U      = model.W_U   # [d_model, vocab_size]

    # Per-head accumulators
    ctx_acc  = {h: 0.0 for h in target_heads}
    mem_acc  = {h: 0.0 for h in target_heads}
    n_valid  = 0

    # Only cache layers we need + final resid
    needed_layers = set(l for l, h in target_heads)
    needed_hooks  = {f"blocks.{l}.attn.hook_z" for l in needed_layers}
    needed_hooks.add(f"blocks.{n_layers-1}.hook_resid_post")
    names_filter = lambda name: name in needed_hooks

    for row in tqdm(prompts, desc=f"  {label}", unit="prompt"):
        tokens = model.to_tokens(row["prompt"])
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
            logging.warning(f"Forward pass failed: {e}")
            continue

        ln_scale = get_ln_scale(model, cache, norm_type)  # [1,1]
        mem_tok  = row["memory_token_id"]
        ctx_tok  = row["context_token_id"]

        for (l, h) in target_heads:
            hook_name = f"blocks.{l}.attn.hook_z"
            z         = cache[hook_name][0, -1, h, :]     # [d_head]
            out_h     = z @ W_O[l, h]                     # [d_model]
            scaled    = out_h * ln_scale.squeeze()         # [d_model]

            ctx_acc[(l, h)] += (scaled @ W_U[:, ctx_tok]).item()
            mem_acc[(l, h)] += (scaled @ W_U[:, mem_tok]).item()

        del cache
        n_valid += 1

    if n_valid == 0:
        raise RuntimeError("No prompts processed successfully.")

    # Average and compute ratios
    def _role(r: float) -> str:
        if r > CONTEXT_THRESH:  return "context"
        if r < MEMORY_THRESH:   return "memory"
        return "superposition"

    heads = {}
    for (l, h) in target_heads:
        avg_ctx = ctx_acc[(l, h)] / n_valid
        avg_mem = mem_acc[(l, h)] / n_valid
        if abs(avg_mem) > 1e-8:
            ratio = avg_ctx / avg_mem
        else:
            ratio = 1e6 if avg_ctx > 0 else -1e6

        key = f"L{l}H{h}"
        heads[key] = {
            "layer":    l,
            "head":     h,
            "ctx_pull": round(avg_ctx, 6),
            "mem_pull": round(avg_mem, 6),
            "ratio":    round(ratio,   6),
            "role":     _role(ratio),
        }

    # Print summary
    roles = Counter(v["role"] for v in heads.values())
    print(f"\n  [{label}] n_prompts={n_valid}  "
          f"context={roles['context']}  "
          f"superposition={roles['superposition']}  "
          f"memory={roles['memory']}")

    sorted_heads = sorted(heads.items(), key=lambda x: x[1]["ratio"], reverse=True)
    print("  Top-5 by ratio:")
    for k, v in sorted_heads[:5]:
        print(f"    {k}: ratio={v['ratio']:+.3f}  "
              f"ctx={v['ctx_pull']:.4f}  mem={v['mem_pull']:.4f}  [{v['role']}]")
    print("  Bottom-5 by ratio:")
    for k, v in sorted_heads[-5:]:
        print(f"    {k}: ratio={v['ratio']:+.3f}  "
              f"ctx={v['ctx_pull']:.4f}  mem={v['mem_pull']:.4f}  [{v['role']}]")

    return {
        "family":     label.split("/")[0] if "/" in label else label,
        "variant":    label.split("/")[1] if "/" in label else "",
        "model":      "",
        "n_prompts":  n_valid,
        "thresholds": {"context": CONTEXT_THRESH, "memory": MEMORY_THRESH},
        "heads":      heads,
    }


# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON: base vs instruct
# ══════════════════════════════════════════════════════════════════════════════

def compare_models(base_res: dict, inst_res: dict) -> dict:
    """
    Compute per-head delta_ratio and role shifts between base and instruct.
    Only compares heads present in BOTH results.
    """
    comparison = {}
    for key, bh in base_res["heads"].items():
        if key not in inst_res["heads"]:
            continue
        ih = inst_res["heads"][key]

        base_role = bh["role"]
        inst_role = ih["role"]
        shift     = "stable" if base_role == inst_role else f"{base_role}→{inst_role}"

        comparison[key] = {
            "layer":          bh["layer"],
            "head":           bh["head"],
            "base_ratio":     bh["ratio"],
            "base_role":      base_role,
            "base_ctx_pull":  bh["ctx_pull"],
            "base_mem_pull":  bh["mem_pull"],
            "inst_ratio":     ih["ratio"],
            "inst_role":      inst_role,
            "inst_ctx_pull":  ih["ctx_pull"],
            "inst_mem_pull":  ih["mem_pull"],
            "ratio_shift":    round(ih["ratio"] - bh["ratio"], 6),
            "role_changed":   base_role != inst_role,
            "shift_label":    shift,
        }

    # Print comparison summary
    shifts  = Counter(v["shift_label"] for v in comparison.values())
    changed = [v for v in comparison.values() if v["role_changed"]]
    print(f"\n  [comparison] {len(changed)}/{len(comparison)} heads changed role")
    for s, c in shifts.most_common():
        if s != "stable":
            print(f"    {s}: {c}")

    # Mean ratio shift
    all_shifts = [v["ratio_shift"] for v in comparison.values()]
    mean_shift = sum(all_shifts) / len(all_shifts) if all_shifts else 0
    print(f"  Mean ratio shift (instruct-base): {mean_shift:+.4f}")
    print(f"  (positive = instruction tuning pushed heads toward context)")

    return comparison


# ══════════════════════════════════════════════════════════════════════════════
# LOGIT INTERVENTION — causal validation
# ══════════════════════════════════════════════════════════════════════════════

def _make_scale_hooks(heads: list[tuple[int,int]], scale: float) -> list[tuple]:
    hooks = []
    for (l, h) in heads:
        def make_fn(layer=l, head=h, s=scale):
            def fn(value, hook):
                value[:, :, head, :] = value[:, :, head, :] * s
                return value
            return fn
        hooks.append((f"blocks.{l}.attn.hook_z", make_fn()))
    return hooks


def compute_crr_with_hooks(
    model:   HookedTransformer,
    prompts: list[dict],
    hooks:   list[tuple] | None,
    label:   str = "",
) -> dict:
    n_ctx = n_mem = n_nei = 0

    for row in tqdm(prompts, desc=f"  CRR {label}", unit="prompt"):
        tokens = model.to_tokens(row["prompt"])
        if tokens.shape[1] > model.cfg.n_ctx:
            tokens = tokens[:, -model.cfg.n_ctx:]
        try:
            with torch.no_grad():
                if hooks:
                    logits = model.run_with_hooks(
                        tokens, fwd_hooks=hooks, return_type="logits"
                    )
                else:
                    logits = model(tokens, return_type="logits")

            pred = logits[0, -1, :].argmax().item()
            if pred == row["context_token_id"]:  n_ctx += 1
            elif pred == row["memory_token_id"]: n_mem += 1
            else:                                n_nei += 1
        except Exception:
            n_nei += 1

    total = n_ctx + n_mem + n_nei
    crr   = n_ctx / total if total > 0 else 0.0
    return {"crr": crr, "n_context": n_ctx, "n_memory": n_mem,
            "n_neither": n_nei, "n_total": total}


def run_intervention(
    model:    HookedTransformer,
    prompts:  list[dict],
    family:   str,
    variant:  str,
) -> dict:
    """
    Four intervention conditions using the top-5 heads from HEAD_REGISTRY.
    Validates that head importance is causal, not just correlational.
    """
    ctx_heads = HEAD_REGISTRY[(family, variant, "context")][:5]
    mem_heads = HEAD_REGISTRY[(family, variant, "memory")][:5]

    print(f"\n[intervention/{family}/{variant}]")
    print(f"  ctx_heads: {ctx_heads}")
    print(f"  mem_heads: {mem_heads}")

    # Use known baseline CRR from your existing results
    # (avoids re-running baseline to save time)
    known_baseline = BASELINE_CRR.get((family, variant), None)
    print(f"  Running baseline CRR (known={known_baseline})...")
    baseline = compute_crr_with_hooks(model, prompts, hooks=None,
                                      label=f"{family}/{variant}/baseline")
    print(f"  Computed baseline CRR: {baseline['crr']:.4f}  "
          f"(known from LDS results: {known_baseline})")

    conditions = [
        ("amplify_context",  ctx_heads, 2.0, "expect ΔCRR > 0"),
        ("suppress_context", ctx_heads, 0.0, "expect ΔCRR < 0"),
        ("amplify_memory",   mem_heads, 2.0, "expect ΔCRR < 0"),
        ("suppress_memory",  mem_heads, 0.0, "expect ΔCRR > 0"),
    ]

    results = {
        "family":       family,
        "variant":      variant,
        "baseline_crr": baseline["crr"],
        "known_baseline_crr": known_baseline,
        "ctx_heads":    ctx_heads,
        "mem_heads":    mem_heads,
        "conditions":   {},
    }

    for cond_name, heads, scale, expectation in conditions:
        print(f"\n  [{cond_name}] scale={scale}  {expectation}")
        hooks  = _make_scale_hooks(heads, scale)
        res    = compute_crr_with_hooks(model, prompts, hooks, label=cond_name)
        delta  = res["crr"] - baseline["crr"]
        correct = (delta > 0) if "amplify_context" in cond_name or "suppress_memory" in cond_name \
                  else (delta < 0)
        marker = "✓" if correct else "✗"

        print(f"  CRR={res['crr']:.4f}  ΔCRR={delta:+.4f}  {marker}")

        results["conditions"][cond_name] = {
            "scale":           scale,
            "crr":             res["crr"],
            "delta_crr":       round(delta, 6),
            "n_context":       res["n_context"],
            "n_memory":        res["n_memory"],
            "n_neither":       res["n_neither"],
            "expected_positive": correct,
            "expectation":     expectation,
        }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PER-FAMILY RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_family(family: str, device: str):
    log = setup_logging(family)
    cfg = FAMILIES[family]

    print(f"\n{'='*65}")
    print(f"SUPERPOSITION ANALYSIS: {family.upper()}")
    print(f"  Base:      {cfg['base']}")
    print(f"  Instruct:  {cfg['instruct']}")
    print(f"  norm_type: {cfg['norm_type']}")
    print(f"  Device:    {device}")
    print("="*65)

    # Load tokenizer — pass token explicitly for gated models
    hf_token = os.environ.get("HF_TOKEN", None)
    tok_kwargs = {"token": hf_token} if hf_token else {}
    tokenizer = AutoTokenizer.from_pretrained(cfg["base"], **tok_kwargs)

    # Load prompts
    prompts = load_substitution_prompts(tokenizer, verbose=True)
    if not prompts:
        print(f"[ERROR] No prompts loaded for {family}.")
        return

    saved_sup = {}

    for variant in ["base", "instruct"]:
        sup_path = RESULTS_DIR / f"superposition_{family}_{variant}_substitution.json"

        # Resume support
        if sup_path.exists():
            print(f"\n[{family}/{variant}] Already exists, loading: {sup_path}")
            with open(sup_path) as f:
                saved_sup[variant] = json.load(f)
            continue

        hf_name       = cfg[variant]
        target_ctx    = HEAD_REGISTRY[(family, variant, "context")]
        target_mem    = HEAD_REGISTRY[(family, variant, "memory")]
        target_heads  = list(dict.fromkeys(target_ctx + target_mem))  # deduplicate

        t0    = time.time()
        model = load_tl_model(hf_name, family, device)

        log.info(f"Loaded {hf_name} | "
                 f"n_layers={model.cfg.n_layers} n_heads={model.cfg.n_heads} "
                 f"d_model={model.cfg.d_model}")

        print(f"\n[{family}/{variant}] Scoring {len(target_heads)} heads "
              f"× {len(prompts)} prompts...")

        sup_res = compute_superposition(
            model        = model,
            prompts      = prompts,
            target_heads = target_heads,
            norm_type    = cfg["norm_type"],
            device       = device,
            label        = f"{family}/{variant}",
        )
        sup_res["model"]   = hf_name
        sup_res["variant"] = variant
        sup_res["family"]  = family

        saved_sup[variant] = sup_res
        sup_path.write_text(json.dumps(sup_res, indent=2))
        print(f"\n[{family}/{variant}] Saved → {sup_path}")
        log.info(f"Done in {time.time()-t0:.0f}s  n_prompts={sup_res['n_prompts']}")

        # Run intervention on the BASE model only
        # (interventions are run on base to test causal direction)
        if variant == "base":
            print(f"\n[{family}/{variant}] Running intervention experiment...")
            int_res  = run_intervention(model, prompts, family, variant)
            int_path = RESULTS_DIR / f"intervention_{family}_substitution.json"
            int_path.write_text(json.dumps(int_res, indent=2))
            print(f"[{family}/{variant}] Intervention saved → {int_path}")

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # Comparison
    if "base" in saved_sup and "instruct" in saved_sup:
        comp     = compare_models(saved_sup["base"], saved_sup["instruct"])
        comp["_meta"] = {
            "family":           family,
            "baseline_crr_base":    BASELINE_CRR.get((family, "base")),
            "baseline_crr_instruct":BASELINE_CRR.get((family, "instruct")),
            "note": "ratio_shift > 0 means instruct head is more context-biased",
        }
        comp_path = RESULTS_DIR / f"superposition_{family}_comparison_substitution.json"
        comp_path.write_text(json.dumps(comp, indent=2))
        print(f"\n[{family}] Comparison saved → {comp_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Superposition analysis using pre-computed LDS head weights"
    )
    p.add_argument("--family", choices=["qwen","llama","gemma"],
                   help="Run one family.")
    p.add_argument("--all",    action="store_true",
                   help="Run all three: qwen → llama → gemma.")
    p.add_argument("--device", default=None,
                   help="cuda or cpu. Default: auto-detect.")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Output: {RESULTS_DIR.absolute()}")

    if args.all:
        for fam in ["qwen", "llama", "gemma"]:
            run_family(fam, device)
    elif args.family:
        run_family(args.family, device)
    else:
        print("Specify --family qwen/llama/gemma  or  --all")
        print("Example: python superposition_analysis.py --family qwen")
