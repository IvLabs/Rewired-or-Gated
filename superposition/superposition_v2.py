#!/usr/bin/env python3
"""
superposition_v2.py  —  fixed superposition analysis + ablation harness
=======================================================================
Supersedes `superposition_analysis.py`. Self-contained on purpose so it can be
reviewed and diffed against the old monolith without import tangles.

WHAT CHANGED vs superposition_analysis.py (and WHY)
---------------------------------------------------
1. METRIC FIX (the critical bug).
   Old:  ratio = ctx_pull / mem_pull, classify by ratio thresholds.
         `ctx_pull` and `mem_pull` are SIGNED direct-logit-attribution scores,
         so the ratio sign-flips (a head pulling toward context but with
         mem_pull<0 was labelled "memory") and explodes when mem_pull≈0
         (ratios of ±440 dominated every average).
   New:  context_index = (ctx_pull - mem_pull) / (|ctx_pull| + |mem_pull| + eps)
         Bounded in [-1, +1]. Sign-aware. Never divides by a signed near-zero.
             +1  → pure context      (ctx>0, mem≤0)
              0  → balanced           → SUPERPOSITION
             -1  → pure memory        (mem>0, ctx≤0)
         Classification: index > +T → context, < -T → memory, else superposition.
         We also keep raw pulls, a legacy ratio (only where both pulls > 0), and
         a `sign_pattern` diagnostic so nothing is hidden.

2. CONSISTENT HEAD SET (fixes the ~15-head moving-intersection comparison).
   Heads are read straight from the LDS score files. We score the UNION of
   (base top-k context ∪ base top-k memory ∪ instruct top-k context ∪ instruct
   top-k memory) in BOTH models, so every head is comparable base-vs-instruct.

3. REWIRING vs GATING is now measured, not thrown away.
   The comparison reports, per family:
     - node-level Jaccard of base-vs-instruct top-k context and memory sets
     - Spearman correlation of the full 576-head LDS ranking base-vs-instruct
   Low Jaccard / low Spearman => the important heads CHANGED => rewiring (H1).
   High overlap + functional (index) shift on the shared heads => gating (H2).

4. PROMPT SET CONSISTENCY (fixes the baseline-CRR mismatch).
   Prompts are loaded from the foundation's single_token_survival_*.json rows
   when present (Llama, Gemma) with the PINNED memory_answer (a21_naive_answer),
   so we score exactly the rows LDS/CRR used. Qwen (no survival file committed)
   falls back to re-filtering the HF dataset, which reproduces the same 764 rows.

Run:
    python superposition/superposition_v2.py --family qwen   --mode all
    python superposition/superposition_v2.py --family llama  --mode superposition
    python superposition/superposition_v2.py --family gemma  --mode ablation
    python superposition/superposition_v2.py --all           --mode all
"""

from __future__ import annotations
import argparse
import ast
import json
import logging
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import torch
from tqdm import tqdm

# ── Paths ───────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent          # inner BBNLP repo
LDS_DIR     = REPO_ROOT / "results"                           # LDS + CRR inputs
OUT_DIR     = REPO_ROOT / "results_superposition_v2"          # NEW dir, does not clobber old
LOGS_DIR    = OUT_DIR / "logs"

# ── Superposition classification thresholds (on context_index ∈ [-1, 1]) ────────
INDEX_CONTEXT_T = 0.33    # index > +0.33 → context head
INDEX_MEMORY_T  = 0.33    # index < -0.33 → memory head
EPS             = 1e-9

# ── Family registry ─────────────────────────────────────────────────────────────
FAMILIES = {
    # dtype: fp32 for Qwen/Llama (3B fp32 ≈ 12GB, comfortable on a 24GB L4 --
    # this is the config the already-validated Qwen run used). Gemma-3-4B in
    # fp32 is ~17GB just for weights, too tight alongside the activation cache
    # and repeated run_with_hooks calls during ablation -- use bf16 (~8.6GB,
    # the same dtype the working LDS/CRR pipeline already uses for Gemma on L4).
    "qwen":  {"base": "Qwen/Qwen2.5-3B",        "instruct": "Qwen/Qwen2.5-3B-Instruct",
              "tag": "qwen25_3b",  "norm_type": "rms", "gated": False, "dtype": "float32",
              "n_layers": 36, "n_heads": 16},
    "llama": {"base": "meta-llama/Llama-3.2-3B", "instruct": "meta-llama/Llama-3.2-3B-Instruct",
              "tag": "llama32_3b", "norm_type": "rms", "gated": True, "dtype": "float32",
              "n_layers": 28, "n_heads": 24},
    "gemma": {"base": "google/gemma-3-4b-pt",    "instruct": "google/gemma-3-4b-it",
              "tag": "gemma3_4b",  "norm_type": "rms", "gated": True, "dtype": "bfloat16",
              "n_layers": 34, "n_heads": 8},
}

# Known behavioural CRR from the LDS pipeline (for cross-checking our recompute).
BASELINE_CRR = {
    ("qwen", "base"): 0.6453,  ("qwen", "instruct"): 0.2199,
    ("llama", "base"): 0.6754, ("llama", "instruct"): 0.2631,
    ("gemma", "base"): 0.8701, ("gemma", "instruct"): 0.3690,
}

log = logging.getLogger("superposition_v2")


# ════════════════════════════════════════════════════════════════════════════════
# LOGGING
# ════════════════════════════════════════════════════════════════════════════════
def setup_logging(family: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[logging.FileHandler(LOGS_DIR / f"superposition_v2_{family}_{ts}.log"),
                  logging.StreamHandler(sys.stdout)],
        force=True,
    )
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3", "transformers", "filelock"):
        logging.getLogger(noisy).setLevel(logging.ERROR)


# ════════════════════════════════════════════════════════════════════════════════
# LDS HEAD LOADING  — read important heads straight from the score files
# ════════════════════════════════════════════════════════════════════════════════
def load_lds_scores(family: str, variant: str) -> dict[tuple[int, int], float]:
    """Return {(layer, head): lds_score} for all heads. +score = context-leaning."""
    tag  = FAMILIES[family]["tag"]
    path = LDS_DIR / tag / f"lds2_{tag}_{variant}_substitution_st_eap.json"
    if not path.exists():
        raise FileNotFoundError(f"LDS scores not found: {path}")
    raw = json.loads(path.read_text())["scores"]
    out = {}
    for key, val in raw.items():
        l, h = key.split("_")
        out[(int(l), int(h))] = float(val)
    return out


def top_heads(scores: dict, k: int, kind: str) -> list[tuple[int, int]]:
    """kind='context' → highest LDS scores; kind='memory' → lowest (most negative)."""
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=(kind == "context"))
    return [hd for hd, _ in ordered[:k]]


def build_union_head_set(family: str, k: int) -> tuple[list[tuple[int, int]], dict]:
    """
    Union of base+instruct top-k context and memory heads → the CONSISTENT set
    scored in both models. Also return the node-overlap (rewiring) diagnostics.
    """
    base_s = load_lds_scores(family, "base")
    inst_s = load_lds_scores(family, "instruct")

    sets = {
        "base_ctx":  top_heads(base_s, k, "context"),
        "base_mem":  top_heads(base_s, k, "memory"),
        "inst_ctx":  top_heads(inst_s, k, "context"),
        "inst_mem":  top_heads(inst_s, k, "memory"),
    }
    union = list(dict.fromkeys(sets["base_ctx"] + sets["base_mem"]
                               + sets["inst_ctx"] + sets["inst_mem"]))

    def jaccard(a, b):
        A, B = set(a), set(b)
        return len(A & B) / len(A | B) if (A | B) else 1.0

    # Spearman of the full ranking (rewiring vs gating at the ranking level).
    common = sorted(set(base_s) & set(inst_s))
    bv = np.array([base_s[h] for h in common])
    iv = np.array([inst_s[h] for h in common])
    spearman = _spearman(bv, iv)

    node_diag = {
        "topk": k,
        "jaccard_context_topk": round(jaccard(sets["base_ctx"], sets["inst_ctx"]), 4),
        "jaccard_memory_topk":  round(jaccard(sets["base_mem"], sets["inst_mem"]), 4),
        "spearman_full_ranking": round(spearman, 4),
        "interpretation": _rewiring_verdict(jaccard(sets["base_ctx"], sets["inst_ctx"]),
                                            jaccard(sets["base_mem"], sets["inst_mem"]),
                                            spearman),
        "base_context_topk":  [f"L{l}H{h}" for l, h in sets["base_ctx"]],
        "inst_context_topk":  [f"L{l}H{h}" for l, h in sets["inst_ctx"]],
        "base_memory_topk":   [f"L{l}H{h}" for l, h in sets["base_mem"]],
        "inst_memory_topk":   [f"L{l}H{h}" for l, h in sets["inst_mem"]],
    }
    return union, node_diag


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = (ra - ra.mean()); rb = (rb - rb.mean())
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def _rewiring_verdict(j_ctx: float, j_mem: float, spearman: float) -> str:
    j = 0.5 * (j_ctx + j_mem)
    if j >= 0.6 and spearman >= 0.5:
        return "GATING (H2): same heads matter base vs instruct"
    if j < 0.3 or spearman < 0.2:
        return "REWIRING (H1): important heads changed base vs instruct"
    return "MIXED: partial overlap — triangulate with path patching"


# ════════════════════════════════════════════════════════════════════════════════
# PROMPT LOADING  — reuse the foundation's exact rows when available
# ════════════════════════════════════════════════════════════════════════════════
def _single_token_id(text: str, tokenizer) -> int | None:
    for variant in (" " + text.strip(), text.strip()):
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    return None


def _best_alias_token(answer_field, tokenizer) -> tuple[int | None, str | None]:
    try:
        aliases = ast.literal_eval(answer_field) if isinstance(answer_field, str) else answer_field
        if isinstance(aliases, str):
            aliases = [aliases]
    except Exception:
        aliases = [answer_field]
    for alias in aliases:
        tid = _single_token_id(str(alias), tokenizer)
        if tid is not None:
            return tid, str(alias)
    return None, None


def load_prompts(family: str, variant: str, tokenizer, limit: int | None = None) -> list[dict]:
    """
    Prefer the foundation's single_token_survival rows (pinned memory_answer).
    Fall back to re-filtering the HF dataset (Qwen path). Both converge on the
    same rows LDS/CRR used, so our recomputed CRR matches the pipeline baseline.
    """
    tag = FAMILIES[family]["tag"]
    surv = LDS_DIR / tag / f"single_token_survival_{tag}_{variant}.json"
    rows_out: list[dict] = []

    if surv.exists():
        rows = json.loads(surv.read_text())["rows"]
        for r in rows:
            mem_txt = r.get("memory_answer") or r.get("a21_naive_answer")
            ctx_txt = r["context_answer"]
            mid = _single_token_id(str(mem_txt), tokenizer)
            did = _single_token_id(str(ctx_txt), tokenizer)
            if mid is None or did is None or mid == did:
                continue
            rows_out.append({
                "prompt": r["substitution_text"], "domain": r.get("domain", ""),
                "memory_token_id": mid, "context_token_id": did,
                "memory_surface": str(mem_txt), "context_surface": str(ctx_txt),
            })
        src = f"survival:{surv.name}"
    else:
        from datasets import load_dataset
        ds = load_dataset("gaotang/ParaConflict", split="test")
        for r in ds:
            conflict = str(r.get("Substitution Conflict", "") or "").strip()
            answer   = str(r.get("Answer", "") or "").strip()
            distract = str(r.get("Distracted Token", "") or "").strip()
            if not (conflict and answer and distract):
                continue
            mid, mem_surface = _best_alias_token(answer, tokenizer)
            did = _single_token_id(distract, tokenizer)
            if mid is None or did is None or mid == did:
                continue
            rows_out.append({
                "prompt": conflict, "domain": str(r.get("Category", "") or "").strip(),
                "memory_token_id": mid, "context_token_id": did,
                "memory_surface": mem_surface, "context_surface": distract,
            })
        src = "hf:gaotang/ParaConflict"

    if limit:
        rows_out = rows_out[:limit]
    log.info(f"[prompts] {family}/{variant}: {len(rows_out)} rows  (source={src})")
    return rows_out


# ════════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ════════════════════════════════════════════════════════════════════════════════
def load_model(family: str, variant: str, device: str):
    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from huggingface_hub import login as hf_login

    cfg      = FAMILIES[family]
    hf_name  = cfg[variant]
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        hf_login(token=hf_token, add_to_git_credential=False)
    if cfg["gated"] and not hf_token:
        raise EnvironmentError(f"{hf_name} is gated — set HF_TOKEN and accept the license.")

    tok = AutoTokenizer.from_pretrained(hf_name, token=hf_token) if hf_token \
        else AutoTokenizer.from_pretrained(hf_name)

    torch_dtype = getattr(torch, cfg.get("dtype", "float32"))
    tl_kwargs = dict(center_unembed=False, center_writing_weights=False,
                     fold_ln=True, dtype=torch.bfloat16)
    try:
        model = HookedTransformer.from_pretrained(hf_name, **tl_kwargs)
    except Exception as e:
        log.warning(f"Direct TL load failed ({e}); HF fallback.")
        hf_model = AutoModelForCausalLM.from_pretrained(
            hf_name, torch_dtype=torch.bfloat16, token=hf_token)
        model = HookedTransformer.from_pretrained(cfg["base"], hf_model=hf_model, **tl_kwargs)

    model.to(device).eval()
    assert model.cfg.n_layers == cfg["n_layers"], \
        f"{family}: expected {cfg['n_layers']} layers got {model.cfg.n_layers}"
    log.info(f"[model] {hf_name}: {model.cfg.n_layers}L × {model.cfg.n_heads}H "
             f"d_model={model.cfg.d_model} dtype={torch_dtype}")
    return model, tok


def ln_scale_final(model, cache, norm_type: str) -> torch.Tensor:
    """Linear approximation of the final norm as a per-prompt scalar. Shape [1,1]."""
    resid = cache[f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"][:, -1, :]
    eps = getattr(getattr(model, "ln_final", None), "eps", 1e-5)
    if norm_type == "rms":
        scale = 1.0 / ((resid ** 2).mean(-1, keepdim=True) + eps).sqrt()
    else:
        mean  = resid.mean(-1, keepdim=True)
        scale = 1.0 / (((resid - mean) ** 2).mean(-1, keepdim=True) + eps).sqrt()
    return scale


# ════════════════════════════════════════════════════════════════════════════════
# CORE: fixed superposition metric
# ════════════════════════════════════════════════════════════════════════════════
def _sign_pattern(ctx: float, mem: float) -> str:
    return ("both_pos" if ctx >= 0 and mem >= 0 else
            "both_neg" if ctx < 0 and mem < 0 else
            "ctx_pos_mem_neg" if ctx >= 0 > mem else "ctx_neg_mem_pos")


def _role_from_index(idx: float) -> str:
    if idx > INDEX_CONTEXT_T:  return "context"
    if idx < -INDEX_MEMORY_T:  return "memory"
    return "superposition"


def compute_superposition(model, prompts, heads, norm_type, label="") -> dict:
    """
    Direct Logit Attribution per head, averaged over prompts, classified by the
    signed, bounded context_index (NOT a signed ratio).
    """
    W_O, W_U = model.W_O, model.W_U
    ctx_acc = {hd: 0.0 for hd in heads}
    mem_acc = {hd: 0.0 for hd in heads}
    n = 0

    needed = {f"blocks.{l}.attn.hook_z" for l, _ in heads}
    needed.add(f"blocks.{model.cfg.n_layers - 1}.hook_resid_post")
    nf = lambda name: name in needed
    n_failed = 0

    for row in tqdm(prompts, desc=f"  sup {label}", unit="p"):
        try:
            tokens = model.to_tokens(row["prompt"])
            if tokens.shape[1] > model.cfg.n_ctx:
                tokens = tokens[:, -model.cfg.n_ctx:]
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=nf, return_type=None)
            scale = ln_scale_final(model, cache, norm_type).squeeze()
            mt, ct = row["memory_token_id"], row["context_token_id"]
            for (l, h) in heads:
                z      = cache[f"blocks.{l}.attn.hook_z"][0, -1, h, :]
                scaled = (z @ W_O[l, h]) * scale
                ctx_acc[(l, h)] += (scaled @ W_U[:, ct]).item()
                mem_acc[(l, h)] += (scaled @ W_U[:, mt]).item()
            del cache
            n += 1
        except Exception as e:                    # one bad prompt must not kill a 4h run
            n_failed += 1
            log.warning(f"  [{label}] forward failed (row skipped): {e}")
    if n == 0:
        raise RuntimeError(f"[{label}] every prompt failed ({n_failed} failures).")
    if n_failed:
        log.warning(f"  [{label}] skipped {n_failed} failed prompt(s); scored {n}.")

    heads_out = {}
    for (l, h) in heads:
        c = ctx_acc[(l, h)] / n
        m = mem_acc[(l, h)] / n
        index = (c - m) / (abs(c) + abs(m) + EPS)          # ← the fix, ∈ [-1, 1]
        legacy_ratio = (c / m) if (c > 0 and m > 0) else None   # only meaningful both-positive
        heads_out[f"L{l}H{h}"] = {
            "layer": l, "head": h,
            "ctx_pull": round(c, 6), "mem_pull": round(m, 6),
            # pull_magnitude exposes how much this head actually moves the two
            # logits. context_index normalises magnitude away, so a near-inert
            # head can still get a strong index — filter/weight by this to avoid
            # over-reading tiny heads. (LDS-selected heads are non-trivial, but
            # importance can come via indirect paths not captured by direct DLA.)
            "pull_magnitude": round(abs(c) + abs(m), 6),
            "context_index": round(index, 6),
            "role": _role_from_index(index),
            "sign_pattern": _sign_pattern(c, m),
            "legacy_ratio": (round(legacy_ratio, 4) if legacy_ratio is not None else None),
        }

    roles = Counter(v["role"] for v in heads_out.values())
    signs = Counter(v["sign_pattern"] for v in heads_out.values())
    log.info(f"  [{label}] n={n}  roles={dict(roles)}  signs={dict(signs)}")
    return {"n_prompts": n, "thresholds": {"context": INDEX_CONTEXT_T, "memory": -INDEX_MEMORY_T},
            "metric": "context_index=(ctx_pull-mem_pull)/(|ctx_pull|+|mem_pull|)",
            "heads": heads_out}


def compare_models(base_res, inst_res, node_diag) -> dict:
    """delta_index = inst_index - base_index (>0 ⇒ more context after tuning)."""
    per_head = {}
    for key, bh in base_res["heads"].items():
        ih = inst_res["heads"].get(key)
        if ih is None:
            continue
        per_head[key] = {
            "layer": bh["layer"], "head": bh["head"],
            "base_index": bh["context_index"], "inst_index": ih["context_index"],
            "delta_index": round(ih["context_index"] - bh["context_index"], 6),
            "base_role": bh["role"], "inst_role": ih["role"],
            "role_changed": bh["role"] != ih["role"],
            "shift_label": "stable" if bh["role"] == ih["role"] else f"{bh['role']}→{ih['role']}",
            "base_ctx_pull": bh["ctx_pull"], "base_mem_pull": bh["mem_pull"],
            "inst_ctx_pull": ih["ctx_pull"], "inst_mem_pull": ih["mem_pull"],
        }
    deltas = [v["delta_index"] for v in per_head.values()]
    changed = [v for v in per_head.values() if v["role_changed"]]
    summary = {
        "n_heads_compared": len(per_head),
        "n_role_changed": len(changed),
        "median_delta_index": round(float(np.median(deltas)), 6) if deltas else 0.0,
        "mean_delta_index":   round(float(np.mean(deltas)),   6) if deltas else 0.0,
        "frac_shift_context": round(float(np.mean([d > 0 for d in deltas])), 4) if deltas else 0.0,
        "note": "delta_index>0 ⇒ head became MORE context-biased after instruction tuning. "
                "Given instruct CRR falls (context is a FALSE injection), expect delta_index<0 "
                "for most heads — that is the coherent story, not a bug.",
    }
    return {"_node_overlap": node_diag, "_summary": summary, "heads": per_head}


# ════════════════════════════════════════════════════════════════════════════════
# ABLATION HARNESS  (generic: any head set; zero / mean / amplify; base + instruct)
# ════════════════════════════════════════════════════════════════════════════════
def compute_crr(model, prompts, hooks=None, label="") -> dict:
    n_ctx = n_mem = n_nei = 0
    for row in tqdm(prompts, desc=f"  crr {label}", unit="p"):
        try:
            tokens = model.to_tokens(row["prompt"])
            if tokens.shape[1] > model.cfg.n_ctx:
                tokens = tokens[:, -model.cfg.n_ctx:]
            with torch.no_grad():
                logits = (model.run_with_hooks(tokens, fwd_hooks=hooks, return_type="logits")
                          if hooks else model(tokens, return_type="logits"))
            pred = logits[0, -1, :].argmax().item()
            if   pred == row["context_token_id"]: n_ctx += 1
            elif pred == row["memory_token_id"]:  n_mem += 1
            else:                                 n_nei += 1
        except Exception as e:
            n_nei += 1
            log.warning(f"  [crr {label}] forward failed (counted neither): {e}")
    total = n_ctx + n_mem + n_nei
    return {"crr": n_ctx / total if total else 0.0,
            "n_context": n_ctx, "n_memory": n_mem, "n_neither": n_nei, "n_total": total}


def compute_mean_head_z(model, prompts, heads, limit=256) -> dict:
    """Dataset-mean of each head's final-position z, for mean-ablation (keeps the
    model on-distribution instead of the harsher zero-ablation)."""
    acc = {hd: None for hd in heads}
    needed = {f"blocks.{l}.attn.hook_z" for l, _ in heads}
    nf = lambda name: name in needed
    n = 0
    for row in tqdm(prompts[:limit], desc="  mean-z", unit="p"):
        try:
            tokens = model.to_tokens(row["prompt"])
            if tokens.shape[1] > model.cfg.n_ctx:
                tokens = tokens[:, -model.cfg.n_ctx:]
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=nf, return_type=None)
            for (l, h) in heads:
                z = cache[f"blocks.{l}.attn.hook_z"][0, -1, h, :].detach()
                acc[(l, h)] = z.clone() if acc[(l, h)] is None else acc[(l, h)] + z
            del cache
            n += 1
        except Exception as e:
            log.warning(f"  [mean-z] forward failed (row skipped): {e}")
    if n == 0:
        raise RuntimeError("mean-z: every prompt failed.")
    return {hd: (v / n) for hd, v in acc.items()}


def make_hooks(heads, mode: str, scale: float = 2.0, mean_z: dict | None = None):
    """mode ∈ {'zero','mean','amplify'}. Returns TransformerLens fwd_hooks."""
    hooks = []
    for (l, h) in heads:
        name = f"blocks.{l}.attn.hook_z"
        if mode == "zero":
            def fn(value, hook, hh=h):
                value[:, :, hh, :] = 0.0
                return value
        elif mode == "amplify":
            def fn(value, hook, hh=h, s=scale):
                value[:, :, hh, :] = value[:, :, hh, :] * s
                return value
        elif mode == "mean":
            mv = mean_z[(l, h)]
            def fn(value, hook, hh=h, m=mv):
                value[:, :, hh, :] = m.to(value.dtype).to(value.device)
                return value
        else:
            raise ValueError(mode)
        hooks.append((name, fn))
    return hooks


def run_ablation(model, prompts, family, variant, k=5, device="cuda") -> dict:
    """
    Four causal conditions on THIS model (run once per variant, base AND instruct):
        amplify_context / suppress_context (top-k context heads)
        amplify_memory  / suppress_memory  (top-k memory heads)   ← suppress_memory = Merge B
    Suppression uses MEAN-ablation (on-distribution) and also reports ZERO-ablation.
    """
    scores    = load_lds_scores(family, variant)
    ctx_heads = top_heads(scores, k, "context")
    mem_heads = top_heads(scores, k, "memory")
    all_heads = list(dict.fromkeys(ctx_heads + mem_heads))
    mean_z    = compute_mean_head_z(model, prompts, all_heads)

    base = compute_crr(model, prompts, hooks=None, label=f"{family}/{variant}/baseline")
    known = BASELINE_CRR.get((family, variant))
    log.info(f"[ablation] {family}/{variant} baseline CRR={base['crr']:.4f} (known={known})")

    conditions = [
        ("amplify_context",  ctx_heads, "amplify", "expect ΔCRR>0"),
        ("suppress_context", ctx_heads, "mean",    "expect ΔCRR<0"),
        ("suppress_context_zero", ctx_heads, "zero", "expect ΔCRR<0 (harsher)"),
        ("amplify_memory",   mem_heads, "amplify", "expect ΔCRR<0"),
        ("suppress_memory",  mem_heads, "mean",    "expect ΔCRR>0  [Merge B: validates LDS]"),
        ("suppress_memory_zero", mem_heads, "zero", "expect ΔCRR>0 (harsher)"),
    ]
    out = {"family": family, "variant": variant, "topk": k,
           "baseline_crr": base["crr"], "known_baseline_crr": known,
           "ctx_heads": [f"L{l}H{h}" for l, h in ctx_heads],
           "mem_heads": [f"L{l}H{h}" for l, h in mem_heads],
           "conditions": {}}

    want_up = {"amplify_context", "suppress_memory", "suppress_memory_zero"}
    for cname, heads, mode, expectation in conditions:
        hooks = make_hooks(heads, mode, mean_z=mean_z)
        res   = compute_crr(model, prompts, hooks=hooks, label=cname)
        delta = res["crr"] - base["crr"]
        matched = (delta > 0) if cname in want_up else (delta < 0)
        log.info(f"  [{cname:22s}] CRR={res['crr']:.4f} ΔCRR={delta:+.4f} "
                 f"{'OK' if matched else 'WRONG'}")
        out["conditions"][cname] = {
            "ablation_mode": mode, "crr": round(res["crr"], 6), "delta_crr": round(delta, 6),
            "matched_expectation": bool(matched), "expectation": expectation,
            "n_context": res["n_context"], "n_memory": res["n_memory"], "n_neither": res["n_neither"],
        }
    return out


# ════════════════════════════════════════════════════════════════════════════════
# DLA RECONSTRUCTION CHECK  — is the superposition math correctly scaled?
# ════════════════════════════════════════════════════════════════════════════════
def verify_dla(model, prompts, norm_type: str, family: str, n: int = 8) -> dict:
    """
    Definitive test of whether our Direct-Logit-Attribution scaling is correct.

    The DLA identity: after the final norm, the model's logit for token t is
        logit_t = (resid_final * ln_scale) @ W_U[:, t]  (+ b_U[t])
    If OUR ln_scale + W_U handling is right, `reconstructed` matches the model's
    own `actual` logit (up to the final-logit soft-cap, which we report).

    If reconstructed ≈ actual  → scaling is CORRECT; tiny Gemma per-head pulls are
                                 REAL (heads act indirectly), not a code bug.
    If reconstructed ≈ 100× off → the bug is in our final-norm / unembed scaling.
    """
    last    = model.cfg.n_layers - 1
    softcap = getattr(model.cfg, "final_logit_softcap", None)
    b_U     = getattr(model, "b_U", None)
    log.info(f"[verify:{family}] final_logit_softcap={softcap}  has_b_U={b_U is not None}")
    nf = lambda nm: nm == f"blocks.{last}.hook_resid_post"

    rows_out = []
    for row in prompts[:n]:
        tokens = model.to_tokens(row["prompt"])
        if tokens.shape[1] > model.cfg.n_ctx:
            tokens = tokens[:, -model.cfg.n_ctx:]
        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens, names_filter=nf, return_type="logits")
        resid = cache[f"blocks.{last}.hook_resid_post"][0, -1, :]
        rms   = (resid ** 2).mean().sqrt().item()
        scale = ln_scale_final(model, cache, norm_type).squeeze()
        for who, tok in (("ctx", row["context_token_id"]), ("mem", row["memory_token_id"])):
            recon = ((resid * scale) @ model.W_U[:, tok]).item()
            if b_U is not None:
                recon += b_U[tok].item()
            actual = logits[0, -1, tok].item()
            # If a soft-cap is active, the linear recon is PRE-cap; map it through
            # the cap so it is comparable to the model's post-cap `actual`.
            recon_capped = (softcap * math.tanh(recon / softcap)) if softcap else recon
            rows_out.append({"who": who, "resid_rms": round(rms, 3),
                             "ln_scale": round(scale.item(), 6),
                             "recon_raw": round(recon, 4),
                             "recon_capped": round(recon_capped, 4),
                             "actual": round(actual, 4),
                             "abs_err_vs_actual": round(abs(recon_capped - actual), 4)})
            log.info(f"[verify:{family}] {who}: rms={rms:8.2f} scale={scale.item():.6f} "
                     f"recon={recon:+.3f} (capped {recon_capped:+.3f}) actual={actual:+.3f}")

    errs = [r["abs_err_vs_actual"] for r in rows_out]
    med  = float(np.median(errs))
    verdict = ("SCALING OK — DLA reconstructs the model logits; small Gemma pulls are REAL"
               if med < 0.5 else
               "SCALING SUSPECT — reconstruction does not match; likely a DLA bug for this model")
    log.info(f"[verify:{family}] median |recon_capped - actual| = {med:.4f}  →  {verdict}")
    return {"family": family, "final_logit_softcap": softcap,
            "median_abs_err": round(med, 4), "verdict": verdict, "samples": rows_out}


# ════════════════════════════════════════════════════════════════════════════════
# DRIVER
# ════════════════════════════════════════════════════════════════════════════════
def run_family(family: str, mode: str, device: str, topk: int, limit: int | None):
    setup_logging(family)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fam_dir = OUT_DIR / family
    fam_dir.mkdir(parents=True, exist_ok=True)
    cfg = FAMILIES[family]
    log.info(f"===== {family.upper()}  mode={mode}  device={device} =====")

    union, node_diag = build_union_head_set(family, topk)
    log.info(f"[heads] union set = {len(union)} heads; node overlap: {node_diag['interpretation']}")

    # verify mode: load base only, run the DLA reconstruction check, done.
    if mode == "verify":
        model, tok = load_model(family, "base", device)
        prompts    = load_prompts(family, "base", tok, limit=limit)
        vr = verify_dla(model, prompts, cfg["norm_type"], family)
        (fam_dir / f"verify_dla_{family}.json").write_text(json.dumps(vr, indent=2))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        return

    sup_res = {}
    for variant in ("base", "instruct"):
        t0 = time.time()
        model, tok = load_model(family, variant, device)
        prompts    = load_prompts(family, variant, tok, limit=limit)

        if mode in ("superposition", "all"):
            res = compute_superposition(model, prompts, union, cfg["norm_type"], f"{family}/{variant}")
            res.update({"family": family, "variant": variant, "model": cfg[variant]})
            (fam_dir / f"superposition_{family}_{variant}.json").write_text(json.dumps(res, indent=2))
            sup_res[variant] = res

        if mode in ("ablation", "all"):
            ab = run_ablation(model, prompts, family, variant, k=5, device=device)
            (fam_dir / f"ablation_{family}_{variant}.json").write_text(json.dumps(ab, indent=2))

        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        log.info(f"[{family}/{variant}] done in {time.time()-t0:.0f}s")

    if mode in ("superposition", "all") and {"base", "instruct"} <= sup_res.keys():
        comp = compare_models(sup_res["base"], sup_res["instruct"], node_diag)
        (fam_dir / f"comparison_{family}.json").write_text(json.dumps(comp, indent=2))
        log.info(f"[{family}] comparison → median Δindex={comp['_summary']['median_delta_index']:+.4f}, "
                 f"role changed {comp['_summary']['n_role_changed']}/{comp['_summary']['n_heads_compared']}, "
                 f"node overlap: {node_diag['interpretation']}")


def parse_args():
    p = argparse.ArgumentParser(description="Fixed superposition + ablation harness")
    p.add_argument("--family", choices=list(FAMILIES))
    p.add_argument("--all", action="store_true")
    p.add_argument("--mode", choices=["superposition", "ablation", "all", "verify"], default="all",
                   help="'verify' runs the DLA reconstruction check (is the scaling correct?)")
    p.add_argument("--topk", type=int, default=10, help="top-k heads per role for the union set")
    p.add_argument("--limit", type=int, default=None, help="cap #prompts (smoke testing)")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default=None,
                   help="override model dtype (e.g. --dtype float32 to test Gemma in fp32)")
    p.add_argument("--device", default=None)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    fams = list(FAMILIES) if a.all else ([a.family] if a.family else [])
    if not fams:
        print("Specify --family qwen|llama|gemma  or  --all"); sys.exit(1)
    if a.dtype:                                   # CLI override of the per-family default
        for f in fams:
            FAMILIES[f]["dtype"] = a.dtype
    for fam in fams:
        run_family(fam, a.mode, dev, a.topk, a.limit)
