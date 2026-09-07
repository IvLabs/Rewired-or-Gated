"""run_gemma3.py — EAP-screen → PP-verify pipeline for Gemma-3-4B.

Drives the existing model-agnostic pipeline (run_lds / run_pp_topheads / triangulation)
for the Gemma-3-4B base/instruct pair. Produces the same file layout as the Qwen and
Llama runs and writes results/gemma3_4b_summary.json at the end.

Usage:
  # Base model — both prompt types + summary
  python run_gemma3.py --variant base

  # Instruct model
  python run_gemma3.py --variant instruct

  # Both models back-to-back (runs base first, then instruct)
  python run_gemma3.py --variant both

  # Smoke test (16-prompt subset, skip summary)
  python run_gemma3.py --variant base --n-subset 16

  # Build summary only (after both cells exist)
  python run_gemma3.py --summary-only

All outputs → results/. Files are skipped if they already exist (resumable).

Access prereqs:
  1. transformer-lens must support Gemma-3 — verify before anything else.
     If `HookedTransformer.from_pretrained("google/gemma-3-4b-pt")` raises ValueError,
     bump transformer-lens (see spec §3 for fallbacks).
  2. HF_TOKEN in .env must belong to an account that accepted the Gemma license on
     BOTH google/gemma-3-4b-pt AND google/gemma-3-4b-it (acceptance is per-repo).
     A 404 on either repo means unaccepted/unauthenticated, not a bad repo id.
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if sys.platform == "win32":
    # This script prints unicode (rho, >=) in build_summary()'s gate-pass
    # line -- Windows' default console codepage (cp1252) crashes on those
    # with UnicodeEncodeError. Force UTF-8 stdout/stderr regardless of the
    # user's console/env setup, so this never depends on remembering
    # PYTHONUTF8=1 or chcp 65001.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import torch

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Path patching lives in its own sub-package.
sys.path.insert(0, os.path.join(_ROOT, "path_patching"))

from foundation import load_model, compute_crr, compute_crr_logprob
from lds_attribution import run_and_save, enable_gradient_checkpointing
from path_patching import run_path_patching
from contract import HeadScores
from singleton_fix.single_token_loader import (
    load_single_token_prompts,
    load_cached_single_token_prompts,
    print_domain_composition,
)
from singleton_fix.provenance import build_provenance

FAMILY = "gemma3_4b"  # per-family results folder, mirrors singleton_fix.run_single_token_filter_all
RESULTS_DIR = Path(_ROOT) / "results" / FAMILY
PROMPT_TYPES = ("substitution", "coherent")
# Path Patching now runs on BOTH prompt types (2026-07-10). The original
# spec Part H gate (substitution-only) existed *solely* because the old
# prefix∪suffix alignment_indices dropped the whole middle span of coherent
# prompts (where the distractor repeats). alignment_indices now uses a
# position-by-position diff whenever the swap is length-preserving -- which
# it always is for the single-token backbone (distractor and memory answer
# are both single tokens), verified on all 3 real tokenizers: coherent swaps
# are length-preserving in 600/600 sampled rows, so the new alignment keeps
# ~115/118 positions vs the old ~45, and is byte-identical to the old one on
# substitution (100/100 rows). So the Part H alignment objection is resolved
# and both prompt types are now scored the same way.
PP_PROMPT_TYPES = ("substitution", "coherent")

# ---------------------------------------------------------------------------
# Frozen config
# ---------------------------------------------------------------------------

VARIANTS = {
    "base": {
        "model":    "google/gemma-3-4b-pt",
        "tag":      "gemma3_4b_base",
        "hf_model": None,
    },
    "instruct": {
        "model":    "google/gemma-3-4b-it",
        "tag":      "gemma3_4b_instruct",
        "hf_model": None,
    },
}
DTYPE = "bfloat16"
PP_K  = 10   # top-k per direction → ~20 heads sent to PP
SET_TAG = "st"  # spec §A3.2/§Part-E — distinguishes new single-token results from the old "full" tag
ALIAS_POLICY = "first_single_token"  # spec §A2.1 only. §A2.2 log-prob refinement
# was disabled as the canonical default on 2026-07-09 -- it picked different
# aliases for base vs instruct on the same row (e.g. base="football" vs
# instruct="soccer"), confounding the base-vs-instruct comparison. A2.1 is
# tokenizer-only, so base/instruct always agree within a family.


# ---------------------------------------------------------------------------
# Tokenizer round-trip sanity (spec §6 — run before attribution)
# ---------------------------------------------------------------------------

def _check_tokenizer(model) -> None:
    """Verify _answer_ids round-trips correctly for Gemma's SentencePiece tokenizer.

    Gemma uses SentencePiece with a ▁ leading-space meta-symbol, which may merge
    or strip the prepended space differently from BPE. This is the exact failure
    mode flagged in the _answer_ids docstring. A silent failure here means every
    log-prob is wrong.
    """
    from foundation import _answer_ids
    probe = "soccer"
    ids = _answer_ids(model, probe)
    decoded = model.tokenizer.decode(ids.tolist())
    if probe not in decoded:
        raise RuntimeError(
            f"[tokenizer-check] FAIL: _answer_ids('{probe}') decoded to {decoded!r}. "
            f"The SentencePiece leading-space convention is not round-tripping correctly "
            f"for this tokenizer. Fix _answer_ids in foundation.py before running attribution."
        )
    print(
        f"[tokenizer-check] OK: _answer_ids('{probe}') → {ids.tolist()} → {decoded!r} "
        f"(contains '{probe}')"
    )
    max_id = ids.max().item()
    if max_id >= model.cfg.d_vocab:
        raise RuntimeError(
            f"[tokenizer-check] FAIL: answer id {max_id} out of vocab range "
            f"(d_vocab={model.cfg.d_vocab})."
        )
    print(f"[tokenizer-check] ids in-vocab (max={max_id} < d_vocab={model.cfg.d_vocab}): OK")


# ---------------------------------------------------------------------------
# Head selection (mirrors run_pp_topheads.py logic)
# ---------------------------------------------------------------------------

def _load_eap(path: Path) -> Dict[Tuple[int, int], float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    scores = raw.get("scores", raw) if isinstance(raw, dict) and "_provenance" in raw else raw
    return {(int(k.split("_")[0]), int(k.split("_")[1])): float(v) for k, v in scores.items()}


def _load_gradact(tag: str, ptype: str, set_tag: str = SET_TAG) -> Dict[Tuple[int, int], float]:
    """Spec §A4.1 — Merge A MUST correlate PP against gradact (single-run, no
    counterfactual), never against eap (which, after spec §A1, shares PP's
    exact contrast -- correlating them would be partly circular)."""
    path = RESULTS_DIR / f"lds2_{tag}_{ptype}_{set_tag}_gradact.json"
    return _load_eap(path)  # same flat {"l_h": v} shape, reuse the loader


def _load_prompts(
    model,
    variant: str,
    ptype: str,
    heldout_domain: Optional[str] = None,
    split: str = "all",
) -> List:
    """Prefer the cached single-token survival file (no model reload, no
    re-running the GPU-bound A2.2 oracle, no ParaConflict re-fetch) --
    fall back to live filtering if the cache doesn't exist yet for this
    variant (e.g. singleton_fix/run_single_token_filter_all.py or
    singleton_fix/modal_single_token_filter.py hasn't been run for it)."""
    try:
        return load_cached_single_token_prompts(
            FAMILY, variant, prompt_type=ptype, heldout_domain=heldout_domain, split=split,
        )
    except FileNotFoundError:
        print(
            f"[warn] no cached single-token survival file for {FAMILY}/{variant} -- "
            f"falling back to live filtering (slower; run "
            f"singleton_fix/run_single_token_filter_all.py first to avoid this)."
        )
        return load_single_token_prompts(
            model, prompt_type=ptype, heldout_domain=heldout_domain, split=split,
        )


def _select_top_heads(eap: Dict[Tuple[int, int], float], k: int) -> List[Tuple[int, int]]:
    if k <= 0 or not eap:
        return []
    k = min(k, len(eap))
    asc  = sorted(eap.items(), key=lambda kv: kv[1])
    desc = sorted(eap.items(), key=lambda kv: kv[1], reverse=True)
    ctx_heads = [h for h, _ in desc[:k]]
    mem_heads = [h for h, _ in asc[:k]]
    seen = set(ctx_heads)
    union = list(ctx_heads)
    for h in mem_heads:
        if h not in seen:
            union.append(h)
            seen.add(h)
    return union


# ---------------------------------------------------------------------------
# Per-cell CRR run (one-run method -- safe on both prompt types, spec Part H)
# ---------------------------------------------------------------------------

def _run_crr(
    model,
    tag: str,
    variant: str,
    ptype: str,
    set_tag: str = SET_TAG,
    heldout_domain: Optional[str] = None,
    split: str = "all",
    n_subset: Optional[int] = None,
    seed: int = 42,
) -> Optional[Dict]:
    """Run behavioral + log-prob-proxy CRR for one (tag, ptype) cell. Skip if output exists."""
    out_path = RESULTS_DIR / f"crr_{tag}_{ptype}_{set_tag}.json"
    if out_path.exists():
        print(f"[crr] skip {tag}/{ptype} (already done → {out_path})")
        return json.loads(out_path.read_text(encoding="utf-8"))

    prompts = _load_prompts(model, variant, ptype, heldout_domain=heldout_domain, split=split)
    if n_subset is not None:
        import random as _random
        rng = _random.Random(seed)
        prompts = rng.sample(prompts, min(n_subset, len(prompts)))
        print(f"[crr] smoke: using {len(prompts)} prompts for {tag}/{ptype}")

    behavioral = compute_crr(model, prompts)
    logprob = compute_crr_logprob(model, prompts)

    result = {
        "_provenance": build_provenance(
            contrast="none_single_run",
            dataset=f"single_token_{set_tag}",
            n_used=behavioral.get("total", 0),
            alias_policy=ALIAS_POLICY,
            prompt_type=ptype,
            model_tag=tag,
        ),
        "model": tag, "prompt_type": ptype,
        "crr_behavioral": behavioral,
        "crr_logprob": logprob,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[crr] saved → {out_path}")
    return result


# ---------------------------------------------------------------------------
# Per-cell PP run (adapted from run_pp_topheads.main)
# ---------------------------------------------------------------------------

def _run_pp(
    model,
    tag: str,
    variant: str,
    ptype: str,
    n_subset: Optional[int],
    seed: int,
    set_tag: str = SET_TAG,
    heldout_domain: Optional[str] = None,
    split: str = "all",
) -> Optional[Dict]:
    """Run PP-topheads for one (tag, ptype) cell. Skip if output exists."""
    out_path = RESULTS_DIR / f"pp_topheads_{tag}_{ptype}_{set_tag}.json"
    if out_path.exists():
        print(f"[pp] skip {tag}/{ptype} (already done → {out_path})")
        return json.loads(out_path.read_text(encoding="utf-8"))

    eap_path = RESULTS_DIR / f"lds2_{tag}_{ptype}_{set_tag}_eap.json"
    if not eap_path.exists():
        print(f"[pp] ERROR: EAP file missing: {eap_path}. Run EAP screen first.", file=sys.stderr)
        return None

    eap_scores = _load_eap(eap_path)
    gradact_scores = _load_gradact(tag, ptype, set_tag)
    top_heads  = _select_top_heads(eap_scores, PP_K)
    if not top_heads:
        print(f"[pp] ERROR: no heads selected from {eap_path}.", file=sys.stderr)
        return None

    sorted_desc = sorted(eap_scores.items(), key=lambda kv: kv[1], reverse=True)
    eap_rank    = {h: i + 1 for i, (h, _) in enumerate(sorted_desc)}
    ctx_set     = set(h for h, _ in sorted_desc[:PP_K])

    ctx_disp = [(h, v) for h, v in sorted_desc if h in set(top_heads)][:PP_K]
    mem_disp = [(h, v) for h, v in reversed(sorted_desc) if h in set(top_heads)][:PP_K]
    print(f"[pp] {tag}/{ptype} — {len(top_heads)} heads (k={PP_K}/direction):")
    print(f"  ctx: {[f'L{l}H{h}={v:.3f}' for (l,h),v in ctx_disp]}")
    print(f"  mem: {[f'L{l}H{h}={v:.3f}' for (l,h),v in mem_disp]}")

    prompts = _load_prompts(model, variant, ptype, heldout_domain=heldout_domain, split=split)
    meta: Dict = {}
    # Crash-resilient: checkpoint next to the output file so a hard crash
    # (CUDA death / OOM / kill) during a long coherent cell resumes instead of
    # losing all scored prompts. Auto-removed on successful completion.
    scores = run_path_patching(
        model, prompts,
        n_subset=n_subset,
        seed=seed,
        head_subset=top_heads,
        checkpoint_path=str(out_path) + ".checkpoint.json",
        _meta_out=meta,
    )

    # Build selected_heads list (mirrors run_pp_topheads._build_result_json)
    selected = []
    for h in top_heads:
        direction = "context" if h in ctx_set else "memory"
        selected.append({
            "layer": h[0], "head": h[1],
            "eap_score": eap_scores.get(h, float("nan")),
            "eap_rank":  eap_rank.get(h),
            "gradact_score": gradact_scores.get(h, float("nan")),
            "direction": direction,
            "pp_score":  scores.get(h, float("nan")),
        })

    result = {
        "_provenance": build_provenance(
            contrast="inplace_swap",
            dataset=f"single_token_{set_tag}",
            n_used=meta.get("n_prompts_used", 0),
            alias_policy=ALIAS_POLICY,
            prompt_type=ptype,
            model_tag=tag,
        ),
        "model": tag, "prompt_type": ptype, "k": PP_K,
        "source_eap_file": str(eap_path),
        "selected_heads": selected,
        "head_scores": {f"{l}_{h}": v for (l, h), v in scores.items()},
        "n_prompts_input": meta.get("n_prompts_input", 0),
        "n_prompts_used":  meta.get("n_prompts_used", 0),
        "noise_floor":     meta.get("noise_floor", {}),
        "meta": {
            "n_heads_scored":              meta.get("n_heads_scored", len(top_heads)),
            "n_skipped_degenerate_margin": meta.get("n_skipped_degenerate_margin", 0),
            "n_skipped_length_mismatch":   meta.get("n_skipped_length_mismatch", 0),
            "n_skipped_no_swap":           meta.get("n_skipped_no_swap", 0),
            "n_skipped_error":             meta.get("n_skipped_error", 0),
            "n_skipped_clean_neither":     meta.get("n_skipped_clean_neither", 0),
            "torch_version":               torch.__version__,
        },
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[pp] saved → {out_path}")
    return result


# ---------------------------------------------------------------------------
# Summary builder (mirrors qwen25_3b_summary.json / llama32_3b_summary.json schema)
# ---------------------------------------------------------------------------

def _spearman(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    """Spearman ρ via stdlib (mirrors triangulation.py — pure ranks, no scipy)."""
    n = len(xs)
    if n < 3:
        return float("nan"), float("nan")

    def _ranks(vals):
        order = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        for rank, idx in enumerate(order):
            r[idx] = rank + 1.0
        return r

    rx, ry = _ranks(xs), _ranks(ys)
    mean_x = sum(rx) / n
    mean_y = sum(ry) / n
    num = sum((rx[i] - mean_x) * (ry[i] - mean_y) for i in range(n))
    dx  = sum((rx[i] - mean_x) ** 2 for i in range(n)) ** 0.5
    dy  = sum((ry[i] - mean_y) ** 2 for i in range(n)) ** 0.5
    if dx == 0 or dy == 0:
        return float("nan"), float("nan")
    rho = num / (dx * dy)

    import math
    if abs(rho) >= 1.0:
        return rho, 0.0
    t = rho * math.sqrt((n - 2) / (1 - rho ** 2))
    p = math.erfc(abs(t) / math.sqrt(2 * (n - 2)))
    return rho, p


def _build_cell_summary(tag: str, ptype: str) -> Optional[Dict]:
    pp_path  = RESULTS_DIR / f"pp_topheads_{tag}_{ptype}_{SET_TAG}.json"
    eap_path = RESULTS_DIR / f"lds2_{tag}_{ptype}_{SET_TAG}_eap.json"
    if not pp_path.exists() or not eap_path.exists():
        print(f"[summary] missing files for {tag}/{ptype} — skip.", file=sys.stderr)
        return None

    pp_data  = json.loads(pp_path.read_text(encoding="utf-8"))
    selected = pp_data["selected_heads"]

    # Spec §A4.1: correlate PP against gradact, NOT eap -- after the §A1 fix,
    # eap and PP share the exact same in-place-swap contrast, so an eap-vs-PP
    # correlation would be partly circular (EAP is PP's first-order
    # approximation of the same intervention). gradact is single-run (no
    # counterfactual) and therefore a genuinely independent check.
    gradact_vals = [s["gradact_score"] for s in selected]
    pp_vals      = [s["pp_score"]      for s in selected]
    rho, pval = _spearman(gradact_vals, pp_vals)

    sign_agree = sum(
        1 for e, p in zip(gradact_vals, pp_vals)
        if (e > 0 and p > 0) or (e < 0 and p < 0)
    )

    top_ctx = sorted(
        [s for s in selected if s["direction"] == "context"],
        key=lambda s: s["eap_score"], reverse=True,
    )[:PP_K]
    top_mem = sorted(
        [s for s in selected if s["direction"] == "memory"],
        key=lambda s: s["eap_score"],
    )[:PP_K]

    return {
        "spearman_rho":    round(rho, 4),
        "p_value":         round(pval, 6),
        "sign_agreement":  f"{sign_agree}/{len(selected)}",
        "gate_pass":       rho >= 0.5,
        "n_prompts_input": pp_data.get("n_prompts_input", 0),
        "n_prompts_used":  pp_data.get("n_prompts_used", 0),
        "noise_floor":     pp_data.get("noise_floor", {}),
        "top_heads": [
            {
                "head":      f"L{s['layer']}H{s['head']}",
                "direction": s["direction"],
                "eap_score": round(s["eap_score"], 4),
                "eap_rank":  s["eap_rank"],
                "pp_score":  round(s["pp_score"], 4),
            }
            for s in (top_ctx + top_mem)
        ],
    }


def build_summary() -> None:
    out_path = RESULTS_DIR / f"gemma3_4b_summary_{SET_TAG}.json"
    summary  = {
        "experiment": "Gemma-3-4B EAP-screen -> PP-verify",
        "method":     "multi-token mean log-prob",
        "models": {},
    }

    total_n_used = 0
    for variant, cfg in VARIANTS.items():
        tag   = cfg["tag"]
        cells = {}
        for ptype in PROMPT_TYPES:
            cell = _build_cell_summary(tag, ptype)
            if cell is not None:
                cells[ptype] = cell
                total_n_used += cell.get("n_prompts_used", 0)
        if cells:
            summary["models"][tag] = {
                "model_id": cfg["model"],
                "cells": cells,
            }
        gate_results = [c.get("gate_pass") for c in cells.values()]
        passes = sum(1 for g in gate_results if g)
        print(f"[summary] {tag}: {passes}/{len(gate_results)} cells pass ρ≥0.5 gate")

    # spans multiple models/prompt-types, so model_tag/prompt_type are "all"
    summary["_provenance"] = build_provenance(
        contrast="inplace_swap", dataset=f"single_token_{SET_TAG}",
        n_used=total_n_used, alias_policy=ALIAS_POLICY,
        prompt_type="all", model_tag="all_variants",
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[summary] saved → {out_path}")


# ---------------------------------------------------------------------------
# Per-variant runner
# ---------------------------------------------------------------------------

def run_variant(
    variant: str,
    n_subset: Optional[int],
    no_shuffle: bool,
    seed: int,
    heldout_domain: Optional[str] = None,
    split: str = "all",
    steps: Tuple[str, ...] = ("lds", "crr", "pp"),
) -> None:
    """steps: which stage(s) to run -- any subset of {"lds", "crr", "pp"}.
    "pp" requires "lds" to have already produced this variant/prompt_type's
    lds2_..._eap.json (either in this same call, or a prior run with "lds"
    in steps) since PP selects its top-k heads from the EAP scores. Lets
    run_all_lds.py / run_all_pathpatching.py drive LDS and PP as separate
    passes across all 6 model variants without duplicating this function's
    model-loading/prompt-loading logic."""
    cfg        = VARIANTS[variant]
    tag        = cfg["tag"]
    model_name = cfg["model"]
    hf_model   = cfg["hf_model"]
    # Spec §B4.3: a held-out-domain run gets its own set_tag so it can never
    # collide with (or silently overwrite) the default all-domains results.
    set_tag = SET_TAG if split == "all" else f"{SET_TAG}_{split}"

    print(f"\n{'='*70}")
    print(f"[gemma3] variant={variant!r}  tag={tag!r}  model={model_name!r}")
    print(f"{'='*70}")

    print(f"[gemma3] Loading model (dtype={DTYPE}) ...")
    # Warn early if config.json is not in HF cache — avoids a silent download stall.
    try:
        from huggingface_hub import try_to_load_from_cache
        _cached = try_to_load_from_cache(model_name, "config.json")
        if _cached is None:
            print(
                f"[gemma3] WARNING: {model_name} not found in HF cache. "
                f"TL will attempt a live download — run dl_gemma.py first if "
                f"the connection is unreliable."
            )
        else:
            print(f"[gemma3] HF cache hit for {model_name} — loading from cache.")
    except Exception:
        pass  # try_to_load_from_cache may not exist in older hub versions
    # Let TL load Gemma-3 natively (no hf_model_name override).
    # Passing an externally-loaded AutoModelForCausalLM breaks TL's
    # convert_gemma_weights: newer transformers returns Gemma3Model whose
    # .embed_tokens attribute TL can't find via its expected path. TL's own
    # internal loader uses kwargs that produce the right class layout.
    # dl_gemma.py (huggingface_hub >= 0.24) populates ~/.cache/huggingface,
    # so TL's hf_hub_download finds the files there without re-fetching.
    kwargs = {}
    if hf_model:
        kwargs["hf_model_name"] = hf_model
    model = load_model(model_name, dtype=DTYPE, **kwargs)
    enable_gradient_checkpointing(model)
    print(
        f"[gemma3] cfg: n_layers={model.cfg.n_layers} n_heads={model.cfg.n_heads} "
        f"d_model={model.cfg.d_model} d_vocab={model.cfg.d_vocab} "
        f"dtype={next(model.parameters()).dtype}"
    )
    # d_vocab ≈ 262144 is the tell that the text config (not a vision/wrong config) loaded.
    if model.cfg.d_vocab < 200_000:
        print(
            f"[gemma3] WARNING: d_vocab={model.cfg.d_vocab} is unexpectedly small for "
            f"Gemma-3 (expected ~262144). This may indicate a wrong or multimodal-only "
            f"config was loaded — verify before trusting any results."
        )

    # Tokenizer sanity (spec §6 — fast check, stops early if SentencePiece wiring is broken)
    _check_tokenizer(model)

    run_shuffle = not no_shuffle

    # --- EAP screen (both prompt types) ---
    if "lds" in steps:
        for ptype in PROMPT_TYPES:
            prompts = _load_prompts(model, variant, ptype, heldout_domain=heldout_domain, split=split)
            print_domain_composition(prompts)
            if n_subset is not None:
                import random as _random
                rng = _random.Random(seed)
                prompts = rng.sample(prompts, min(n_subset, len(prompts)))
                print(f"[gemma3] EAP smoke: using {len(prompts)} prompts for {ptype}")

            run_and_save(
                model, prompts,
                model_tag=tag,
                prompt_type=ptype,
                set_tag=set_tag,
                run_shuffle=run_shuffle,
                results_dir=RESULTS_DIR,
                alias_policy=ALIAS_POLICY,
            )

    # --- CRR (behavioral + logprob, both prompt types -- one-run, safe) ---
    if "crr" in steps:
        for ptype in PROMPT_TYPES:
            _run_crr(
                model, tag, variant, ptype, set_tag=set_tag, heldout_domain=heldout_domain,
                split=split, n_subset=n_subset, seed=seed,
            )

    # --- PP verify (substitution only -- spec Part H) ---
    if "pp" in steps:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for ptype in PP_PROMPT_TYPES:
            eap_path = RESULTS_DIR / f"lds2_{tag}_{ptype}_{set_tag}_eap.json"
            if not eap_path.exists():
                print(
                    f"[gemma3] ERROR: {eap_path} missing -- run LDS for this variant "
                    f"first (run_all_lds.py, or run_variant(..., steps=('lds',...))).",
                    file=sys.stderr,
                )
                continue
            _run_pp(
                model, tag, variant, ptype, n_subset=n_subset, seed=seed, set_tag=set_tag,
                heldout_domain=heldout_domain, split=split,
            )

    # Free this variant's model from GPU memory before the orchestrator
    # loads the next one -- don't rely on GC alone.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[gemma3] variant={variant!r} done (steps={steps}).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="EAP-screen → PP-verify for Gemma-3-4B base/instruct.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--variant", default="base",
        choices=["base", "instruct", "both"],
        help="Which model(s) to run. 'both' runs base then instruct. Default: base.",
    )
    parser.add_argument(
        "--n-subset", type=int, default=None, dest="n_subset",
        help="Run on N prompts (smoke test). Omit for the full set.",
    )
    parser.add_argument(
        "--no-shuffle", action="store_true",
        help="Skip the shuffled-label noise-floor control.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for subset sampling and PP random-pair baseline. Default: 42.",
    )
    parser.add_argument(
        "--summary-only", action="store_true", dest="summary_only",
        help="Skip all model runs; just rebuild gemma3_4b_summary.json from existing files.",
    )
    parser.add_argument(
        "--heldout-domain", default=None, dest="heldout_domain",
        help="Spec §B4.3 held-out-domain transfer split, e.g. 'Company Headquarter' "
             "(the recommended choice -- see singleton_fix.single_token_loader."
             "RECOMMENDED_HELDOUT_DOMAIN). Omit for the default: no split, all domains.",
    )
    parser.add_argument(
        "--split", default="all", choices=["train", "heldout", "all"],
        help="Which half of --heldout-domain to load: 'train' (every other domain), "
             "'heldout' (only that domain), or 'all' (default, no split -- "
             "--heldout-domain is then ignored).",
    )
    args = parser.parse_args()

    if args.summary_only:
        build_summary()
        return

    variants = ["base", "instruct"] if args.variant == "both" else [args.variant]
    for v in variants:
        run_variant(
            v, n_subset=args.n_subset, no_shuffle=args.no_shuffle, seed=args.seed,
            heldout_domain=args.heldout_domain, split=args.split,
        )

    if args.n_subset is None and args.split == "all":
        # Only auto-build the combined summary after a full, non-split run --
        # a held-out split produces differently-tagged files that
        # build_summary() (which always reads the plain SET_TAG) won't find.
        build_summary()
    else:
        print("[gemma3] Smoke/split run complete — skipping auto summary.")


if __name__ == "__main__":
    main()
