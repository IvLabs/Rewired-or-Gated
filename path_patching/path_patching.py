"""Path patching attribution for GPT-2 prototype.

Implements activation patching at hook_z (per-head, pre-OV projection) as
described in PP_Spec.md v2. Key design decisions:
  - Suffix alignment for sequence-length mismatch (§3)
  - Four-condition degenerate-prompt filter (§4)
  - ctx-minus-memory margin direction to match LDS sign (§7)
  - Trimmed mean (10%) aggregation (§13)
  - Noise-floor baselines: identity-patch and random-pair (§8)

See PP_Spec.md for full rationale.
"""

from __future__ import annotations

import os
import sys

# Make contract.py and foundation.py importable from the project root.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import json
import subprocess
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformer_lens.utilities import get_act_name as _get_act_name

from contract import ConflictPrompt, HeadScores
from foundation import _answer_ids, answer_log_prob
from singleton_fix.contrast_pair import build_counterfactual_text, build_contrast_pair, alignment_indices, swap_occurred
from singleton_fix.provenance import build_provenance

# Regex string is treated as EXACT MATCH in TransformerLens names_filter.
# Use a lambda to get substring matching across all hook_z hooks.
_HOOK_Z_FILTER = lambda n: "hook_z" in n  # noqa: E731


def _cache_get(cache, hook_name: str) -> torch.Tensor:
    """Get a hook tensor from either a TransformerLens ActivationCache or a plain dict.

    ActivationCache.__getitem__ passes the key through get_act_name() which
    mangles full hook-name strings — access cache_dict directly to avoid that.
    """
    if hasattr(cache, "cache_dict"):
        return cache.cache_dict[hook_name]
    return cache[hook_name]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _trimmed_mean(scores: List[float], trim: float) -> float:
    """Trimmed mean: drop top and bottom `trim` fraction, average the rest."""
    if not scores:
        return float("nan")
    arr = sorted(scores)
    n = len(arr)
    cut = int(n * trim)
    # Guard: if trimming would leave nothing, fall back to plain mean.
    if 2 * cut >= n:
        return float(np.mean(arr))
    return float(np.mean(arr[cut: n - cut]))


def _token_length(model, text: str) -> int:
    """Number of tokens in `text` according to `model.to_tokens`."""
    return model.to_tokens(text).shape[1]


def _make_aligned_patch_fn(
    source_z: torch.Tensor,
    head_idx: int,
    corrupt_idx: torch.Tensor,
    source_idx: torch.Tensor,
    L_full: int,
):
    """Return a hook that patches `source_z[head_idx]` into a destination run
    at the aligned index set (prefix ∪ suffix, generally non-contiguous).

    Args:
        source_z: hook_z from the counterfactual run, shape
            [1, L_cf_or_corrupt, n_heads, d_head].
        head_idx: which head to patch.
        corrupt_idx: positions in the destination run to overwrite (always
            < L_corrupt -- prompt positions only, answer positions untouched).
        source_idx: matching positions in `source_z` to read from.
        L_full: destination run's full sequence length
            (L_corrupt + L_ctx_ans or L_corrupt + L_mem_ans).
    """
    def patch_fn(activation: torch.Tensor, hook) -> torch.Tensor:
        assert activation.shape[1] == L_full, (
            f"Expected sequence length {L_full} but got {activation.shape[1]}. "
            "Ensure run_with_hooks is called with prompt_text + answer_text."
        )
        patched = activation.clone()
        patched[:, corrupt_idx, head_idx, :] = source_z[:, source_idx, head_idx, :]
        return patched
    return patch_fn


def _make_suffix_patch_fn(
    clean_z: torch.Tensor,
    head_idx: int,
    L_corrupt: int,
    L_min: int,
    L_ans: int = 0,
):
    """Suffix-align by raw length only. INTENTIONALLY kept for
    `_run_random_pair_baseline` ONLY: that baseline pairs UNRELATED prompts on
    purpose (a noise-floor control), so `alignment_indices` (which requires
    provably-identical token ids) would return a near-empty window and
    destroy the noise floor. Do NOT use this for the main patching loop or
    the identity check -- those use `_make_aligned_patch_fn` (see spec §A1.4).
    """
    L_full = L_corrupt + L_ans

    def patch_fn(activation: torch.Tensor, hook) -> torch.Tensor:
        assert activation.shape[1] == L_full, (
            f"Expected sequence length {L_full} (L_corrupt={L_corrupt} + "
            f"L_ans={L_ans}) but got {activation.shape[1]}."
        )
        patched = activation.clone()
        patched[:, L_corrupt - L_min:L_corrupt, head_idx, :] = (
            clean_z[:, -L_min:, head_idx, :]
        )
        return patched

    return patch_fn


# ---------------------------------------------------------------------------
# Noise-floor baselines (§8)
# ---------------------------------------------------------------------------

def _run_identity_patch_check(
    model,
    prompts: List[ConflictPrompt],
    n_layers: int,
    n_heads: int,
    head_subset: Optional[List[Tuple[int, int]]] = None,
) -> float:
    """Identity-patch: replace hook_z with itself — every score must be ~0.

    Patches the prompt portion of a (corrupt + ctx_answer) run with the SAME
    corrupt prompt's own hook_z, at the positions `alignment_indices` reports
    as identical to itself (i.e. all prompt positions, since a sequence
    trivially aligns with itself). Reading and writing the same physical
    positions of the same run is a true identity -> recovery must stay ~0.
    This is a wiring sanity check, not a scientific result.

    Returns the max |recovery_score| observed. Any value > 1e-4 indicates a
    hook-wiring bug.
    """
    heads_by_layer: Dict[int, List[int]] = _group_by_layer(head_subset, n_layers, n_heads)
    max_abs = 0.0
    for p in prompts:
        main_tokens = model.to_tokens(p.text)[0]
        L_corrupt = main_tokens.shape[0]
        corrupt_idx, _ = alignment_indices(main_tokens, main_tokens)
        ctx_ids   = _answer_ids(model, p.context_answer)
        mem_ids   = _answer_ids(model, p.memory_answer)
        L_ctx_ans = len(ctx_ids)
        ctx_text  = p.text + " " + p.context_answer.strip()
        mem_text  = p.text + " " + p.memory_answer.strip()

        with torch.no_grad():
            _, cache_corrupt = model.run_with_cache(p.text, names_filter=_HOOK_Z_FILTER)
            ctx_lp_c = answer_log_prob(model(ctx_text), ctx_ids)
            mem_lp_c = answer_log_prob(model(mem_text), mem_ids)
            ctx_margin_c = ctx_lp_c - mem_lp_c

        for l, heads in heads_by_layer.items():
            hook_name = _get_act_name("z", layer=l)
            for h in heads:
                patch_fn = _make_aligned_patch_fn(
                    _cache_get(cache_corrupt, hook_name), h,
                    corrupt_idx, corrupt_idx, L_corrupt + L_ctx_ans,
                )
                with torch.no_grad():
                    ctx_logits_p = model.run_with_hooks(ctx_text, fwd_hooks=[(hook_name, patch_fn)])
                    ctx_lp_p = answer_log_prob(ctx_logits_p, ctx_ids)
                    mem_lp_p = answer_log_prob(model(mem_text), mem_ids)
                ctx_margin_p = ctx_lp_p - mem_lp_p
                recovery = abs(ctx_margin_c - ctx_margin_p)
                if recovery > max_abs:
                    max_abs = recovery

    return max_abs


def _group_by_layer(
    head_subset: Optional[List[Tuple[int, int]]],
    n_layers: int,
    n_heads: int,
) -> Dict[int, List[int]]:
    """Map head subset to {layer: [head, ...]} for efficient per-layer iteration.

    If head_subset is None, returns all layers with all heads.
    """
    if head_subset is None:
        return {l: list(range(n_heads)) for l in range(n_layers)}
    by_layer: Dict[int, List[int]] = {}
    for l, h in head_subset:
        by_layer.setdefault(l, []).append(h)
    return by_layer


def _run_random_pair_baseline(
    model,
    prompts: List[ConflictPrompt],
    n_layers: int,
    n_heads: int,
    epsilon: float,
    l_max_diff: int,
    head_subset: Optional[List[Tuple[int, int]]] = None,
) -> float:
    """Random-pair baseline: mismatch counterfactual texts to establish a noise floor.

    For each prompt p[i], use build_counterfactual_text(p[i+1 mod n]) as the
    patch source. Log-probs for the 'clean' margin are computed using p's
    answers on the mismatched counterfactual text (intentionally wrong
    pairing -> noise floor estimate).

    INTENTIONALLY kept on naive suffix (`_make_suffix_patch_fn`, `L_min`)
    alignment rather than `alignment_indices` (spec §A1.4): p and other are
    UNRELATED prompts, so their prefixes/suffixes need not match and
    `alignment_indices` would return a near-empty window, destroying the
    noise floor.

    Returns the 95th percentile of all resulting normalised scores.
    """
    scores: List[float] = []
    n = len(prompts)
    if n < 2:
        return float("nan")

    for i, p in enumerate(prompts):
        other = prompts[(i + 1) % n]
        other_cf_text = build_counterfactual_text(other)
        if not swap_occurred(other, other_cf_text):
            # other's distractor wasn't found verbatim in its own text --
            # the "counterfactual" patch source would just be other.text
            # unchanged, corrupting this noise-floor sample. Skip it.
            continue

        L_corrupt  = _token_length(model, p.text)
        L_other_cf = _token_length(model, other_cf_text)

        if abs(L_corrupt - L_other_cf) > l_max_diff:
            continue

        L_min     = min(L_corrupt, L_other_cf)
        ctx_ids   = _answer_ids(model, p.context_answer)
        mem_ids   = _answer_ids(model, p.memory_answer)
        L_ctx_ans = len(ctx_ids)
        L_mem_ans = len(mem_ids)
        ctx_text  = p.text + " " + p.context_answer.strip()
        mem_text  = p.text + " " + p.memory_answer.strip()

        with torch.no_grad():
            _, cache_other_cf = model.run_with_cache(
                other_cf_text, names_filter=_HOOK_Z_FILTER
            )
            # Corrupt margins
            ctx_lp_c = answer_log_prob(model(ctx_text), ctx_ids)
            mem_lp_c = answer_log_prob(model(mem_text), mem_ids)
            # Mismatched-counterfactual margins: use p's answers on other's cf text
            ctx_lp_k = answer_log_prob(
                model(other_cf_text + " " + p.context_answer.strip()), ctx_ids
            )
            mem_lp_k = answer_log_prob(
                model(other_cf_text + " " + p.memory_answer.strip()), mem_ids
            )

        ctx_margin_c = ctx_lp_c - mem_lp_c
        ctx_margin_k = ctx_lp_k - mem_lp_k
        denominator  = ctx_margin_c - ctx_margin_k

        if abs(denominator) < epsilon or denominator <= 0:
            continue

        heads_by_layer: Dict[int, List[int]] = _group_by_layer(head_subset, n_layers, n_heads)
        for l, heads in heads_by_layer.items():
            hook_name = _get_act_name("z", layer=l)
            for h in heads:
                patch_fn_ctx = _make_suffix_patch_fn(
                    _cache_get(cache_other_cf, hook_name), h, L_corrupt, L_min, L_ctx_ans
                )
                patch_fn_mem = _make_suffix_patch_fn(
                    _cache_get(cache_other_cf, hook_name), h, L_corrupt, L_min, L_mem_ans
                )
                with torch.no_grad():
                    ctx_logits_p = model.run_with_hooks(ctx_text, fwd_hooks=[(hook_name, patch_fn_ctx)])
                    mem_logits_p = model.run_with_hooks(mem_text, fwd_hooks=[(hook_name, patch_fn_mem)])
                ctx_lp_p = answer_log_prob(ctx_logits_p, ctx_ids)
                mem_lp_p = answer_log_prob(mem_logits_p, mem_ids)
                ctx_margin_p = ctx_lp_p - mem_lp_p
                recovery = ctx_margin_c - ctx_margin_p
                scores.append(recovery / denominator)

    if not scores:
        return float("nan")
    return float(np.percentile(scores, 95))


# ---------------------------------------------------------------------------
# Crash-resilient checkpointing (§ long-run robustness)
# ---------------------------------------------------------------------------

def _save_pp_checkpoint(
    path: str,
    next_prompt_idx: int,
    head_scores_list: Dict[Tuple[int, int], List[float]],
    kept_prompt_indices: List[int],
    l_corrupts: List[int],
    l_cleans: List[int],
    counters: Dict[str, int],
    n_total: int,
) -> None:
    """Atomically write a resumable snapshot of PP progress.

    Written every `checkpoint_every` prompts so a HARD crash (CUDA driver
    death, OOM the per-prompt try/except can't catch, process kill, power
    loss) loses at most that many prompts of work instead of the whole cell.
    The snapshot is taken at the TOP of a loop iteration, so it reflects a
    consistent state of exactly prompts [0, next_prompt_idx).

    Atomic: writes to `path + ".tmp"` then os.replace() -- a crash mid-write
    can never leave a half-written (corrupt) checkpoint.
    """
    payload = {
        "next_prompt_idx": int(next_prompt_idx),
        "n_total": int(n_total),
        "head_keys": [f"{l}_{h}" for (l, h) in head_scores_list.keys()],
        "head_scores_list": {f"{l}_{h}": v for (l, h), v in head_scores_list.items()},
        "kept_prompt_indices": kept_prompt_indices,
        "L_corrupts": l_corrupts,
        "L_cleans": l_cleans,
        **counters,
    }
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


def _load_pp_checkpoint(
    path: str,
    n_total: int,
    heads_iter: List[Tuple[int, int]],
) -> Optional[Dict]:
    """Load a PP checkpoint iff it exists AND matches this exact cell.

    Guards against a stale/mismatched checkpoint silently corrupting a run:
    returns None (→ caller starts fresh, the safe default) unless the prompt
    count and the scored head-set both match the current run. A None return
    also covers a missing or unparseable file.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("n_total") != n_total:
        return None
    if set(data.get("head_keys", [])) != {f"{l}_{h}" for (l, h) in heads_iter}:
        return None
    head_scores_list: Dict[Tuple[int, int], List[float]] = {}
    for k, v in data["head_scores_list"].items():
        ls, hs = k.split("_")
        head_scores_list[(int(ls), int(hs))] = list(v)
    return {
        "next_prompt_idx": int(data["next_prompt_idx"]),
        "head_scores_list": head_scores_list,
        "kept_prompt_indices": list(data.get("kept_prompt_indices", [])),
        "L_corrupts": list(data.get("L_corrupts", [])),
        "L_cleans": list(data.get("L_cleans", [])),
        "n_skipped_length": int(data.get("n_skipped_length", 0)),
        "n_skipped_degenerate_margin": int(data.get("n_skipped_degenerate_margin", 0)),
        "n_skipped_no_swap": int(data.get("n_skipped_no_swap", 0)),
        "n_skipped_error": int(data.get("n_skipped_error", 0)),
    }


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def run_path_patching(
    model,
    prompts: List[ConflictPrompt],
    n_subset: Optional[int] = None,
    seed: int = 42,
    epsilon: float = 0.05,
    l_max_diff: int = 200,
    trim: float = 0.10,
    save_raw: bool = False,
    raw_path: Optional[str] = None,
    kept_indices: Optional[List[int]] = None,
    head_subset: Optional[List[Tuple[int, int]]] = None,
    checkpoint_path: Optional[str] = None,
    checkpoint_every: int = 50,
    _meta_out: Optional[Dict] = None,
) -> HeadScores:
    """Run activation patching at hook_z for all (layer, head) pairs.

    See PP_Spec.md §3 (alignment), §4 (degeneracy), §5 (algorithm), §7
    (LDS interface contract) for the design decisions behind the defaults.

    Returns HeadScores = Dict[(layer, head), float] where the score is the
    trimmed-mean normalized recovery of the (context - memory) margin.
    Higher = more causally responsible for context-following (matches LDS sign).

    Args:
        model: HookedTransformer (or duck-typed mock for testing).
        prompts: full list of ConflictPrompt objects.
        n_subset: if not None, sample this many prompts without replacement.
            Default None → run on all prompts (recommended for GPT-2).
        seed: RNG seed for subset selection and baseline sampling.
        epsilon: minimum |ctx_margin_corrupt - ctx_margin_clean| to keep a
            prompt (avoids near-zero denominator). Unit: logit units.
        l_max_diff: maximum allowed |L_corrupt - L_clean| before skipping.
        trim: fractional trim for the trimmed-mean aggregation (10% each tail).
        save_raw: if True, write per-prompt per-head raw scores to raw_path.
        raw_path: path for the raw JSON output (used only when save_raw=True).
        kept_indices: if not None, skip all filtering and use only these prompt
            indices (0-based into the prompts list). Intended for reusing the
            filtered set from a previous run (e.g. base GPT-2) so that Merge-A
            operates on the exact same prompt set across models.
        head_subset: if not None, score only these (layer, head) pairs instead
            of the full n_layers × n_heads grid. Noise-floor baselines also
            restrict to the subset so total cost scales with len(head_subset).
            None → current behaviour (all heads). Fully backward compatible.
        _meta_out: if not None, populated in-place with skip counts, noise
            floor, and other metadata for JSON serialisation by the caller.
    """
    rng = np.random.default_rng(seed)
    n_layers: int = model.cfg.n_layers
    n_heads: int = model.cfg.n_heads

    # --- Subset / kept-indices selection ------------------------------------
    subset_indices: Optional[List[int]] = None
    working: List[ConflictPrompt] = list(prompts)
    if kept_indices is not None:
        # Bypass all filtering — use exactly the provided indices.
        working = [prompts[i] for i in kept_indices]
        subset_indices = list(kept_indices)
        print(f"[pp] Using {len(working)} pre-filtered prompts from kept_indices (skipping filter step).")
    elif n_subset is not None:
        n_subset = min(n_subset, len(prompts))
        idx = rng.choice(len(prompts), n_subset, replace=False)
        subset_indices = [int(i) for i in idx]
        working = [prompts[i] for i in idx]

    n_total = len(working)

    # --- Noise-floor baselines (20-prompt sample, separate rng) --------------
    baseline_rng = np.random.default_rng(seed + 999)
    n_baseline = min(20, n_total)
    baseline_idx = baseline_rng.choice(n_total, n_baseline, replace=False)
    baseline_prompts = [working[i] for i in baseline_idx]

    try:
        identity_max_abs = _run_identity_patch_check(
            model, baseline_prompts, n_layers, n_heads, head_subset=head_subset
        )
    except Exception as e:  # noise-floor is diagnostic -- never let it kill the cell
        print(f"[pp-warn] identity-patch baseline failed ({type(e).__name__}: {e}); "
              "recording NaN and continuing.", file=sys.stderr)
        identity_max_abs = float("nan")
    # Threshold is dtype-aware: 1e-4 was calibrated against GPT-2 in float32
    # (observed ~1e-6 there). Comparing two separately-computed forward
    # passes (run_with_cache vs run_with_hooks, different sequence lengths)
    # is only bit-identical up to floating-point reproducibility -- bf16
    # GPU matmuls are not shape-invariant, so this genuinely runs ~1e4x
    # noisier at bf16 than fp32. Verified empirically on Gemma-3-4B
    # (2026-07-09): bf16 gave 3.1e-02, the SAME check in float32 on the
    # same prompts/heads collapsed to 2.98e-06 -- matching GPT-2's fp32
    # baseline almost exactly. So this is precision noise, not a
    # hook-wiring bug, for any bf16/fp16 run; a real wiring bug would show
    # a much larger (order-1) recovery regardless of dtype.
    try:
        model_dtype = next(model.parameters()).dtype
    except (AttributeError, StopIteration):
        model_dtype = torch.float32  # test doubles without real params: fp32-strict default
    identity_threshold = 1e-4 if model_dtype in (torch.float32, torch.float64) else 1e-1
    if identity_max_abs > identity_threshold:
        print(
            f"[pp-warn] identity-patch max |score| = {identity_max_abs:.2e} > "
            f"{identity_threshold:.0e} (dtype={model_dtype}). Possible hook-wiring bug "
            "— check _make_aligned_patch_fn."
        )

    try:
        random_pair_p95 = _run_random_pair_baseline(
            model, baseline_prompts, n_layers, n_heads, epsilon, l_max_diff,
            head_subset=head_subset,
        )
    except Exception as e:
        print(f"[pp-warn] random-pair baseline failed ({type(e).__name__}: {e}); "
              "recording NaN and continuing.", file=sys.stderr)
        random_pair_p95 = float("nan")

    noise_floor = {
        # NaN is invalid JSON -- store None so the output file stays parseable.
        "identity_patch_max_abs": float(identity_max_abs) if np.isfinite(identity_max_abs) else None,
        "random_pair_p95": float(random_pair_p95) if np.isfinite(random_pair_p95) else None,
    }

    # --- Per-head score accumulation -----------------------------------------
    heads_iter: List[Tuple[int, int]] = (
        head_subset if head_subset is not None
        else [(l, h) for l in range(n_layers) for h in range(n_heads)]
    )
    head_scores_list: Dict[Tuple[int, int], List[float]] = {k: [] for k in heads_iter}
    _heads_by_layer: Dict[int, List[int]] = _group_by_layer(head_subset, n_layers, n_heads)
    raw_records: Optional[List] = [] if save_raw else None

    n_skipped_length = 0
    n_skipped_degenerate_margin = 0
    n_skipped_no_swap = 0
    n_skipped_error = 0
    kept_prompt_indices: List[int] = []

    L_corrupts: List[int] = []
    L_cleans: List[int] = []

    # --- Resume from checkpoint (if one exists and matches this cell) --------
    resume_from = 0
    if checkpoint_path is not None:
        _ckpt = _load_pp_checkpoint(checkpoint_path, n_total, heads_iter)
        if _ckpt is not None:
            head_scores_list = _ckpt["head_scores_list"]
            kept_prompt_indices = _ckpt["kept_prompt_indices"]
            L_corrupts = _ckpt["L_corrupts"]
            L_cleans = _ckpt["L_cleans"]
            n_skipped_length = _ckpt["n_skipped_length"]
            n_skipped_degenerate_margin = _ckpt["n_skipped_degenerate_margin"]
            n_skipped_no_swap = _ckpt["n_skipped_no_swap"]
            n_skipped_error = _ckpt["n_skipped_error"]
            resume_from = _ckpt["next_prompt_idx"]
            print(f"[pp] resumed from checkpoint: {resume_from}/{n_total} prompts already "
                  f"done ({checkpoint_path}).")

    from tqdm import tqdm
    for prompt_idx, p in enumerate(tqdm(working, desc="patching prompts", unit="prompt", dynamic_ncols=True, ascii=True, file=sys.stdout)):
        if prompt_idx < resume_from:
            continue  # already scored in a prior (checkpointed) run

        # Snapshot progress at the TOP of the iteration (state = prompts
        # [0, prompt_idx)), so a hard crash during this prompt loses only this
        # one prompt, not the whole cell.
        if (checkpoint_path is not None and prompt_idx > resume_from
                and prompt_idx % checkpoint_every == 0):
            _save_pp_checkpoint(
                checkpoint_path, prompt_idx, head_scores_list, kept_prompt_indices,
                L_corrupts, L_cleans,
                {"n_skipped_length": n_skipped_length,
                 "n_skipped_degenerate_margin": n_skipped_degenerate_margin,
                 "n_skipped_no_swap": n_skipped_no_swap,
                 "n_skipped_error": n_skipped_error},
                n_total,
            )

        try:
            main_tokens, cf_tokens = build_contrast_pair(model, p)
            L_corrupt = main_tokens.shape[0]
            cf_text = build_counterfactual_text(p)
            L_cf = cf_tokens.shape[0]
            L_corrupts.append(L_corrupt)
            L_cleans.append(L_cf)  # legacy list name; now holds counterfactual lengths

            if not swap_occurred(p, cf_text):
                # context_answer wasn't found verbatim in p.text -- the in-place
                # swap was a silent no-op (cf_text == p.text). Scoring this row
                # would produce a degenerate zero-effect result indistinguishable
                # from a real null finding, so skip and count it instead.
                n_skipped_no_swap += 1
                continue

            if kept_indices is None:
                if abs(L_corrupt - L_cf) > l_max_diff:
                    n_skipped_length += 1
                    continue

            corrupt_idx, cf_idx = alignment_indices(main_tokens, cf_tokens)
            ctx_ids   = _answer_ids(model, p.context_answer)
            mem_ids   = _answer_ids(model, p.memory_answer)
            L_ctx_ans = len(ctx_ids)
            L_mem_ans = len(mem_ids)
            ctx_text  = p.text + " " + p.context_answer.strip()
            mem_text  = p.text + " " + p.memory_answer.strip()

            # Counterfactual cache for patching + log-prob margins.
            with torch.no_grad():
                _, cache_cf = model.run_with_cache(cf_text, names_filter=_HOOK_Z_FILTER)
                ctx_lp_c = answer_log_prob(model(ctx_text), ctx_ids)
                mem_lp_c = answer_log_prob(model(mem_text), mem_ids)
                ctx_lp_k = answer_log_prob(
                    model(cf_text + " " + p.context_answer.strip()), ctx_ids
                )
                mem_lp_k = answer_log_prob(
                    model(cf_text + " " + p.memory_answer.strip()), mem_ids
                )

            ctx_margin_c = ctx_lp_c - mem_lp_c
            ctx_margin_k = ctx_lp_k - mem_lp_k
            denominator  = ctx_margin_c - ctx_margin_k

            if abs(denominator) < epsilon or denominator <= 0:
                n_skipped_degenerate_margin += 1
                continue

            # Per-head patching loop (restricted to head_subset if given). Patch
            # source is the counterfactual cache -- NEVER clean_text. Buffer this
            # prompt's per-head results and commit them ONLY after every head
            # succeeds, so an error mid-loop can't leave some heads with an extra
            # score and others without (which would skew the trimmed mean).
            prompt_scores: Dict[Tuple[int, int], float] = {}
            prompt_raw: List = []
            for l, heads in _heads_by_layer.items():
                hook_name = _get_act_name("z", layer=l)
                for h in heads:
                    patch_fn_ctx = _make_aligned_patch_fn(
                        _cache_get(cache_cf, hook_name), h,
                        corrupt_idx, cf_idx, L_corrupt + L_ctx_ans,
                    )
                    patch_fn_mem = _make_aligned_patch_fn(
                        _cache_get(cache_cf, hook_name), h,
                        corrupt_idx, cf_idx, L_corrupt + L_mem_ans,
                    )
                    with torch.no_grad():
                        ctx_logits_p = model.run_with_hooks(ctx_text, fwd_hooks=[(hook_name, patch_fn_ctx)])
                        mem_logits_p = model.run_with_hooks(mem_text, fwd_hooks=[(hook_name, patch_fn_mem)])
                    ctx_lp_p     = answer_log_prob(ctx_logits_p, ctx_ids)
                    mem_lp_p     = answer_log_prob(mem_logits_p, mem_ids)
                    ctx_margin_p = ctx_lp_p - mem_lp_p
                    recovery     = ctx_margin_c - ctx_margin_p
                    norm_score   = recovery / denominator
                    prompt_scores[(l, h)] = norm_score

                    if raw_records is not None:
                        prompt_raw.append({
                            "layer": l, "head": h,
                            "L_corrupt": L_corrupt, "L_counterfactual": L_cf,
                            "ctx_margin_corrupt": ctx_margin_c,
                            "ctx_margin_counterfactual": ctx_margin_k,
                            "recovery_score":     float(recovery),
                            "normalized_score":   float(norm_score),
                        })

            # Atomic per-prompt commit (all heads succeeded).
            for k, v in prompt_scores.items():
                head_scores_list[k].append(v)
            if raw_records is not None:
                raw_records.extend(prompt_raw)
            kept_prompt_indices.append(prompt_idx)

        except Exception as e:
            # One bad prompt (tokenizer edge case, degenerate answer, transient
            # CUDA error over a multi-hour run) must not kill the whole cell.
            # Log it, count it, and move on -- the atomic commit above means no
            # partial scores leaked into head_scores_list for this prompt.
            n_skipped_error += 1
            print(f"[pp-err] prompt {prompt_idx} failed ({type(e).__name__}: {e}) "
                  "-- skipping.", file=sys.stderr)
            continue

    # --- Skip-rate warning ---------------------------------------------------
    n_skipped = (n_skipped_length + n_skipped_degenerate_margin
                 + n_skipped_no_swap + n_skipped_error)
    n_used = n_total - n_skipped
    if n_total > 0 and n_skipped / n_total > 0.15:
        print(
            f"[pp-warn] {n_skipped}/{n_total} ({100*n_skipped/n_total:.1f}%) "
            "prompts skipped — run may be suspect."
        )
    if n_skipped_error > 0:
        print(
            f"[pp-warn] {n_skipped_error} prompt(s) skipped due to ERRORS "
            "(see [pp-err] lines above) — investigate if this is more than a few."
        )
    print(
        f"[pp] used {n_used}/{n_total} prompts "
        f"(skipped: length={n_skipped_length}, margin={n_skipped_degenerate_margin}, "
        f"no_swap={n_skipped_no_swap}, error={n_skipped_error})"
    )

    # --- Aggregation (trimmed mean + secondary statistics) -------------------
    head_scores: HeadScores = {}
    secondary_plain: Dict[Tuple[int, int], float] = {}
    secondary_median: Dict[Tuple[int, int], float] = {}

    for (l, h), scores in head_scores_list.items():
        head_scores[(l, h)] = _trimmed_mean(scores, trim)
        secondary_plain[(l, h)] = float(np.mean(scores)) if scores else float("nan")
        secondary_median[(l, h)] = float(np.median(scores)) if scores else float("nan")

    # --- Populate _meta_out --------------------------------------------------
    if _meta_out is not None:
        _meta_out["n_prompts_input"] = n_total
        _meta_out["n_prompts_used"] = n_used
        _meta_out["subset_seed"] = seed if n_subset is not None else None
        _meta_out["subset_indices"] = subset_indices
        _meta_out["aggregation"]    = "trimmed_mean_normalized_recovery_p10"
        _meta_out["scoring_target"] = "mean_log_prob_diff"
        _meta_out["margin_direction"] = "context_minus_memory"
        _meta_out["epsilon_log_prob_units"] = epsilon
        _meta_out["L_max_diff"] = l_max_diff
        _meta_out["noise_floor"] = noise_floor
        _meta_out["n_skipped_length_mismatch"]  = n_skipped_length
        _meta_out["n_skipped_degenerate_margin"] = n_skipped_degenerate_margin
        _meta_out["n_skipped_no_swap"] = n_skipped_no_swap
        _meta_out["n_skipped_error"] = n_skipped_error
        _meta_out["kept_prompt_indices"] = kept_prompt_indices
        _meta_out["head_subset"] = (
            [f"{l}_{h}" for l, h in head_subset] if head_subset is not None else None
        )
        _meta_out["n_heads_scored"] = len(heads_iter)
        if L_corrupts:
            _meta_out["L_corrupt_distribution"] = {
                "min": int(min(L_corrupts)),
                "median": int(np.median(L_corrupts)),
                "max": int(max(L_corrupts)),
            }
            # NOTE: key name "L_clean_distribution" is legacy (A1.6) -- it now
            # holds COUNTERFACTUAL-text length stats, not clean_text. Kept
            # unchanged to avoid breaking run_pp_topheads.py's JSON consumers.
            _meta_out["L_clean_distribution"] = {
                "min": int(min(L_cleans)),
                "median": int(np.median(L_cleans)),
                "max": int(max(L_cleans)),
            }
        _meta_out["secondary_plain_mean"] = {
            f"{l}_{h}": v for (l, h), v in secondary_plain.items()
        }
        _meta_out["secondary_median"] = {
            f"{l}_{h}": v for (l, h), v in secondary_median.items()
        }

    # --- Optional raw output -------------------------------------------------
    if save_raw and raw_records is not None and raw_path is not None:
        with open(raw_path, "w") as fout:
            json.dump(raw_records, fout)

    # Cell completed successfully -- the real output file is about to be
    # written by the caller, so the resume checkpoint is no longer needed.
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
        except OSError:
            pass

    return head_scores


# ---------------------------------------------------------------------------
# Run script — called directly to produce results/pp_*.json
# ---------------------------------------------------------------------------

def _build_result_json(
    head_scores: HeadScores,
    meta: Dict,
    model_tag: str,
    prompt_type: str,
    set_tag: str = "full",
    alias_policy: str = "first_single_token",
) -> Dict:
    """Construct the §10 output JSON structure."""
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = "unknown"

    try:
        import transformer_lens
        tl_version = transformer_lens.__version__
    except Exception:
        tl_version = "unknown"

    return {
        "_provenance": build_provenance(
            contrast="inplace_swap",
            dataset=f"single_token_{set_tag}" if set_tag != "full" else "full_multitoken",
            n_used=meta.get("n_prompts_used", 0),
            alias_policy=alias_policy,
            prompt_type=prompt_type,
            model_tag=model_tag,
        ),
        "model": model_tag,
        "prompt_type": prompt_type,
        "n_prompts_input": meta.get("n_prompts_input", 0),
        "n_prompts_used": meta.get("n_prompts_used", 0),
        "subset_seed": meta.get("subset_seed"),
        "subset_indices": meta.get("subset_indices"),
        "aggregation": meta.get("aggregation", "trimmed_mean_normalized_recovery_p10"),
        "margin_direction": meta.get("margin_direction", "context_minus_memory"),
        "epsilon_logit_units": meta.get("epsilon_logit_units", 0.5),
        "L_max_diff": meta.get("L_max_diff", 4),
        "head_scores": {f"{l}_{h}": v for (l, h), v in head_scores.items()},
        "noise_floor": meta.get("noise_floor", {}),
        "meta": {
            "n_layers": 12,
            "n_heads": 12,
            "n_skipped_degenerate_margin": meta.get("n_skipped_degenerate_margin", 0),
            "n_skipped_length_mismatch": meta.get("n_skipped_length_mismatch", 0),
            "n_skipped_clean_neither": meta.get("n_skipped_clean_neither", 0),
            "L_corrupt_distribution": meta.get("L_corrupt_distribution", {}),
            "L_clean_distribution": meta.get("L_clean_distribution", {}),
            "tokenizer": "gpt2",
            "torch_version": torch.__version__,
            "transformer_lens_version": tl_version,
            "git_hash": git_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }


if __name__ == "__main__":
    import argparse

    sys.path.insert(0, _ROOT)
    from foundation import load_model, load_conflict_prompts

    parser = argparse.ArgumentParser(description="Run path patching and save results.")
    parser.add_argument(
        "--model", default="gpt2", choices=["gpt2", "gpt2_instruct"],
        help="Which model to run ('gpt2' or 'gpt2_instruct')."
    )
    parser.add_argument(
        "--prompt_type", default="substitution",
        choices=["substitution", "coherent"],
    )
    parser.add_argument("--n_subset", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_raw", action="store_true")
    parser.add_argument(
        "--kept_indices_file", type=str, default=None,
        help="Path to a JSON file containing kept prompt indices from a prior run "
             "(e.g. results/pp_gpt2_substitution_kept_indices.json). "
             "If provided, skips the filter step and uses exactly those prompts.",
    )
    parser.add_argument(
        "--save_kept_indices", action="store_true",
        help="Save the kept prompt indices from this run to a sidecar JSON file "
             "so a subsequent run (e.g. instruct model) can reuse them.",
    )
    args = parser.parse_args()

    _MODEL_HF = {
        "gpt2": (None, "gpt2"),
        "gpt2_instruct": ("vicgalle/gpt2-open-instruct-v1", "gpt2_instruct"),
    }
    hf_name, model_tag = _MODEL_HF[args.model]
    kwargs = {"hf_model_name": hf_name} if hf_name else {}

    print(f"[pp] Loading model={args.model}, prompt_type={args.prompt_type} ...")
    m = load_model("gpt2", **kwargs)
    prompts = load_conflict_prompts(prompt_type=args.prompt_type)

    results_dir = os.path.join(_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    out_path = os.path.join(results_dir, f"pp_{model_tag}_{args.prompt_type}.json")
    raw_path = os.path.join(results_dir, f"pp_{model_tag}_{args.prompt_type}_raw.json")
    kept_indices_path = os.path.join(results_dir, f"pp_gpt2_{args.prompt_type}_kept_indices.json")

    # Load pre-filtered indices if requested.
    kept_indices: Optional[List[int]] = None
    if args.kept_indices_file:
        with open(args.kept_indices_file) as f:
            kept_indices = json.load(f)
        print(f"[pp] Loaded {len(kept_indices)} kept indices from {args.kept_indices_file}")

    meta: Dict = {}
    scores = run_path_patching(
        m,
        prompts,
        n_subset=args.n_subset,
        seed=args.seed,
        save_raw=args.save_raw,
        raw_path=raw_path if args.save_raw else None,
        kept_indices=kept_indices,
        _meta_out=meta,
    )

    result = _build_result_json(scores, meta, model_tag, args.prompt_type)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[pp] Saved -> {out_path}")

    if args.save_kept_indices:
        with open(kept_indices_path, "w") as f:
            json.dump(meta.get("kept_prompt_indices", []), f)
        print(f"[pp] Kept indices saved → {kept_indices_path}")
