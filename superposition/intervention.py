"""
superposition/intervention.py
-------------------------------
Logit intervention: amplify or suppress specific attention heads
via TransformerLens fwd_hooks, then measure CRR change.

Four intervention conditions per family:
    amplify_context  — scale top-k context heads by 2.0
    suppress_context — zero-ablate top-k context heads (scale 0.0)
    amplify_memory   — scale top-k memory heads by 2.0
    suppress_memory  — zero-ablate top-k memory heads (scale 0.0)

Expected outcomes if superposition analysis is correct:
    amplify_context  → ΔCRR positive  (more context following)
    suppress_context → ΔCRR negative  (less context following)
    amplify_memory   → ΔCRR negative  (more memory following)
    suppress_memory  → ΔCRR positive  (less memory following)

Deviation from these expectations → attribution may be noisy.
"""

from __future__ import annotations
import logging
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from load_dataset import ConflictRow

log = logging.getLogger(__name__)


# ── CRR measurement ────────────────────────────────────────────────────────────

def compute_crr(
    model:   HookedTransformer,
    prompts: list[ConflictRow],
    hooks:   list[tuple] | None = None,
    label:   str = "",
) -> dict:
    """
    Compute Contextual Reliance Rate with optional intervention hooks.

    CRR = fraction of prompts where model's top-1 prediction is the
          context (distractor) token rather than the memory token.

    Parameters
    ----------
    model   : HookedTransformer in eval mode.
    prompts : List of ConflictRow.
    hooks   : List of (hook_name, hook_fn) tuples for intervention.
              Pass None for baseline (no intervention).
    label   : Display label for tqdm.

    Returns
    -------
    dict with keys:
        crr            — float [0, 1]
        n_context      — count of context-following predictions
        n_memory       — count of memory-following predictions
        n_neither      — count of neither
        n_total        — total prompts evaluated
    """
    n_context = 0
    n_memory  = 0
    n_neither = 0

    for row in tqdm(prompts, desc=f"  CRR {label}", unit="prompt"):
        tokens = model.to_tokens(row.prompt_conflict)
        if tokens.shape[1] > model.cfg.n_ctx:
            tokens = tokens[:, -model.cfg.n_ctx:]

        try:
            with torch.no_grad():
                if hooks:
                    logits = model.run_with_hooks(
                        tokens,
                        fwd_hooks=hooks,
                        return_type="logits",
                    )
                else:
                    logits = model(tokens, return_type="logits")

            # Top-1 prediction at final position
            pred = logits[0, -1, :].argmax().item()

            if pred == row.context_token_id:
                n_context += 1
            elif pred == row.memory_token_id:
                n_memory += 1
            else:
                n_neither += 1

        except Exception as e:
            log.warning(f"CRR forward pass failed (row {row.row_idx}): {e}")
            n_neither += 1

    n_total = n_context + n_memory + n_neither
    crr     = n_context / n_total if n_total > 0 else 0.0

    return {
        "crr":       crr,
        "n_context": n_context,
        "n_memory":  n_memory,
        "n_neither": n_neither,
        "n_total":   n_total,
    }


# ── Hook factory ───────────────────────────────────────────────────────────────

def make_scale_hooks(
    target_heads: list[tuple[int, int]],
    scale_factor: float,
) -> list[tuple]:
    """
    Create TransformerLens fwd_hooks that scale target heads.

    scale_factor = 0.0  →  zero-ablation (suppress)
    scale_factor = 2.0  →  amplification (double output)

    Parameters
    ----------
    target_heads : List of (layer, head) tuples to intervene on.
    scale_factor : Multiplier applied to hook_z at the target heads.

    Returns
    -------
    List of (hook_name, hook_fn) tuples ready for model.run_with_hooks().
    """
    hooks = []
    for (layer, head) in target_heads:
        hook_name = f"blocks.{layer}.attn.hook_z"

        # Capture layer and head in closure
        def make_fn(l=layer, h=head, s=scale_factor):
            def hook_fn(value, hook):
                # value: [batch, seq, n_heads, d_head]
                value[:, :, h, :] = value[:, :, h, :] * s
                return value
            return hook_fn

        hooks.append((hook_name, make_fn()))

    return hooks


# ── Top-k head selection from superposition results ───────────────────────────

def select_top_context_heads(
    comparison: dict,
    k:          int,
) -> list[tuple[int, int]]:
    """
    Select top-k heads with highest base_ratio (most context-biased in BASE model).
    These are the heads we expect to steer context-following when amplified.
    """
    heads = [
        ((v["layer"], v["head"]), v["base_ratio"])
        for v in comparison.values()
    ]
    heads.sort(key=lambda x: x[1], reverse=True)
    return [h for h, _ in heads[:k]]


def select_top_memory_heads(
    comparison: dict,
    k:          int,
) -> list[tuple[int, int]]:
    """
    Select top-k heads with lowest base_ratio (most memory-biased in BASE model).
    """
    heads = [
        ((v["layer"], v["head"]), v["base_ratio"])
        for v in comparison.values()
    ]
    heads.sort(key=lambda x: x[1])
    return [h for h, _ in heads[:k]]


# ── Full intervention experiment ───────────────────────────────────────────────

def run_intervention_experiment(
    model:      HookedTransformer,
    prompts:    list[ConflictRow],
    comparison: dict,
    topk:       int = 5,
    label:      str = "",
) -> dict:
    """
    Run all four intervention conditions and return results dict.

    Parameters
    ----------
    model      : HookedTransformer (base model).
    prompts    : ConflictRow list.
    comparison : Output of superposition.compare_models() — used to select heads.
    topk       : Number of heads per condition.
    label      : Family label for logging.

    Returns
    -------
    dict with baseline CRR and all four intervention conditions.
    """
    # Select target heads
    ctx_heads = select_top_context_heads(comparison, topk)
    mem_heads = select_top_memory_heads(comparison,  topk)

    print(f"\n[intervention/{label}] Top-{topk} context heads: {ctx_heads}")
    print(f"[intervention/{label}] Top-{topk} memory heads:  {mem_heads}")

    # Baseline (no intervention)
    print(f"\n[intervention/{label}] Running baseline CRR...")
    baseline = compute_crr(model, prompts, hooks=None, label=f"{label}/baseline")
    print(f"  Baseline CRR: {baseline['crr']:.4f}")

    results = {
        "family":       label,
        "topk":         topk,
        "baseline":     baseline,
        "ctx_heads":    ctx_heads,
        "mem_heads":    mem_heads,
        "conditions":   {},
    }

    # Four intervention conditions
    conditions = [
        ("amplify_context",  ctx_heads, 2.0),
        ("suppress_context", ctx_heads, 0.0),
        ("amplify_memory",   mem_heads, 2.0),
        ("suppress_memory",  mem_heads, 0.0),
    ]

    for cond_name, heads, scale in conditions:
        print(f"\n[intervention/{label}] Condition: {cond_name} "
              f"(scale={scale}, {len(heads)} heads)...")

        hooks    = make_scale_hooks(heads, scale)
        crr_res  = compute_crr(model, prompts, hooks=hooks, label=cond_name)
        delta    = crr_res["crr"] - baseline["crr"]

        print(f"  CRR: {crr_res['crr']:.4f}  |  ΔCRR: {delta:+.4f}")

        results["conditions"][cond_name] = {
            "scale":     scale,
            "heads":     heads,
            "crr":       crr_res["crr"],
            "delta_crr": delta,
            "n_context": crr_res["n_context"],
            "n_memory":  crr_res["n_memory"],
            "n_neither": crr_res["n_neither"],
        }

    return results
