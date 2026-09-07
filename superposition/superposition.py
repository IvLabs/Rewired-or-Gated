"""
superposition/superposition.py
--------------------------------
Core superposition computation.

ONE forward pass per prompt. Caches hook_z at every layer and
computes each head's logit attribution toward memory and context tokens.

This is NOT a dual-run. JuICE's dual-run is for their intervention
method (saving Run-1 activations to steer Run-2). We only MEASURE,
so one pass is sufficient.

Key output per head:
    mem_pull  — head's average contribution to memory-answer logit
    ctx_pull  — head's average contribution to context-answer logit
    ratio     — ctx_pull / mem_pull
    role      — "context" | "superposition" | "memory"

Thresholds:
    ratio > 2.0  → context head
    ratio < 0.5  → memory head
    otherwise    → superposition
"""

from __future__ import annotations
import logging
import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from load_dataset import ConflictRow
from logit_utils import get_ln_scale

log = logging.getLogger(__name__)

CONTEXT_THRESH = 2.0
MEMORY_THRESH  = 0.5


def _role(ratio: float) -> str:
    if ratio > CONTEXT_THRESH:
        return "context"
    if ratio < MEMORY_THRESH:
        return "memory"
    return "superposition"


def compute_superposition(
    model:     HookedTransformer,
    prompts:   list[ConflictRow],
    norm_type: str,
    device:    str,
    label:     str = "",
) -> dict:
    """
    Run superposition analysis over all prompts for one model.

    Parameters
    ----------
    model     : HookedTransformer, eval mode, on device.
    prompts   : List of ConflictRow (substitution, filtered).
    norm_type : "rms" (Qwen/Llama/Gemma) or "ln" (GPT-2).
    device    : "cuda" or "cpu".
    label     : Label for tqdm progress bar.

    Returns
    -------
    dict with keys:
        n_prompts, n_layers, n_heads, thresholds, heads
    """
    n_layers = model.cfg.n_layers
    n_heads  = model.cfg.n_heads

    # Accumulators: sum over prompts, divide at end
    mem_acc = np.zeros((n_layers, n_heads), dtype=np.float64)
    ctx_acc = np.zeros((n_layers, n_heads), dtype=np.float64)
    n_valid = 0

    # Only cache what we need
    needed = {f"blocks.{l}.attn.hook_z" for l in range(n_layers)}
    needed.add(f"blocks.{n_layers - 1}.hook_resid_post")
    names_filter = lambda name: name in needed

    W_O = model.W_O  # [n_layers, n_heads, d_head, d_model]
    W_U = model.W_U  # [d_model, vocab_size]

    for row in tqdm(prompts, desc=f"  {label}", unit="prompt"):
        tokens = model.to_tokens(row.prompt_conflict)

        # Truncate to model context window
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
            log.warning(f"Forward pass failed (row {row.row_idx}): {e}")
            continue

        # Normalization scale — computed from final residual stream
        ln_scale = get_ln_scale(model, cache, norm_type)  # [1, 1]

        mem_tok = row.memory_token_id
        ctx_tok = row.context_token_id

        for l in range(n_layers):
            # hook_z: [1, seq, n_heads, d_head] — take final position
            z = cache[f"blocks.{l}.attn.hook_z"][:, -1, :, :]  # [1, n_heads, d_head]

            # Per-head residual: z_h @ W_O_h → [1, n_heads, d_model]
            # einsum works for both MHA and GQA (TL expands GQA in hook_z)
            head_out = torch.einsum("bnh,nhd->bnd", z, W_O[l])

            # Apply norm scale
            head_scaled = head_out * ln_scale.unsqueeze(1)  # [1, n_heads, d_model]

            # Project to two specific tokens (avoid materializing full vocab)
            logit_mem = (head_scaled @ W_U[:, mem_tok]).squeeze(0)  # [n_heads]
            logit_ctx = (head_scaled @ W_U[:, ctx_tok]).squeeze(0)  # [n_heads]

            mem_acc[l] += logit_mem.detach().cpu().float().numpy()
            ctx_acc[l] += logit_ctx.detach().cpu().float().numpy()

        del cache
        n_valid += 1

    if n_valid == 0:
        raise RuntimeError("No prompts were successfully processed.")

    # Average over prompts
    mem_acc /= n_valid
    ctx_acc /= n_valid

    # Ratio with guard for near-zero denominator
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            np.abs(mem_acc) > 1e-8,
            ctx_acc / mem_acc,
            np.where(ctx_acc > 0, 1e6, -1e6),
        )

    # Build output dict
    heads = {}
    for l in range(n_layers):
        for h in range(n_heads):
            r = float(ratio[l, h])
            heads[f"L{l}H{h}"] = {
                "layer":    l,
                "head":     h,
                "mem_pull": float(mem_acc[l, h]),
                "ctx_pull": float(ctx_acc[l, h]),
                "ratio":    r,
                "role":     _role(r),
            }

    return {
        "n_prompts":  n_valid,
        "n_layers":   n_layers,
        "n_heads":    n_heads,
        "thresholds": {"context": CONTEXT_THRESH, "memory": MEMORY_THRESH},
        "heads":      heads,
    }


def compare_models(base_res: dict, inst_res: dict) -> dict:
    """
    Compute per-head delta ratios and role shifts between base and instruct.

    For each head in both results:
        ratio_shift = inst_ratio - base_ratio
        role_changed = True if role classification changed
        shift_label  = e.g. "memory→context", "stable"
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
            "base_mem_pull":  bh["mem_pull"],
            "base_ctx_pull":  bh["ctx_pull"],
            "base_ratio":     bh["ratio"],
            "base_role":      base_role,
            "inst_mem_pull":  ih["mem_pull"],
            "inst_ctx_pull":  ih["ctx_pull"],
            "inst_ratio":     ih["ratio"],
            "inst_role":      inst_role,
            "ratio_shift":    ih["ratio"] - bh["ratio"],
            "role_changed":   base_role != inst_role,
            "shift_label":    shift,
        }

    return comparison
