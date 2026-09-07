"""
superposition/logit_utils.py
-----------------------------
Core primitives used by superposition.py and intervention.py.

get_ln_scale()          — compute normalization scale from final residual
get_head_contribution() — scalar logit contribution of one head to one token
"""

from __future__ import annotations
import torch
from transformer_lens import HookedTransformer, ActivationCache


def get_ln_scale(
    model: HookedTransformer,
    cache: ActivationCache,
    norm_type: str,
) -> torch.Tensor:
    """
    Compute scalar normalization scale from the final residual stream.
    Used as a linear approximation of the final LayerNorm/RMSNorm.

    For RMSNorm (Qwen, Llama, Gemma):
        scale = 1 / sqrt(mean(x^2) + eps)
        NO mean subtraction.

    For LayerNorm (GPT-2):
        scale = 1 / sqrt(var(x) + eps)
        WITH mean subtraction before variance.

    Parameters
    ----------
    model     : HookedTransformer
    cache     : ActivationCache from model.run_with_cache()
    norm_type : "rms" or "ln"

    Returns
    -------
    torch.Tensor shape [1, 1] — broadcast-compatible scalar scale.
    """
    n_layers = model.cfg.n_layers
    resid    = cache[f"blocks.{n_layers - 1}.hook_resid_post"][:, -1, :]

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


def get_head_contribution(
    model:    HookedTransformer,
    cache:    ActivationCache,
    layer:    int,
    head:     int,
    token_id: int,
    ln_scale: torch.Tensor,
) -> float:
    """
    Scalar logit contribution of head (layer, head) to token_id.

    Formula:
        z_h    = hook_z[0, -1, head, :]          [d_head]
        out_h  = z_h @ W_O[layer, head]           [d_model]
        scaled = out_h * ln_scale                 [d_model]
        logit  = scaled @ W_U[:, token_id]        scalar

    Parameters
    ----------
    model    : HookedTransformer
    cache    : ActivationCache (must contain hook_z for this layer)
    layer    : Layer index
    head     : Head index
    token_id : Vocab index of the answer token
    ln_scale : Precomputed normalization scale [1,1] from get_ln_scale()

    Returns
    -------
    float
    """
    hook_name = f"blocks.{layer}.attn.hook_z"
    z         = cache[hook_name][0, -1, head, :]        # [d_head]
    W_O_head  = model.W_O[layer, head]                  # [d_head, d_model]
    head_out  = z @ W_O_head                            # [d_model]
    scaled    = head_out * ln_scale.squeeze()           # [d_model]
    logit     = (scaled @ model.W_U[:, token_id])       # scalar

    return logit.item()
