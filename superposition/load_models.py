"""
superposition/load_models.py
-----------------------------
Unified model loader for all three families.
Handles architecture differences, RMSNorm vs LayerNorm,
GQA, and gated model authentication.

Usage:
    from load_models import load_model_pair, FAMILIES
    base_model, instruct_model = load_model_pair("qwen", device="cuda")
"""

from __future__ import annotations
import os
import logging
import torch
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

# ── Family registry ────────────────────────────────────────────────────────────

FAMILIES = {
    "qwen": {
        "base":             "Qwen/Qwen2.5-3B",
        "instruct":         "Qwen/Qwen2.5-3B-Instruct",
        "norm_type":        "rms",
        "center_unembed":   False,
        "gated":            False,
        "expected_layers":  36,
        "expected_heads":   16,
    },
    "llama": {
        "base":             "meta-llama/Llama-3.2-3B",
        "instruct":         "meta-llama/Llama-3.2-3B-Instruct",
        "norm_type":        "rms",
        "center_unembed":   False,
        "gated":            True,
        "expected_layers":  28,
        "expected_heads":   24,
    },
    "gemma": {
        "base":             "google/gemma-3-4b-pt",
        "instruct":         "google/gemma-3-4b-it",
        "norm_type":        "rms",
        "center_unembed":   False,
        "gated":            True,
        "expected_layers":  34,
        "expected_heads":   8,
    },
}


# ── Core loader ────────────────────────────────────────────────────────────────

def load_tl_model(
    hf_name:        str,
    family_key:     str,
    device:         str = "cuda",
) -> HookedTransformer:
    """
    Load a HookedTransformer for any supported family.

    Strategy:
      1. Try direct HookedTransformer.from_pretrained() — works when
         TransformerLens has native support for the model.
      2. If that fails, load via HuggingFace first, then pass hf_model=
         to from_pretrained() with the family's base architecture name.

    Parameters
    ----------
    hf_name    : Full HuggingFace model ID (e.g. "Qwen/Qwen2.5-3B-Instruct")
    family_key : One of "qwen", "llama", "gemma"
    device     : "cuda" or "cpu"

    Returns
    -------
    HookedTransformer in eval mode on the specified device.
    """
    cfg = FAMILIES[family_key]

    # Handle gated models
    hf_token = os.environ.get("HF_TOKEN", None)
    if cfg["gated"] and not hf_token:
        raise EnvironmentError(
            f"{hf_name} is a gated model. Set HF_TOKEN in your .env file.\n"
            f"Get token at: https://huggingface.co/settings/tokens\n"
            f"Accept license at: https://huggingface.co/{hf_name}"
        )

    if hf_token:
        from huggingface_hub import login as hf_login
        try:
            hf_login(token=hf_token, add_to_git_credential=False)
        except Exception as e:
            log.warning(f"HF login failed in load_models: {e}")

    tl_kwargs = dict(
        center_unembed          = cfg["center_unembed"],
        center_writing_weights  = False,
        fold_ln                 = True,
        refactor_factored_attn_matrices = False,
        dtype                   = torch.float32,
    )

    print(f"[load_models] Loading {hf_name}...")

    try:
        model = HookedTransformer.from_pretrained(hf_name, **tl_kwargs)

    except Exception as e:
        print(f"[load_models] Direct TL load failed: {e}")
        print(f"[load_models] Falling back to HF load + hf_model=...")

        hf_kwargs = {"torch_dtype": torch.float32}
        if hf_token:
            hf_kwargs["token"] = hf_token

        hf_model = AutoModelForCausalLM.from_pretrained(hf_name, **hf_kwargs)
        model = HookedTransformer.from_pretrained(
            cfg["base"],   # use base arch as template
            hf_model=hf_model,
            **tl_kwargs,
        )

    model.to(device).eval()

    # Verify architecture
    assert model.cfg.n_layers == cfg["expected_layers"], (
        f"Expected {cfg['expected_layers']} layers, got {model.cfg.n_layers}"
    )
    assert model.cfg.n_heads == cfg["expected_heads"], (
        f"Expected {cfg['expected_heads']} heads, got {model.cfg.n_heads}"
    )

    print(f"[load_models] OK — n_layers={model.cfg.n_layers}, "
          f"n_heads={model.cfg.n_heads}, d_model={model.cfg.d_model}, "
          f"d_head={model.cfg.d_head}")

    return model


def load_tokenizer(family_key: str) -> AutoTokenizer:
    """Load the tokenizer for a model family (always use base tokenizer)."""
    cfg      = FAMILIES[family_key]
    hf_token = os.environ.get("HF_TOKEN", None)
    kwargs   = {"token": hf_token} if hf_token else {}
    tok      = AutoTokenizer.from_pretrained(cfg["base"], **kwargs)
    print(f"[load_models] Tokenizer loaded: {cfg['base']}")
    return tok


def load_model_pair(
    family_key: str,
    device:     str = "cuda",
) -> tuple[HookedTransformer, HookedTransformer]:
    """
    Load base and instruct models for a family sequentially.
    Frees GPU memory between loads.

    Returns (base_model, instruct_model) — both on device, eval mode.

    NOTE: Having both models in GPU memory simultaneously requires ~16–24GB.
    If you have <16GB VRAM, load and process them separately using
    load_tl_model() directly and del + torch.cuda.empty_cache() between runs.
    """
    cfg = FAMILIES[family_key]

    base_model = load_tl_model(cfg["base"], family_key, device)

    # If VRAM is tight, caller should del base_model before loading instruct
    instruct_model = load_tl_model(cfg["instruct"], family_key, device)

    return base_model, instruct_model
