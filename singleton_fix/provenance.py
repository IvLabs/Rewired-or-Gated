"""Self-describing provenance header for result JSONs (spec §A3.1).

The old `_st` filename tag encoded only the DATASET filter, not the contrast
fix -- the more important change. Every LDS/PP/summary JSON this pipeline
writes should embed this block so a number always describes itself.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict

_ROOT = Path(__file__).resolve().parent.parent


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def build_provenance(
    *,
    contrast: str,
    dataset: str,
    n_used: int,
    alias_policy: str,
    prompt_type: str,
    model_tag: str,
) -> Dict:
    """Build the `_provenance` block to embed in a result JSON.

    Args:
        contrast: e.g. "inplace_swap" (this pipeline) vs "clean_text" (old, buggy).
        dataset: e.g. "single_token_st" (this pipeline's set_tag).
        n_used: number of prompts actually used (post-filtering).
        alias_policy: e.g. "first_single_token" or "first_single_token+logprob_refined".
        prompt_type: "substitution" or "coherent".
        model_tag: e.g. "qwen25_3b_base".
    """
    return {
        "contrast": contrast,
        "dataset": dataset,
        "n_used": n_used,
        "alias_policy": alias_policy,
        "code_commit": _git_commit(),
        "prompt_type": prompt_type,
        "model_tag": model_tag,
    }
