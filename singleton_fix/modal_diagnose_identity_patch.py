"""One-off diagnostic: is Gemma-3-4B's identity-patch warning
(max |score| = 7.81e-02, seen during the LDS/PP smoke test) a genuine
hook-wiring bug, or bf16 floating-point noise from comparing two
separately-computed forward passes of different sequence lengths?

Compares the SAME identity-patch check at bf16 vs float32 on a tiny
2-prompt sample. If it's precision noise, float32 should collapse the
score back down near GPT-2's fp32 baseline (~1e-6, see
results/pp_topheads_gpt2_*_v2_*.json). If it stays large in fp32, it's a
real bug in the hook wiring, not precision.

Usage: modal run singleton_fix/modal_diagnose_identity_patch.py
"""
from __future__ import annotations

import os
from pathlib import Path

import modal

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = Path(_ROOT) / "results"

app = modal.App("bbnlp-diagnose-identity-patch")

_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformer-lens", "transformers", "datasets", "python-dotenv", "huggingface_hub")
    .add_local_dir(
        _ROOT, remote_path="/root/BBNLP",
        ignore=["**/.venv", "**/.git", "**/__pycache__", "**/results", "**/.env", "**/*.pt", "**/*.pth", "**/.pytest_cache"],
    )
    .add_local_file(
        RESULTS_DIR / "gemma3_4b" / "single_token_survival_gemma3_4b_base.json",
        "/root/BBNLP/results/gemma3_4b/single_token_survival_gemma3_4b_base.json",
    )
)
hf_cache = modal.Volume.from_name("bbnlp-hf-cache", create_if_missing=True)


@app.function(
    image=_image, gpu="A10G",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/root/.cache/huggingface": hf_cache},
    timeout=1200, single_use_containers=True,
)
def check(dtype: str, n_prompts: int = 2) -> float:
    import sys
    sys.path.insert(0, "/root/BBNLP")
    from foundation import load_model
    from singleton_fix.single_token_loader import load_cached_single_token_prompts
    from path_patching.path_patching import _run_identity_patch_check

    model = load_model("google/gemma-3-4b-pt", dtype=dtype)
    prompts = load_cached_single_token_prompts("gemma3_4b", "base", prompt_type="substitution")[:n_prompts]
    # Check only a handful of heads (not all 34*8=272) -- this is a
    # precision diagnostic, not a full sweep; a few heads is enough to see
    # whether the noise floor collapses.
    head_subset = [(l, h) for l in (0, 10, 20, 33) for h in (0, 3)]
    max_abs = _run_identity_patch_check(model, prompts, model.cfg.n_layers, model.cfg.n_heads, head_subset=head_subset)
    print(f"[diagnose] dtype={dtype} n_prompts={n_prompts} identity_patch_max_abs={max_abs:.6e}")
    return max_abs


@app.local_entrypoint()
def main():
    bf16_score = check.remote("bfloat16", 2)
    fp32_score = check.remote("float32", 2)
    print(f"\n=== RESULT ===\nbfloat16: {bf16_score:.6e}\nfloat32:  {fp32_score:.6e}")
    if fp32_score < 1e-4 and bf16_score > 1e-4:
        print("=> CONFIRMED: bf16 floating-point precision noise, not a hook-wiring bug. "
              "float32 collapses back under the 1e-4 threshold.")
    elif fp32_score > 1e-4:
        print("=> STILL FAILS in float32 -- this is a real hook-wiring bug, not precision noise. Investigate _make_aligned_patch_fn.")
    else:
        print("=> Inconclusive -- both under or both over threshold.")
