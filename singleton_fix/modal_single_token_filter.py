"""modal_single_token_filter.py — run the single-token filter (spec A2 +
A2.2) on a Modal cloud GPU for all 6 model variants, since Llama-3.2-3B
(~6.4GB bf16), Qwen2.5-3B (~6.2GB), and Gemma-3-4B (~8.6GB) all exceed a
6GB laptop GPU just to load weights.

Mirrors singleton_fix/run_single_token_filter_all.py exactly (same
FAMILIES dict, same per-row logic via the SAME imported functions, same
output schema) -- only WHERE the model-loading + filtering runs changes.
The GPU-side function returns a plain dict; the local entrypoint (running
on your laptop) writes it straight to results/<family>/... -- there is no
separate "download the results" step, the file lands locally as part of
this script running.

One-time setup (per your screenshots, already done):
    pip install modal
    modal setup
    # In the Modal dashboard -> Secrets, create a secret named
    # "huggingface-secret" with key HF_TOKEN = <your token>. Needed for the
    # two gated families (Llama, Gemma); Qwen is public and ignores it.

Usage:
    # All 6 variants (default)
    modal run singleton_fix/modal_single_token_filter.py

    # Just one family, or one variant
    modal run singleton_fix/modal_single_token_filter.py --families llama32_3b
    modal run singleton_fix/modal_single_token_filter.py --families llama32_3b --variants base

    # Recompute even if a local result file already exists
    modal run singleton_fix/modal_single_token_filter.py --force

    # After any subset finishes, rebuild the aggregate summary LOCALLY (no
    # GPU/Modal needed for this step -- it just reads the JSON files above):
    ../.venv/Scripts/python.exe singleton_fix/run_single_token_filter_all.py --summary-only
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import modal

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = Path(_ROOT) / "results"
DTYPE = "bfloat16"

FAMILIES: Dict[str, Dict[str, str]] = {
    "llama32_3b": {
        "base": "meta-llama/Llama-3.2-3B",
        "instruct": "meta-llama/Llama-3.2-3B-Instruct",
    },
    "gemma3_4b": {
        "base": "google/gemma-3-4b-pt",
        "instruct": "google/gemma-3-4b-it",
    },
    "qwen25_3b": {
        "base": "Qwen/Qwen2.5-3B",
        "instruct": "Qwen/Qwen2.5-3B-Instruct",
    },
}

# ---------------------------------------------------------------------------
# Modal app / image / cache volume
# ---------------------------------------------------------------------------

app = modal.App("bbnlp-single-token-filter")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformer-lens",
        "transformers",
        "datasets",
        "python-dotenv",
        "huggingface_hub",
    )
    # Ship this repo's code (foundation.py, contract.py, singleton_fix/) into
    # the container so the remote function can import it unchanged. Excludes
    # .env explicitly -- auth goes through the Modal Secret below, never a
    # baked-in file.
    .add_local_dir(
        _ROOT,
        remote_path="/root/BBNLP",
        ignore=[
            "**/.venv", "**/.git", "**/__pycache__", "**/results",
            "**/.env", "**/*.pt", "**/*.pth", "**/.pytest_cache",
        ],
    )
)

# Persists downloaded model weights across runs/retries so a re-invocation
# doesn't re-download 6-8GB per variant.
hf_cache = modal.Volume.from_name("bbnlp-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/root/.cache/huggingface": hf_cache},
    timeout=1800,  # 30 min ceiling per variant (download + load + filter)
    # Force a fresh container per call -- without this, Modal reuses a warm
    # container across consecutive .remote() calls to save cold-start time,
    # and the PREVIOUS variant's model stays resident in GPU memory (bit us:
    # llama32_3b/base's ~6.4GB was still loaded when llama32_3b/instruct
    # tried to load its own ~6.4GB into the same container -> CUDA OOM even
    # though a single model comfortably fits on a T4 alone).
    single_use_containers=True,
)
def check_variant_remote(model_name: str, family: str, variant: str) -> Dict:
    """Runs on a Modal GPU container. Same logic as
    run_single_token_filter_all.check_variant, via the SAME imported
    functions (not reimplemented) -- only returns the dict instead of
    writing to disk; the local entrypoint does that. Same rich schema as
    run_single_token_filter_all.check_variant: each row carries a stable
    row_index (position in ParaConflict's deterministic load order -- valid
    across every model, since load_conflict_prompts does no model-dependent
    filtering), both text variants, the full alias list, and the
    A2.1-vs-A2.2 comparison -- enough to reconstruct a ConflictPrompt later
    via load_cached_single_token_prompts() without reloading the model."""
    import sys
    sys.path.insert(0, "/root/BBNLP")

    from foundation import load_conflict_prompts, load_model
    from singleton_fix.single_token_loader import _select_memory_answer, load_single_token_prompts

    print(f"[{family}/{variant}] loading {model_name} on Modal GPU...")
    model = load_model(model_name, dtype=DTYPE)

    all_substitution = load_conflict_prompts(prompt_type="substitution")
    all_coherent = load_conflict_prompts(prompt_type="coherent")
    index_by_key = {(p.domain, p.clean_text): idx for idx, p in enumerate(all_substitution)}
    coherent_text_by_key = {(p.domain, p.clean_text): p.text for p in all_coherent}

    prompts = load_single_token_prompts(model, prompt_type="substitution")

    n_a22_changed = 0
    rows: List[Dict] = []
    for p in prompts:
        naive = _select_memory_answer(model, p.memory_aliases, use_logprob_refinement=False)
        changed = naive != p.memory_answer
        if changed:
            n_a22_changed += 1
        key = (p.domain, p.clean_text)
        rows.append({
            "row_index": index_by_key.get(key),
            "domain": p.domain,
            "clean_text": p.clean_text,
            "substitution_text": p.text,
            "coherent_text": coherent_text_by_key.get(key),
            "memory_aliases": p.memory_aliases,
            "context_answer": p.context_answer,
            "memory_answer": p.memory_answer,
            "a21_naive_answer": naive,
            "a22_changed": changed,
        })

    domain_counts: Dict[str, int] = {}
    for p in prompts:
        domain_counts[p.domain] = domain_counts.get(p.domain, 0) + 1
    domain_counts = dict(sorted(domain_counts.items(), key=lambda kv: -kv[1]))

    result = {
        "family": family,
        "variant": variant,
        "model": model_name,
        "n_survived": len(prompts),
        "domain_counts": domain_counts,
        "n_a22_changed_vs_a21": n_a22_changed,
        "rows": rows,
    }
    print(f"[{family}/{variant}] done: {len(prompts)} survived")

    # Defense-in-depth: free GPU memory explicitly too (single_use_containers
    # above already guarantees a fresh container next call regardless, but
    # this costs nothing and matches run_single_token_filter_all.py's parity).
    import torch
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def _out_path(family: str, variant: str) -> Path:
    return RESULTS_DIR / family / f"single_token_survival_{family}_{variant}.json"


@app.local_entrypoint()
def main(families: str = "", variants: str = "", force: bool = False):
    """
    families: comma-separated subset of {llama32_3b, gemma3_4b, qwen25_3b}; empty = all.
    variants: comma-separated subset of {base, instruct}; empty = both.
    force: recompute even if a local result file already exists.
    """
    fam_list = families.split(",") if families else list(FAMILIES.keys())
    var_list = variants.split(",") if variants else ["base", "instruct"]

    results: List[Dict] = []
    for family in fam_list:
        for variant in var_list:
            out_path = _out_path(family, variant)
            if out_path.exists() and not force:
                print(f"[survival] {family}/{variant} already done locally -> {out_path} (skip)")
                results.append(json.loads(out_path.read_text(encoding="utf-8")))
                continue

            model_name = FAMILIES[family][variant]
            result = check_variant_remote.remote(model_name, family, variant)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"[survival] saved -> {out_path}")
            results.append(result)

    print("\n=== per-variant survival ===")
    for r in results:
        print(f"{r['family']}/{r['variant']}: {r['n_survived']} survived, "
              f"{r['n_a22_changed_vs_a21']} rows where A2.2 changed the alias vs A2.1")
