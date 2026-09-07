"""modal_smoke_lds_pp.py -- small-subset smoke test for the LDS + Path
Patching (+ behavioral CRR) pipeline against the real target models
(Llama-3.2-3B, Gemma-3-4B, Qwen-2.5-3B), on Modal cloud GPU since none of
these fit on a 6GB laptop GPU even to load weights.

LDS and Path Patching are already code-complete in run_llama32.py /
run_gemma3.py / singleton_fix/run_qwen25.py -- nothing new needed there.
What has NEVER been run is the actual pipeline against the 3 real models
(only the
single-token *filtering* step has real output so far). This script is the
first real run: a small n_subset per variant to confirm the whole chain
(cache load -> EAP screen (gradnorm/gradact/eap) -> behavioral+logprob CRR
-> PP-verify on top-k heads) executes cleanly end-to-end on each family,
without paying for a full run before knowing it works.

Calls each script's OWN run_variant() (not a reimplementation) so this
smoke test exercises the exact code path a real run would use. Outputs are
tagged with split="smoke" -> set_tag="st_smoke", which keeps them in
entirely separate files from a real "st"-tagged run (see run_and_save's
skip-if-exists check) so this can never be mistaken for, or block, a real
full run later.

Ships the already-computed single-token survival cache (results/<family>/
single_token_survival_*.json) into the container so run_variant() takes
the fast cached path instead of live-filtering -- exercising the exact
cache-wiring this session added, not a fallback path.

One-time setup: same as modal_single_token_filter.py (modal setup +
"huggingface-secret" secret with HF_TOKEN). Reuses the same
"bbnlp-hf-cache" Modal Volume, so model weights already downloaded during
single-token filtering are NOT re-downloaded here.

Usage:
    # All 3 families, base variant only (default -- cheapest useful check)
    modal run singleton_fix/modal_smoke_lds_pp.py

    # Specific families/variants, bigger/smaller subset
    modal run singleton_fix/modal_smoke_lds_pp.py --families llama32_3b --variants base,instruct --n-subset 12
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import modal

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = Path(_ROOT) / "results"

FAMILY_MODULE = {
    "llama32_3b": "run_llama32",
    "gemma3_4b": "run_gemma3",
    "qwen25_3b": "singleton_fix.run_qwen25",
}
ALL_FAMILIES = list(FAMILY_MODULE.keys())
ALL_VARIANTS = ["base", "instruct"]

# ---------------------------------------------------------------------------
# Modal app / image / cache volume
# ---------------------------------------------------------------------------

app = modal.App("bbnlp-smoke-lds-pp")

_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformer-lens",
        "transformers",
        "datasets",
        "python-dotenv",
        "huggingface_hub",
    )
    .add_local_dir(
        _ROOT,
        remote_path="/root/BBNLP",
        ignore=[
            "**/.venv", "**/.git", "**/__pycache__", "**/results",
            "**/.env", "**/*.pt", "**/*.pth", "**/.pytest_cache",
        ],
    )
)

# The base add_local_dir call excludes results/ entirely (it's 7.7MB of
# mostly-irrelevant GPT-2/legacy files) -- explicitly re-add just the 6
# single-token survival cache files so run_variant()'s _load_prompts() can
# take the fast cached path instead of live-filtering.
for _family in ALL_FAMILIES:
    for _variant in ALL_VARIANTS:
        _survival_file = RESULTS_DIR / _family / f"single_token_survival_{_family}_{_variant}.json"
        if _survival_file.exists():
            _image = _image.add_local_file(
                _survival_file,
                f"/root/BBNLP/results/{_family}/single_token_survival_{_family}_{_variant}.json",
            )

hf_cache = modal.Volume.from_name("bbnlp-hf-cache", create_if_missing=True)


@app.function(
    image=_image,
    gpu="A10G",  # more headroom than the T4 used for filtering -- LDS's
    # backward passes + gradient checkpointing cost more memory than a
    # forward-only filter pass.
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/root/.cache/huggingface": hf_cache},
    timeout=2400,
    single_use_containers=True,
)
def run_smoke_remote(family: str, variant: str, n_subset: int, seed: int) -> Dict:
    """Runs on a Modal GPU container. Imports and calls the real
    run_variant() from the family's own run script -- same code path a
    full real run uses, just with a small n_subset and split="smoke" so
    outputs land in distinct st_smoke-tagged files."""
    import sys
    sys.path.insert(0, "/root/BBNLP")

    import importlib
    mod = importlib.import_module(FAMILY_MODULE[family])

    print(f"[smoke:{family}/{variant}] running LDS+CRR pass (n_subset={n_subset})...")
    # Deliberately TWO separate calls (steps split, not one combined call)
    # to exercise the exact two-stage flow run_all_lds.py +
    # run_all_pathpatching.py drive on a local GPU: LDS/CRR
    # first, then PP reading back the eap file LDS just wrote. Proves the
    # steps= split (added 2026-07-09) actually works end-to-end, not just
    # that the unchanged default steps=("lds","crr","pp") still works.
    mod.run_variant(
        variant=variant,
        n_subset=n_subset,
        no_shuffle=True,  # skip the eap shuffled-label noise-floor control -- smoke test only
        seed=seed,
        heldout_domain=None,
        split="smoke",  # -> set_tag="st_smoke", isolated from real "st" runs
        steps=("lds", "crr"),
    )
    print(f"[smoke:{family}/{variant}] running PP pass (reads back the eap file just written)...")
    mod.run_variant(
        variant=variant,
        n_subset=n_subset,
        no_shuffle=True,
        seed=seed,
        heldout_domain=None,
        split="smoke",
        steps=("pp",),
    )

    out_files: Dict[str, dict] = {}
    for f in sorted(mod.RESULTS_DIR.glob("*st_smoke*.json")):
        out_files[f.name] = json.loads(f.read_text(encoding="utf-8"))
    print(f"[smoke:{family}/{variant}] wrote {len(out_files)} result files: {list(out_files)}")

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"family": family, "variant": variant, "files": out_files}


def _summarize(payload: Dict) -> None:
    family, variant, files = payload["family"], payload["variant"], payload["files"]
    print(f"\n--- {family}/{variant} ---")
    if not files:
        print("  !! NO OUTPUT FILES -- pipeline did not complete. Check logs above.")
        return
    for name, content in files.items():
        if name.startswith("lds2_"):
            n_heads = len(content.get("scores", {}))
            print(f"  {name}: {n_heads} head scores")
        elif name.startswith("crr_"):
            beh = content.get("crr_behavioral", {})
            print(f"  {name}: total={beh.get('total')} memory={beh.get('memory')} "
                  f"context={beh.get('context')} neither={beh.get('neither')}")
        elif name.startswith("pp_topheads_"):
            n_sel = len(content.get("selected_heads", []))
            meta = content.get("meta", {})
            print(f"  {name}: {n_sel} heads patched, "
                  f"n_skipped_no_swap={meta.get('n_skipped_no_swap')}, "
                  f"n_skipped_length_mismatch={meta.get('n_skipped_length_mismatch')}, "
                  f"n_skipped_degenerate_margin={meta.get('n_skipped_degenerate_margin')}")
        else:
            print(f"  {name}: (unrecognized file, {len(content) if hasattr(content,'__len__') else '?'} entries)")


@app.local_entrypoint()
def main(families: str = "", variants: str = "base", n_subset: int = 8, seed: int = 42):
    """
    families: comma-separated subset of {llama32_3b, gemma3_4b, qwen25_3b}; empty = all 3.
    variants: comma-separated subset of {base, instruct}; default "base" only
        (cheapest check that machinery runs on all 3 families; add
        "instruct" once base is confirmed clean).
    n_subset: prompts per (variant, prompt_type) cell -- small on purpose.
    """
    fam_list = families.split(",") if families else ALL_FAMILIES
    var_list = variants.split(",") if variants else ["base"]

    results: List[Dict] = []
    for family in fam_list:
        for variant in var_list:
            payload = run_smoke_remote.remote(family, variant, n_subset, seed)
            results.append(payload)

            out_dir = RESULTS_DIR / family / "smoke"
            out_dir.mkdir(parents=True, exist_ok=True)
            for name, content in payload["files"].items():
                (out_dir / name).write_text(json.dumps(content, indent=2), encoding="utf-8")

    print("\n\n=== SMOKE TEST SUMMARY ===")
    for payload in results:
        _summarize(payload)
