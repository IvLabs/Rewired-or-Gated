"""modal_run_all_lds.py -- runs the existing, unmodified run_all_lds.py on a
Modal GPU container for the REAL (non-smoke) full-dataset LDS + CRR pass
across the 3 real model families (Llama-3.2-3B, Gemma-3-4B, Qwen-2.5-3B),
both base and instruct.

Does NOT reimplement run_all_lds.py's logic -- invokes it via subprocess
exactly as documented in its own docstring ("python run_all_lds.py
--families FAM --variants VAR"), one (family, variant) at a time so a
timeout/failure on one cell doesn't lose already-completed cells. Uses the
single-token survival cache already committed to the repo (results/<family>/
single_token_survival_<family>_<variant>.json) -- run_all_lds.py's own
_load_prompts() helper picks that up automatically, no re-filtering needed.

Mirrors modal_smoke_lds_pp.py's image/secret/volume setup. Output is tagged
with the REAL set_tag ("st", not "st_smoke") since run_all_lds.py defaults
to split="all", n_subset=None (full dataset) -- this is what
run_all_pathpatching.py's real run will read.

Usage:
    modal run singleton_fix/modal_run_all_lds.py
    modal run singleton_fix/modal_run_all_lds.py --families llama32_3b --variants base
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import modal

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = Path(_ROOT) / "results"

ALL_FAMILIES = ["llama32_3b", "gemma3_4b", "qwen25_3b"]
ALL_VARIANTS = ["base", "instruct"]

app = modal.App("bbnlp-real-lds")

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

for _family in ALL_FAMILIES:
    for _variant in ALL_VARIANTS:
        _survival_file = RESULTS_DIR / _family / f"single_token_survival_{_family}_{_variant}.json"
        if _survival_file.exists():
            _image = _image.add_local_file(
                _survival_file,
                f"/root/BBNLP/results/{_family}/single_token_survival_{_family}_{_variant}.json",
            )

# Ship in EVERY existing lds2_*/crr_* result file too (2026-07-10 fix) --
# without this, each container starts with an EMPTY results/ dir, so
# run_all_lds.py's skip-if-exists check never finds anything already done
# and silently recomputes substitution + CRR from scratch on every cell,
# even when only coherent's eap was deleted. Costs real Modal credits for
# no reason. Shipping in what's already correct makes the skip check work
# as intended -- only genuinely-missing cells (coherent eap, here) recompute.
for _family in ALL_FAMILIES:
    _family_dir = RESULTS_DIR / _family
    if _family_dir.exists():
        for _pattern in ("lds2_*.json", "crr_*.json"):
            for _f in sorted(_family_dir.glob(_pattern)):
                if "smoke" in _f.name:
                    continue
                _image = _image.add_local_file(_f, f"/root/BBNLP/results/{_family}/{_f.name}")

hf_cache = modal.Volume.from_name("bbnlp-hf-cache", create_if_missing=True)


@app.function(
    image=_image,
    gpu="L4",  # same 24GB as A10G, cheaper ($0.80/h vs $1.10/h on Modal)
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/root/.cache/huggingface": hf_cache},
    timeout=21600,  # 6h -- real full-dataset LDS (764-870 prompts x 2 prompt
    # types x 4 passes/prompt with grad checkpointing) is far longer than the
    # 40min smoke-scale timeout used elsewhere.
    single_use_containers=True,
)
def run_lds_remote(family: str, variant: str, seed: int) -> Dict:
    """Runs the EXISTING run_all_lds.py as a subprocess inside the
    container -- not a reimplementation. Scoped to one (family, variant)
    per call so a failure/timeout on one cell doesn't lose others."""
    import subprocess
    import sys

    results_dir = Path("/root/BBNLP/results") / family

    def _relevant_files():
        if not results_dir.exists():
            return {}
        return {
            f.name: f.stat().st_mtime
            for f in results_dir.glob("*.json")
            if "smoke" not in f.name and not f.name.startswith("single_token_survival")
        }

    # Snapshot mtimes of every shipped-in file BEFORE running, so the summary
    # can report what was ACTUALLY (re)computed vs. what was already correct
    # and merely shipped in unchanged (2026-07-10 fix -- the old "collected N
    # files" count included every file sitting in the folder regardless of
    # whether this run touched it, which reads as "N files recomputed" and
    # is misleading/alarming when N is mostly untouched shipped-in files).
    mtimes_before = _relevant_files()

    cmd = [
        sys.executable, "run_all_lds.py",
        "--families", family,
        "--variants", variant,
        "--seed", str(seed),
    ]
    print(f"[real-lds:{family}/{variant}] running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd="/root/BBNLP")

    mtimes_after = _relevant_files()
    written_names = sorted(
        name for name, mtime in mtimes_after.items()
        if name not in mtimes_before or mtime > mtimes_before[name]
    )
    unchanged_names = sorted(name for name in mtimes_after if name not in written_names)

    out_files: Dict[str, dict] = {}
    if results_dir.exists():
        for name in mtimes_after:
            out_files[name] = json.loads((results_dir / name).read_text(encoding="utf-8"))

    print(f"[real-lds:{family}/{variant}] subprocess exit code={proc.returncode}; "
          f"ACTUALLY (re)computed {len(written_names)} file(s): {written_names}; "
          f"{len(unchanged_names)} file(s) already correct, untouched.")

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "family": family, "variant": variant, "returncode": proc.returncode,
        "files": out_files, "written": written_names, "unchanged": unchanged_names,
    }


@app.local_entrypoint()
def main(families: str = "", variants: str = "", seed: int = 42):
    fam_list = families.split(",") if families else ALL_FAMILIES
    var_list = variants.split(",") if variants else ALL_VARIANTS

    results: List[Dict] = []
    for family in fam_list:
        for variant in var_list:
            payload = run_lds_remote.remote(family, variant, seed)
            results.append(payload)

            out_dir = RESULTS_DIR / family
            out_dir.mkdir(parents=True, exist_ok=True)
            # Only rewrite files the container ACTUALLY (re)computed. Writing
            # back the "unchanged" ones too (the old behavior) refreshed their
            # local mtime to "now" on every run even when zero GPU work
            # happened for them -- directly misleading (looked like a fresh
            # recompute when it was really a skip round-tripped through disk).
            for name in payload["written"]:
                (out_dir / name).write_text(
                    json.dumps(payload["files"][name], indent=2), encoding="utf-8",
                )
            print(f"--- {family}/{variant}: exit={payload['returncode']}, "
                  f"ACTUALLY recomputed {len(payload['written'])} file(s), "
                  f"{len(payload['unchanged'])} already correct (left untouched on disk) ---")

    print("\n\n=== REAL LDS RUN SUMMARY ===")
    for payload in results:
        status = "OK" if payload["returncode"] == 0 else f"FAILED (exit {payload['returncode']})"
        print(f"  {payload['family']}/{payload['variant']}: {status}, "
              f"recomputed={payload['written']}")
