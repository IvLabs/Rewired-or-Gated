"""modal_run_all_pathpatching.py -- runs the existing, unmodified
run_all_pathpatching.py on a Modal GPU container for the REAL (non-smoke)
full-dataset Path Patching pass across the 3 real model families
(Llama-3.2-3B, Gemma-3-4B, Qwen-2.5-3B), both base and instruct, both
substitution AND coherent (as of 2026-07-10, PP runs on both prompt types).

Mirrors modal_run_all_lds.py's exact pattern -- invokes run_all_pathpatching.py
via subprocess exactly as documented in its own docstring, one
(family, variant) at a time so a timeout/failure on one cell doesn't lose
already-completed cells.

REQUIRES run_all_lds.py to have already produced this family/variant's
lds2_<tag>_<ptype>_st_eap.json (both prompt types) -- run
modal_run_all_lds.py FIRST and let its local_entrypoint write the results
back to your local results/<family>/ before running this script. This
script ships in whatever lds2_*_eap*.json files currently exist on your
local disk for that family (both flavors' main + shuffled), so pull the LDS
results locally before invoking PP.

Timeout note: coherent prompts are ~7-8x longer than substitution (measured:
~118 vs ~15 tokens), and PP does ~40 forward passes per prompt (20 heads x
2), so a coherent cell is far slower than substitution. Set generously
(12h/cell) since a mid-run Modal timeout on a fresh container LOSES the
in-container checkpoint (this container's local disk is not a persistent
Modal Volume) -- the checkpoint mechanism added 2026-07-10 protects against
in-process crashes/bad prompts WITHIN one container's run, not against the
container itself being killed by Modal's timeout. If you expect timeouts to
be a real risk, ask before adding a persistent Volume for results/ -- not
done here to keep this a straightforward mirror of the existing LDS script.

Usage:
    modal run singleton_fix/modal_run_all_pathpatching.py
    modal run singleton_fix/modal_run_all_pathpatching.py --families llama32_3b --variants base
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

app = modal.App("bbnlp-real-pp")

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

# Ship in the single-token survival cache (PP's own _load_prompts() needs it,
# same as LDS) AND whatever lds2_*_eap*.json files currently exist locally
# (the files run_all_lds.py / modal_run_all_lds.py just produced) -- PP
# selects its top-k heads from the eap flavor and refuses to run without it.
for _family in ALL_FAMILIES:
    for _variant in ALL_VARIANTS:
        _survival_file = RESULTS_DIR / _family / f"single_token_survival_{_family}_{_variant}.json"
        if _survival_file.exists():
            _image = _image.add_local_file(
                _survival_file,
                f"/root/BBNLP/results/{_family}/single_token_survival_{_family}_{_variant}.json",
            )
    _family_dir = RESULTS_DIR / _family
    if _family_dir.exists():
        for _eap_file in sorted(_family_dir.glob("lds2_*_eap*.json")):
            if "smoke" in _eap_file.name:
                continue
            _image = _image.add_local_file(
                _eap_file, f"/root/BBNLP/results/{_family}/{_eap_file.name}",
            )

hf_cache = modal.Volume.from_name("bbnlp-hf-cache", create_if_missing=True)


@app.function(
    image=_image,
    gpu="L4",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/root/.cache/huggingface": hf_cache},
    timeout=43200,  # 12h -- coherent PP is far slower than substitution
    # (see module docstring); generous on purpose since a Modal timeout on a
    # fresh container loses the in-container checkpoint.
    single_use_containers=True,
)
def run_pp_remote(family: str, variant: str, seed: int) -> Dict:
    """Runs the EXISTING run_all_pathpatching.py as a subprocess inside the
    container -- not a reimplementation. Scoped to one (family, variant)
    per call so a failure/timeout on one cell doesn't lose others."""
    import subprocess
    import sys

    eap_check = list(Path(f"/root/BBNLP/results/{family}").glob(
        f"lds2_{family}_{variant}_*_eap.json"))
    if not eap_check:
        print(f"[real-pp:{family}/{variant}] ERROR: no lds2_..._eap.json shipped in for "
              f"this variant -- run modal_run_all_lds.py first and re-invoke this script "
              f"so the fresh eap files get shipped in.")
        return {"family": family, "variant": variant, "returncode": 2, "files": {}}

    cmd = [
        sys.executable, "run_all_pathpatching.py",
        "--families", family,
        "--variants", variant,
        "--seed", str(seed),
    ]
    print(f"[real-pp:{family}/{variant}] running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd="/root/BBNLP")

    results_dir = Path("/root/BBNLP/results") / family
    out_files: Dict[str, dict] = {}
    if results_dir.exists():
        for f in sorted(results_dir.glob("*.json")):
            if "smoke" in f.name or f.name.startswith("single_token_survival"):
                continue
            out_files[f.name] = json.loads(f.read_text(encoding="utf-8"))

    print(f"[real-pp:{family}/{variant}] subprocess exit code={proc.returncode}; "
          f"collected {len(out_files)} real-tagged result files: {list(out_files)}")

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"family": family, "variant": variant, "returncode": proc.returncode, "files": out_files}


@app.local_entrypoint()
def main(families: str = "", variants: str = "", seed: int = 42):
    fam_list = families.split(",") if families else ALL_FAMILIES
    var_list = variants.split(",") if variants else ALL_VARIANTS

    results: List[Dict] = []
    for family in fam_list:
        for variant in var_list:
            payload = run_pp_remote.remote(family, variant, seed)
            results.append(payload)

            out_dir = RESULTS_DIR / family
            out_dir.mkdir(parents=True, exist_ok=True)
            for name, content in payload["files"].items():
                (out_dir / name).write_text(json.dumps(content, indent=2), encoding="utf-8")
            print(f"--- {family}/{variant}: exit={payload['returncode']}, "
                  f"wrote {len(payload['files'])} files ---")

    print("\n\n=== REAL PATH PATCHING RUN SUMMARY ===")
    for payload in results:
        status = "OK" if payload["returncode"] == 0 else f"FAILED (exit {payload['returncode']})"
        print(f"  {payload['family']}/{payload['variant']}: {status}, {len(payload['files'])} files")
