"""modal_run_all_pp.py -- runs the existing, unmodified run_all_pathpatching.py
on a Modal GPU container (NVIDIA L4) for the REAL (non-smoke) Path Patching
pass, causally verifying LDS's top-k heads on the substitution prompt set
(spec Part H: PP is substitution-only).

Does NOT reimplement run_all_pathpatching.py's logic -- invokes it via
subprocess exactly as documented in its own docstring ("python
run_all_pathpatching.py --families FAM --variants VAR"), one (family,
variant) at a time so a timeout/failure on one cell doesn't lose already-
completed cells. Requires run_all_lds.py to have already produced the
lds2_<tag>_substitution_st_eap.json file for each (family, variant) --
PP selects its top-k heads from those scores.

Runs Python with `-u` (unbuffered stdout) so the existing tqdm progress bar
in path_patching.run_path_patching (desc="patching prompts") actually
streams per-prompt ETA into the captured log instead of being hidden until
the whole cell finishes, as happened with LDS's default-buffered subprocess.

Mirrors modal_run_all_lds.py's image/secret/volume setup, but on an L4 GPU
instead of A10G.

Checkpoint persistence: results/ inside the container is backed by the
persistent Modal Volume "bbnlp-pp-checkpoints" (mounted at
/root/BBNLP/results), not container-local disk. run_path_patching()'s
built-in checkpoint_path (results/<family>/<...>.checkpoint.json, written
every 50 prompts -- see path_patching.py) therefore survives a container
death (network blip, Modal preemption, etc): the NEXT container to run the
same (family, variant) cell picks the volume back up and
_load_pp_checkpoint() resumes mid-cell instead of restarting at prompt 0.
Required inputs (lds2_*_eap*.json, single_token_survival_*.json) are shipped
into the image at /root/BBNLP/_seed_inputs/ and copied into the volume-backed
results/ dir on first use only (never overwriting a fresher/checkpointed
file). The volume is committed periodically (every 30s) while the subprocess
runs, and once more when it exits, so a mid-run kill loses at most ~30s +
one checkpoint interval of work instead of the whole cell.

Usage:
    modal run singleton_fix/modal_run_all_pp.py --families llama32_3b --variants base,instruct
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
            "**/.env", "**/*.pt", "**/*.pth", "**/.pytest_cache", "**/logs",
        ],
    )
)

# Ship the survival caches AND the already-completed real LDS output into a
# SEPARATE staging path (not results/ -- that's now volume-backed, see
# module docstring) so run_pp_remote can seed them into the volume on first
# use without ever clobbering a fresher/checkpointed file already there.
for _family in ALL_FAMILIES:
    for _variant in ALL_VARIANTS:
        _survival_file = RESULTS_DIR / _family / f"single_token_survival_{_family}_{_variant}.json"
        if _survival_file.exists():
            _image = _image.add_local_file(
                _survival_file,
                f"/root/BBNLP/_seed_inputs/{_family}/single_token_survival_{_family}_{_variant}.json",
            )
    _family_dir = RESULTS_DIR / _family
    if _family_dir.exists():
        for _f in _family_dir.glob("*.json"):
            if "smoke" in _f.name or _f.name.startswith("single_token_survival"):
                continue
            _image = _image.add_local_file(_f, f"/root/BBNLP/_seed_inputs/{_family}/{_f.name}")

hf_cache = modal.Volume.from_name("bbnlp-hf-cache", create_if_missing=True)
# Persistent across container restarts -- backs results/ (outputs AND
# run_path_patching's checkpoint files) so a killed container's progress
# is resumable by the next one instead of lost. See module docstring.
pp_checkpoints = modal.Volume.from_name("bbnlp-pp-checkpoints", create_if_missing=True)


@app.function(
    image=_image,
    gpu="L4",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/BBNLP/results": pp_checkpoints,
    },
    timeout=21600,  # 6h, same headroom as the real LDS run
    single_use_containers=True,
)
def run_pp_remote(family: str, variant: str, seed: int) -> Dict:
    """Runs the EXISTING run_all_pathpatching.py as a subprocess -- not a
    reimplementation. `-u` keeps stdout unbuffered so tqdm's progress bar
    (with ETA) actually streams instead of appearing only at cell-end.

    results/ is volume-backed (see module docstring): seeds required LDS
    inputs into it on first use, then commits the volume periodically while
    the subprocess runs so a killed container doesn't lose PP's own
    checkpoint progress."""
    import shutil
    import subprocess
    import sys
    import threading
    import time

    seed_dir = Path("/root/BBNLP/_seed_inputs") / family
    dest_dir = Path("/root/BBNLP/results") / family
    dest_dir.mkdir(parents=True, exist_ok=True)
    if seed_dir.exists():
        for f in seed_dir.glob("*.json"):
            dest = dest_dir / f.name
            if not dest.exists():  # never clobber a fresher/checkpointed file
                shutil.copy(f, dest)

    stop_commit = threading.Event()

    def _periodic_commit(interval: float = 30.0) -> None:
        while not stop_commit.wait(interval):
            try:
                pp_checkpoints.commit()
            except Exception as e:
                print(f"[real-pp:{family}/{variant}] volume commit failed (will retry): {e}")

    committer = threading.Thread(target=_periodic_commit, daemon=True)
    committer.start()

    cmd = [
        sys.executable, "-u", "run_all_pathpatching.py",
        "--families", family,
        "--variants", variant,
        "--seed", str(seed),
    ]
    print(f"[real-pp:{family}/{variant}] running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd="/root/BBNLP")

    stop_commit.set()
    committer.join()
    pp_checkpoints.commit()  # final flush so this cell's output is durable

    results_dir = Path("/root/BBNLP/results") / family
    out_files: Dict[str, dict] = {}
    if results_dir.exists():
        for f in sorted(results_dir.glob("*.json")):
            if "smoke" in f.name or f.name.startswith("single_token_survival"):
                continue
            out_files[f.name] = json.loads(f.read_text(encoding="utf-8"))

    print(f"[real-pp:{family}/{variant}] subprocess exit code={proc.returncode}; "
          f"collected {len(out_files)} result files: {list(out_files)}")

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"family": family, "variant": variant, "returncode": proc.returncode, "files": out_files}


@app.local_entrypoint()
def main(families: str = "", variants: str = "", seed: int = 42):
    import time

    from tqdm import tqdm

    fam_list = families.split(",") if families else ALL_FAMILIES
    var_list = variants.split(",") if variants else ALL_VARIANTS
    cells = [(family, variant) for family in fam_list for variant in var_list]
    n_cells = len(cells)

    results: List[Dict] = []
    run_start = time.monotonic()
    bar = tqdm(cells, total=n_cells, unit="cell", desc="real-pp cells",
               dynamic_ncols=True)
    for family, variant in bar:
        bar.set_postfix_str(f"{family}/{variant}")

        cell_start = time.monotonic()
        payload = run_pp_remote.remote(family, variant, seed)
        cell_elapsed = time.monotonic() - cell_start
        results.append(payload)

        out_dir = RESULTS_DIR / family
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, content in payload["files"].items():
            (out_dir / name).write_text(json.dumps(content, indent=2), encoding="utf-8")
        bar.write(f"--- {family}/{variant}: exit={payload['returncode']}, "
                   f"wrote {len(payload['files'])} files, took {cell_elapsed / 60:.1f} min ---")
    bar.close()
    total_elapsed = time.monotonic() - run_start
    print(f"\n[real-pp] all {n_cells} cells done in {total_elapsed / 60:.1f} min total")

    print("\n\n=== REAL PP RUN SUMMARY ===")
    for payload in results:
        status = "OK" if payload["returncode"] == 0 else f"FAILED (exit {payload['returncode']})"
        print(f"  {payload['family']}/{payload['variant']}: {status}, {len(payload['files'])} files")
