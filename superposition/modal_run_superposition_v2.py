"""modal_run_superposition_v2.py -- runs the FIXED superposition + ablation
pipeline (superposition/superposition_v2.py) on a Modal GPU container, for the
3 real model families (Qwen-2.5-3B, Llama-3.2-3B, Gemma-3-4B), base + instruct.

Mirrors singleton_fix/modal_run_all_lds.py's image/secret/volume conventions:
  * subprocess-invokes superposition_v2.py (does NOT reimplement it), one family
    per container call so a failure on one family doesn't lose the others;
  * ships in the committed LDS score files + single-token survival caches the
    script consumes (results/ is otherwise excluded from the image);
  * writes outputs back locally under results_superposition_v2/<family>/.

Inputs each family needs (already in the repo — this is why re-running is cheap):
  results/<tag>/lds2_<tag>_<variant>_substitution_st_eap.json   (which heads)
  results/<tag>/single_token_survival_<tag>_<variant>.json      (which prompts; Qwen
                                                                  re-filters from HF)

Persistence (so a dropped laptop / network never loses GPU work):
  * Outputs are written INTO a Modal Volume ("bbnlp-sup-v2-results") mounted at
    the script's output dir, and committed after every family — they survive on
    Modal's side even if the local `modal run` disconnects.
  * Each family's files are ALSO written back to the local results_superposition_v2/
    folder as soon as that family returns (not only at the very end).
  * Run detached so the job keeps going server-side if you close your laptop:
        modal run --detach superposition/modal_run_superposition_v2.py
    then later pull anything from the volume with:
        modal volume get bbnlp-sup-v2-results / ./results_superposition_v2 --force

Usage:
    modal run --detach superposition/modal_run_superposition_v2.py
    modal run superposition/modal_run_superposition_v2.py --families qwen
    modal run superposition/modal_run_superposition_v2.py --families llama,gemma --mode ablation
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import modal

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # inner BBNLP repo
RESULTS_DIR = Path(_ROOT) / "results"
OUT_SUBDIR  = "results_superposition_v2"

# superposition_v2 family short-name  ->  LDS/result folder tag
FAM_TAG = {"qwen": "qwen25_3b", "llama": "llama32_3b", "gemma": "gemma3_4b"}
ALL_FAMILIES = list(FAM_TAG)
VARIANTS     = ["base", "instruct"]

app = modal.App("bbnlp-superposition-v2")

_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformer-lens",
        "transformers",
        "datasets",
        "python-dotenv",
        "huggingface_hub",
        "numpy",
    )
    .add_local_dir(
        _ROOT,
        remote_path="/root/BBNLP",
        ignore=[
            "**/.venv", "**/.git", "**/__pycache__", "**/results",
            "**/results_superposition", "**/results_superposition_v2",
            "**/.env", "**/*.pt", "**/*.pth", "**/.pytest_cache", "**/*.pdf",
        ],
    )
)

# results/ is excluded above, so explicitly ship the two input files each family
# needs: the LDS substitution scores (head selection) and the single-token
# survival cache (prompt set + pinned memory answer). Missing files are simply
# skipped -- Qwen has no survival cache and re-filters from HF at runtime.
for _fam, _tag in FAM_TAG.items():
    for _variant in VARIANTS:
        for _fname in (
            f"lds2_{_tag}_{_variant}_substitution_st_eap.json",
            f"single_token_survival_{_tag}_{_variant}.json",
        ):
            _src = RESULTS_DIR / _tag / _fname
            if _src.exists():
                _image = _image.add_local_file(_src, f"/root/BBNLP/results/{_tag}/{_fname}")

hf_cache = modal.Volume.from_name("bbnlp-hf-cache", create_if_missing=True)
# Persistent output volume: GPU results land here and are committed per family,
# so nothing is lost if the local `modal run` disconnects mid-run.
out_vol  = modal.Volume.from_name("bbnlp-sup-v2-results", create_if_missing=True)

# The script writes to /root/BBNLP/results_superposition_v2 — mount the volume
# THERE so every file it writes goes straight into persistent storage.
_OUT_MOUNT = f"/root/BBNLP/{OUT_SUBDIR}"


@app.function(
    image=_image,
    gpu="L4",                       # 24GB, enough for 3B/4B fp32 forward+cache
    secrets=[modal.Secret.from_name("huggingface-secret")],   # provides HF_TOKEN
    volumes={"/root/.cache/huggingface": hf_cache, _OUT_MOUNT: out_vol},
    timeout=14400,                  # 4h -- superposition + ablation, both variants
    single_use_containers=True,
)
def run_superposition_remote(family: str, mode: str, topk: int, dtype: str) -> Dict:
    """Invoke the EXISTING superposition_v2.py as a subprocess for ONE family
    (base + instruct handled inside the script). Commits the output volume so the
    files persist server-side, and returns them so the local entrypoint also
    saves them into the local results_superposition_v2/ folder."""
    import subprocess
    import sys

    out_dir = Path(_OUT_MOUNT) / family

    cmd = [
        sys.executable, "superposition/superposition_v2.py",
        "--family", family, "--mode", mode, "--topk", str(topk),
    ]
    if dtype:
        cmd += ["--dtype", dtype]
    print(f"[sup-v2:{family}] running: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, cwd="/root/BBNLP")
        returncode = proc.returncode
    finally:
        # Commit whatever was written even if the subprocess errored partway,
        # so partial progress (e.g. base done, instruct pending) is not lost.
        out_vol.commit()

    files: Dict[str, str] = {}          # relative-path -> text content
    if out_dir.exists():
        for f in sorted(out_dir.rglob("*")):
            if f.is_file():
                files[str(f.relative_to(out_dir))] = f.read_text(encoding="utf-8")

    print(f"[sup-v2:{family}] exit={returncode}; committed + produced {len(files)} file(s): "
          f"{sorted(files)}")

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"family": family, "returncode": returncode, "files": files}


@app.local_entrypoint()
def main(families: str = "", mode: str = "all", topk: int = 10, dtype: str = ""):
    fam_list = families.split(",") if families else ALL_FAMILIES
    bad = [f for f in fam_list if f not in FAM_TAG]
    if bad:
        raise SystemExit(f"unknown families {bad}; choose from {ALL_FAMILIES}")

    results: List[Dict] = []
    for family in fam_list:
        payload = run_superposition_remote.remote(family, mode, topk, dtype)
        results.append(payload)

        out_dir = Path(_ROOT) / OUT_SUBDIR / family
        for rel, content in payload["files"].items():
            dest = out_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
        print(f"--- {family}: exit={payload['returncode']}, "
              f"wrote {len(payload['files'])} file(s) to {OUT_SUBDIR}/{family}/ ---")

    print("\n=== SUPERPOSITION-V2 RUN SUMMARY ===")
    for payload in results:
        status = "OK" if payload["returncode"] == 0 else f"FAILED (exit {payload['returncode']})"
        # surface the headline numbers if the comparison file came back
        fam = payload["family"]
        comp = payload["files"].get(f"comparison_{fam}.json")
        head = ""
        if comp:
            d = json.loads(comp)
            s = d.get("_summary", {}); no = d.get("_node_overlap", {})
            head = (f"  median Δindex={s.get('median_delta_index'):+.3f}, "
                    f"role changed {s.get('n_role_changed')}/{s.get('n_heads_compared')}, "
                    f"{no.get('interpretation','')}")
        print(f"  {fam}: {status}{head}")
