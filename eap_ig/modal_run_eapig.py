"""modal_run_eapig.py — run SINGLE-TOKEN EAP-IG on a Modal cloud GPU for all 3
model families x base/instruct.

Single-token substitution only (context<->memory is a one-token swap, so main
and counterfactual stay equal-length and every position is comparable). This is
the cloud counterpart of eap_ig/run_eapig.py: the SAME driver
(compute_edge_attribution_single_token) and the SAME committed single-token
survival caches, just executed on a rented GPU because Qwen2.5-3B (~6.2GB),
Llama-3.2-3B (~6.4GB) and Gemma-3-4B (~8.6GB) don't fit a 6GB laptop card at
full granularity.

Progress visibility (per the run request):
  * a per-variant banner prints WHICH model/repo/granularity is running;
  * a live token preview prints the model's top next-token predictions on a few
    prompts, so you can see real tokens flowing before scoring starts;
  * the driver's own tqdm bar streams the per-prompt progress (N/764 ...);
  * a nonzero-edge sanity line prints after each pass (guards the silent-zero
    failure mode where a misconfigured hook yields all-zero gradients).

Reliability (two independent layers, since a multi-hour run over home wifi WILL
occasionally lose the connection):
  1. Dispatch uses .spawn() + .get(), not .remote() -- the remote container's
     lifetime is decoupled from this local process continuously polling it (this
     is what Modal's own "may be cancelled, use .spawn()" warning is telling you
     to do). Combined with `--detach` (see below), a local disconnect no longer
     risks the remote job at all.
  2. Even so: every 50 prompts, the running attribution is checkpointed to the
     durable bbnlp-eapig-results Volume under checkpoints/<family>/... . If a
     container is EVER actually killed (GPU preemption, quota, anything), just
     re-run the EXACT SAME command -- the new container reads the checkpoint and
     resumes from the last saved prompt instead of starting over from 0. A
     completed pass deletes its own checkpoint.

One-time setup (same as the other modal_*.py scripts):
    pip install modal
    modal setup
    # Modal dashboard -> Secrets: create "huggingface-secret" with
    # HF_TOKEN = <token that accepted the Llama-3.2 + Gemma-3 licenses>.
    # Qwen is public and ignores it.

Usage:
    # ALWAYS use --detach for real (non-smoke) runs -- keeps the job alive on
    # Modal's servers even if your laptop sleeps/loses network. Check progress
    # any time on the Modal dashboard's App Logs tab, or by re-running the same
    # command later (it just re-attaches / resumes).

    # Everything: 3 families x base/instruct (default), full granularity
    modal run --detach eap_ig/modal_run_eapig.py

    # One family / one variant
    modal run --detach eap_ig/modal_run_eapig.py --families qwen25_3b --variants base

    # Quick cloud smoke (no --detach needed, this is short): coarse
    # granularity, 8 prompts, no plain-EAP pass
    modal run eap_ig/modal_run_eapig.py --families qwen25_3b --variants base \
        --granularity coarse --n-subset 8 --no-plain

    # Bigger card for extra safety margin (default is A10G)
    modal run --detach eap_ig/modal_run_eapig.py --families gemma3_4b --gpu A100-40GB

Results land LOCALLY (written by the local entrypoint, not the container).
Granularity is always in the filename so coarse and full never collide; smoke
(--n-subset) runs are quarantined under a smoke/ subdir so they can't be mistaken
for the real result:
    results/<family>/eapig_<tag>_substitution_st_<granularity>.json
    results/<family>/eap_<tag>_substitution_st_<granularity>.json   (unless --no-plain)
    results/<family>/smoke/eapig_<tag>_substitution_st_<gran>_smoke_n<N>.json
Each carries the single-token _provenance block.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import modal

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = Path(_ROOT) / "results"
DTYPE = "bfloat16"
SET_TAG = "st"
PROMPT_TYPE = "substitution"

# (family, variant) -> {tag, repo}. Kept in sync with eap_ig/run_eapig.py's
# FAMILY_SPECS; duplicated here so this script stays self-contained (the local
# entrypoint must not need transformer_lens just to iterate cells).
FAMILY_SPECS: Dict[str, Dict[str, Dict[str, str]]] = {
    "llama32_3b": {
        "base":     {"tag": "llama32_3b_base",     "repo": "meta-llama/Llama-3.2-3B"},
        "instruct": {"tag": "llama32_3b_instruct", "repo": "meta-llama/Llama-3.2-3B-Instruct"},
    },
    "gemma3_4b": {
        "base":     {"tag": "gemma3_4b_base",     "repo": "google/gemma-3-4b-pt"},
        "instruct": {"tag": "gemma3_4b_instruct", "repo": "google/gemma-3-4b-it"},
    },
    "qwen25_3b": {
        "base":     {"tag": "qwen25_3b_base",     "repo": "Qwen/Qwen2.5-3B"},
        "instruct": {"tag": "qwen25_3b_instruct", "repo": "Qwen/Qwen2.5-3B-Instruct"},
    },
}
ALL_FAMILIES = list(FAMILY_SPECS)
ALL_VARIANTS = ["base", "instruct"]

# ---------------------------------------------------------------------------
# Modal app / image / cache volume
# ---------------------------------------------------------------------------

app = modal.App("bbnlp-eapig-single-token")

_CACHE_FILES = [
    f"results/{fam}/single_token_survival_{fam}_{variant}.json"
    for fam in ALL_FAMILIES for variant in ALL_VARIANTS
]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformer-lens",
        "transformers",
        "datasets",
        "python-dotenv",
        "huggingface_hub",
        "tqdm",
    )
    # Ship the repo code (foundation.py, contract.py, eap_ig/, singleton_fix/)
    # but NOT the results tree (large: archives, gpt2 files, intersection JSON).
    .add_local_dir(
        _ROOT,
        remote_path="/root/BBNLP",
        ignore=[
            "**/.venv", "**/.git", "**/__pycache__", "**/results",
            "**/.env", "**/*.pt", "**/*.pth", "**/.pytest_cache",
        ],
    )
)
# ...then add back JUST the 6 single-token survival caches the loader reads
# (results/ was ignored above). No model load or ParaConflict re-fetch needed.
for _cf in _CACHE_FILES:
    image = image.add_local_file(os.path.join(_ROOT, _cf), f"/root/BBNLP/{_cf}")

# Persist downloaded weights across runs so a re-invocation doesn't re-pull 6-8GB.
hf_cache = modal.Volume.from_name("bbnlp-hf-cache", create_if_missing=True)

# Durable copy of every result the container produces. The local entrypoint also
# writes results to your disk from the returned payload, but a multi-hour full run
# may outlive a laptop's network/sleep -- this Volume means a finished variant is
# NEVER lost. Recover any file with:
#   modal volume get bbnlp-eapig-results <family>/<file>.json ./results/<family>/
results_vol = modal.Volume.from_name("bbnlp-eapig-results", create_if_missing=True)


# ---------------------------------------------------------------------------
# Remote GPU function
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    # A10G, not L4: an identical qwen25_3b/base full run measured ~5-7s/prompt on
    # A10G vs ~21-25s/prompt on L4 (roughly 3-4x) -- this workload is backward-
    # pass/memory-bandwidth heavy and A10G's ~600GB/s beats L4's ~300GB/s enough
    # that A10G is cheaper in TOTAL cost despite its higher $/hr. Override with
    # --gpu if you want to re-test this on your account.
    gpu="A10G",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/root/.cache/huggingface": hf_cache, "/out": results_vol},
    # 8h ceiling per variant (one container = one variant = eapig + plain-EAP).
    # Measured on A10G: Llama full ~8.3s/prompt (764 rows -> ~106min eapig +
    # ~21min plain-EAP), Gemma full ~11.6s/prompt (870 rows -> ~168+34 = ~3.4h).
    # L4 is up to ~2x slower on this bandwidth-bound workload, so Gemma-on-L4 can
    # approach ~6h in one call -> 8h gives real margin. (A high timeout is free;
    # you're only billed for time actually used.)
    timeout=28800,
    # Fresh container per call so the previous variant's ~6-8GB weights are never
    # still resident when the next variant loads (the cross-variant CUDA-OOM guard
    # the filter script needed).
    single_use_containers=True,
)
def run_variant_remote(
    family: str,
    variant: str,
    tag: str,
    repo: str,
    ig_steps: int,
    granularity: str,
    also_plain: bool,
    top_k: int,
    n_subset: Optional[int],
    seed: int,
    rel_igp: str,
    rel_eap: str,
) -> Dict[str, dict]:
    """Runs on a Modal GPU. Loads the model + the shipped single-token cache,
    runs EAP-IG, writes each result to the durable /out Volume AND returns
    {"eapig": payload, ["eap": payload]} for the local entrypoint to write to
    your disk. Reuses the SAME driver as run_eapig.py. rel_igp/rel_eap are the
    result paths relative to results/ (e.g. 'qwen25_3b/eapig_..._full.json')."""
    import sys
    sys.path.insert(0, "/root/BBNLP")

    import torch
    from eap_ig.eap_ig import EdgeAttributionConfig
    from eap_ig.eap_ig_single_token import (
        build_result_json_single_token,
        compute_edge_attribution_single_token,
    )
    from eap_ig.run_eap_ig import load_hooked_model
    from singleton_fix.single_token_loader import (
        load_cached_single_token_prompts,
        print_domain_composition,
    )

    print(f"\n{'#'*72}\n# EAP-IG (single-token) | {family}/{variant}\n"
          f"#   repo={repo}\n#   granularity={granularity}  ig_steps={ig_steps}  "
          f"also_plain={also_plain}\n{'#'*72}", flush=True)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[gpu] {gpu_name}", flush=True)

    model = load_hooked_model(repo, dtype=DTYPE)
    print(f"[model] loaded {tag}: n_layers={model.cfg.n_layers} "
          f"n_heads={model.cfg.n_heads} "
          f"n_kv_heads={getattr(model.cfg, 'n_key_value_heads', None)} "
          f"d_vocab={model.cfg.d_vocab}", flush=True)

    prompts = load_cached_single_token_prompts(family, variant, prompt_type=PROMPT_TYPE)
    print_domain_composition(prompts)

    # --- live token preview: show real next-token predictions on a few prompts
    _preview_predictions(model, prompts, k=3)

    if n_subset is not None and n_subset < len(prompts):
        import random as _random
        prompts = _random.Random(seed).sample(prompts, n_subset)
        print(f"[subset] using {len(prompts)} prompts (smoke)", flush=True)

    import json as _json
    from pathlib import Path as _Path
    out: Dict[str, dict] = {}
    _rel = {"eapig": rel_igp, "eap": rel_eap}

    # Checkpoint dir on the DURABLE /out Volume (not local container disk) --
    # a fresh container that resumes this same (family, variant, granularity,
    # key, steps, subset) cell after a crash mounts the same Volume and finds
    # the checkpoint the killed container last committed.
    subset_tag = f"_smoke_n{n_subset}" if n_subset is not None else ""

    def _ckpt_path(key: str, steps: int) -> Path:
        from pathlib import Path as _P
        return _P("/out/checkpoints") / family / (
            f"{tag}_{PROMPT_TYPE}_{SET_TAG}_{granularity}{subset_tag}_{key}_steps{steps}.pt")

    def _run(steps: int, key: str) -> None:
        ckpt_path = _ckpt_path(key, steps)
        print(f"\n[pass] {key}: n_steps={steps} over {len(prompts)} prompts "
              f"(checkpoint every 50 -> {ckpt_path}) ...", flush=True)
        cfg = EdgeAttributionConfig(n_steps=steps, granularity=granularity, contract_device="cpu")
        meta: Dict = {}
        scores = compute_edge_attribution_single_token(
            model, prompts, cfg, _meta_out=meta,
            checkpoint_path=ckpt_path, checkpoint_every=50,
            on_checkpoint=lambda p: results_vol.commit(),
            # Labels this container's tqdm bar so concurrent containers (e.g.
            # base + instruct spawned together) are distinguishable in one
            # shared terminal instead of two anonymous "EAP-IG-st" bars
            # interleaving with no way to tell them apart.
            desc=f"EAP-IG-st[{family}/{variant}/{key}]",
        )
        nonzero = sum(1 for v in scores.values() if v != 0.0)
        frac = 100.0 * nonzero / len(scores) if scores else 0.0
        print(f"[sanity] {key}: nonzero edges {nonzero}/{len(scores)} ({frac:.1f}%) "
              f"| length_mismatch={meta.get('n_skipped_length_mismatch', 0)}", flush=True)
        if nonzero == 0:
            print(f"[sanity] WARNING: ALL edges are zero for {key} -- the hooks likely "
                  f"did not populate for this architecture/granularity. Do NOT trust "
                  f"this result; try --granularity coarse.", flush=True)
        payload = build_result_json_single_token(scores, meta, tag, tokenizer_name=repo, top_k=top_k)
        out[key] = payload
        # Durable write to the /out Volume so this result survives even if the
        # local entrypoint disconnects before the call returns.
        vol_path = _Path("/out") / _rel[key]
        vol_path.parent.mkdir(parents=True, exist_ok=True)
        vol_path.write_text(_json.dumps(payload, indent=2), encoding="utf-8")
        results_vol.commit()
        print(f"[durable] committed to Volume: {_rel[key]}", flush=True)

    _run(ig_steps, "eapig")
    if also_plain and ig_steps != 1:
        _run(1, "eap")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def _preview_predictions(model, prompts, k: int = 3) -> None:
    """Print the model's top-5 next-token predictions on the first k prompts, so
    the run shows real tokens (not just a progress %). Teacher-forced: EAP-IG
    never samples, this is purely a visibility diagnostic."""
    import torch
    print(f"[preview] top-5 next-token predictions on first {k} prompts:", flush=True)
    for p in prompts[:k]:
        toks = model.to_tokens(p.text)
        with torch.no_grad():
            logits = model(toks)
        top = torch.topk(logits[0, -1], 5).indices.tolist()
        decoded = [repr(model.tokenizer.decode([t])) for t in top]
        print(f"  ctx={p.context_answer!r} -> mem={p.memory_answer!r} | "
              f"next-token top5: {', '.join(decoded)}", flush=True)


# ---------------------------------------------------------------------------
# Local entrypoint (runs on YOUR machine; dispatches GPU calls, writes files)
# ---------------------------------------------------------------------------

def _out_paths(family: str, tag: str, granularity: str, subset: Optional[int]):
    """Result paths for (eapig, eap). Granularity is ALWAYS in the filename so a
    coarse run and a full run never collide (and can't masquerade as each other).
    A subset (smoke) run is quarantined under results/<family>/smoke/ with an
    _smoke_n<N> marker so an 8-prompt smoke can never sit in a canonical path and
    be mistaken for the real 764/870-row result."""
    base = RESULTS_DIR / family
    if subset is not None:
        base = base / "smoke"
        suffix = f"_{granularity}_smoke_n{subset}"
    else:
        suffix = f"_{granularity}"
    igp = base / f"eapig_{tag}_{PROMPT_TYPE}_{SET_TAG}{suffix}.json"
    eap = base / f"eap_{tag}_{PROMPT_TYPE}_{SET_TAG}{suffix}.json"
    return igp, eap


@app.local_entrypoint()
def main(
    families: str = ",".join(ALL_FAMILIES),
    variants: str = ",".join(ALL_VARIANTS),
    granularity: str = "full",
    ig_steps: int = 5,
    n_subset: int = -1,
    no_plain: bool = False,
    top_k: int = 1000,
    seed: int = 42,
    gpu: str = "A10G",
    force: bool = False,
) -> None:
    fam_list = [f.strip() for f in families.split(",") if f.strip()]
    var_list = [v.strip() for v in variants.split(",") if v.strip()]
    subset = None if n_subset is not None and n_subset <= 0 else n_subset
    also_plain = not no_plain
    # Let --gpu override the decorator default (e.g. A100-40GB for extra margin).
    remote_fn = run_variant_remote.with_options(gpu=gpu)

    # --- Phase 1: spawn every requested (family, variant) cell that isn't
    # already done. spawn() returns immediately (it does NOT wait for the GPU
    # job), so this loop finishes in seconds regardless of how many cells there
    # are -- Modal then runs all of them CONCURRENTLY on separate containers
    # (subject to your account's GPU concurrency limit; if you hit it, extra
    # cells just queue and start as earlier ones finish -- no worse than
    # running them one at a time). This is how "run Llama and Gemma at once"
    # works: just request both families in one command, or run two separate
    # `modal run` commands -- either way, dispatch is independent per cell so
    # they overlap on Modal's side. Wall-clock time drops to roughly the
    # SLOWEST cell instead of the sum of all of them; total billed GPU-seconds
    # is unchanged.
    pending: List[tuple] = []  # (family, variant, igp, eap, call)
    for family in fam_list:
        if family not in FAMILY_SPECS:
            print(f"[skip] unknown family {family!r}")
            continue
        for variant in var_list:
            spec = FAMILY_SPECS[family][variant]
            tag, repo = spec["tag"], spec["repo"]
            igp, eap = _out_paths(family, tag, granularity, subset)
            want = [igp] + ([eap] if (also_plain and ig_steps != 1) else [])
            if not force and all(p.exists() for p in want):
                print(f"[local] outputs exist for {family}/{variant}; skipping (use --force).")
                continue

            rel_igp = str(igp.relative_to(RESULTS_DIR)).replace(os.sep, "/")
            rel_eap = str(eap.relative_to(RESULTS_DIR)).replace(os.sep, "/")
            # spawn(), not remote(): remote() is a blocking RPC whose result depends
            # on this local process continuously polling Modal -- if your network
            # drops while polling, Modal's own client warns the call "may be
            # cancelled". spawn() dispatches the job and returns immediately; the
            # container's lifetime is no longer tied to this local process at all.
            call = remote_fn.spawn(
                family=family, variant=variant, tag=tag, repo=repo,
                ig_steps=ig_steps, granularity=granularity, also_plain=also_plain,
                top_k=top_k, n_subset=subset, seed=seed,
                rel_igp=rel_igp, rel_eap=rel_eap,
            )
            print(f"[local] spawned {family}/{variant} -> call_id={call.object_id}")
            pending.append((family, variant, igp, eap, call))

    if not pending:
        print("\n[local] nothing to do (all outputs already exist -- use --force to redo).")
        return

    print(f"\n[local] {len(pending)} cell(s) running on Modal (concurrently, subject to "
          f"your account's GPU quota). Waiting for results -- if THIS local process "
          f"dies while waiting, the jobs keep running remotely; re-run this same "
          f"command later and it will resume from checkpoints / pick up finished "
          f"results.\n")

    # --- Phase 2: collect results. .get() blocks per-call, but since every cell
    # was already dispatched above, they finish in whatever order Modal completes
    # them -- this loop's ORDER of printing/writing is just submission order, not
    # completion order, which only affects console readability, not correctness.
    for family, variant, igp, eap, call in pending:
        print(f"[local] waiting on {family}/{variant} (call_id={call.object_id}) ...")
        payloads = call.get()
        out_dir = igp.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        if "eapig" in payloads:
            igp.write_text(json.dumps(payloads["eapig"], indent=2), encoding="utf-8")
            print(f"[local] wrote {igp}")
        if "eap" in payloads:
            eap.write_text(json.dumps(payloads["eap"], indent=2), encoding="utf-8")
            print(f"[local] wrote {eap}")

    print("\n[local] done.")
