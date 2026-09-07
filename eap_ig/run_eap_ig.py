"""End-to-end driver for edge-level attribution (EAP / EAP-IG) across all six
Phase-2 Tier-1 models: Llama-3.2-3B, Qwen2.5-3B, Gemma-3-4B, base + instruct.

Both prompt types (substitution + coherent) are covered. Results land in
`results/eapig_{model_key}_{prompt_type}.json` and, with plain EAP,
`results/eap_{model_key}_{prompt_type}.json`.

Model loading (see `load_hooked_model`) is deliberately strict:
  1. native TransformerLens (`HookedTransformer.from_pretrained`);
  2. on an *unsupported-architecture* error only, the TL HF-bridge
     (`from_pretrained_no_processing` with the HF model attached);
  3. otherwise a clear, actionable error -- the loader never silently
     switches backends or downgrades granularity.

NOTE: Llama and Gemma are gated repos -- run `huggingface-cli login` (or set
HF_TOKEN in .env) first. These are 3-4B models; use bf16 (--dtype bfloat16,
the default here) and the `--subset` flag to keep EAP-IG within budget.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from eap_ig.eap_ig import EdgeAttributionConfig, build_result_json, compute_edge_attribution
from foundation import load_conflict_prompts

RESULTS_DIR = Path(_ROOT) / "results"

# alias -> (model_key, hf_repo). Base and instruct per family.
MODEL_SPECS: Dict[str, Dict[str, str]] = {
    "llama-base": {"model_key": "llama3.2_3b_base", "repo": "meta-llama/Llama-3.2-3B"},
    "llama-instruct": {"model_key": "llama3.2_3b_instruct", "repo": "meta-llama/Llama-3.2-3B-Instruct"},
    "qwen-base": {"model_key": "qwen2.5_3b_base", "repo": "Qwen/Qwen2.5-3B"},
    "qwen-instruct": {"model_key": "qwen2.5_3b_instruct", "repo": "Qwen/Qwen2.5-3B-Instruct"},
    "gemma-base": {"model_key": "gemma3_4b_base", "repo": "google/gemma-3-4b-pt"},
    "gemma-instruct": {"model_key": "gemma3_4b_instruct", "repo": "google/gemma-3-4b-it"},
}

PROMPT_TYPES = ("substitution", "coherent")


def _is_unsupported_arch_error(exc: Exception) -> bool:
    """True only for TransformerLens 'this architecture is unknown' failures.

    We match on the specific messages TL raises from its model registry, not
    on the exception type alone -- an OOM or a gated-repo auth error is a
    different problem and must propagate untouched.

    Auth/access failures are checked FIRST and short-circuit to False even if
    an arch-unsupported marker also happens to appear in the same message.
    This matters concretely for Llama-3.2-3B/Gemma-3-4B (both gated): a
    missing/invalid HF_TOKEN triggers huggingface_hub.RepositoryNotFoundError,
    which transformers wraps as "...is not a local folder and is not a valid
    model identifier..." -- containing "not a valid", one of the arch markers
    below. Without this exclusion, a plain auth problem would be misreported
    as "upgrade TransformerLens" instead of "check your HF_TOKEN".
    """
    msg = str(exc).lower()
    auth_markers = (
        "gated repo", "private repository", "pass a token", "authenticated",
        "access to model", "restricted", "401 client error", "403 client error",
    )
    if any(m in msg for m in auth_markers):
        return False
    markers = (
        "not a valid", "not supported", "not officially supported",
        "unsupported", "could not find", "is not in",
        "official_model_names", "no model config",
    )
    return any(m in msg for m in markers)


def load_hooked_model(repo: str, dtype: Optional[str] = "bfloat16", device: Optional[str] = None):
    """Load `repo` as a TransformerLens HookedTransformer, strictly.

    native -> TL HF-bridge -> explicit error. Coarse granularity is NOT
    chosen here; it stays a deliberate `--granularity coarse` flag on the run.
    """
    from foundation import load_model

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        model = load_model(repo, device=device, dtype=dtype)
        print(f"[loader] {repo}: native TransformerLens")
        return model
    except Exception as exc:  # noqa: BLE001 -- re-raised below unless arch-unsupported
        if not _is_unsupported_arch_error(exc):
            raise
        native_err = exc
        print(f"[loader] {repo}: native TL unsupported ({type(exc).__name__}); "
              f"trying HF-bridge")

    try:
        from transformer_lens import HookedTransformer
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf_token = os.getenv("HF_TOKEN")
        torch_dtype = getattr(torch, dtype) if dtype else None
        hf_model = AutoModelForCausalLM.from_pretrained(
            repo, token=hf_token, torch_dtype=torch_dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(repo, token=hf_token)
        model = HookedTransformer.from_pretrained_no_processing(
            repo, hf_model=hf_model, tokenizer=tokenizer, device=device,
        )
        model.eval()
        model.requires_grad_(False)
        print(f"[loader] {repo}: TL HF-bridge (from_pretrained_no_processing)")
        return model
    except Exception as exc:  # noqa: BLE001
        if not _is_unsupported_arch_error(exc):
            raise
        bridge_err = exc  # keep for the final error message -- do not discard

    import transformer_lens

    raise RuntimeError(
        f"{repo} is not supported by the installed TransformerLens "
        f"{getattr(transformer_lens, '__version__', '?')}; EAP-IG at full "
        f"granularity is unavailable. Options: (a) upgrade TransformerLens, "
        f"(b) use an nnsight backend, or (c) re-run with "
        f"`--granularity coarse` to accept coarse (block-level, no split-qkv) "
        f"edges. Native error: {native_err}. HF-bridge error: {bridge_err}"
    )


def analyse(
    model,
    model_key: str,
    repo: str,
    prompt_type: str,
    granularity: str,
    ig_steps: int,
    subset: Optional[int],
    also_plain: bool,
    top_k: int,
    force: bool,
) -> List[str]:
    """Run EAP-IG for one (already-loaded model, prompt_type) pair.

    The model is loaded once per alias by the caller (`main`) and reused
    across both prompt types -- loading it here per-call would mean every
    3-4B model gets downloaded/moved-to-GPU twice under the default
    `--prompt_type both`.
    """
    outputs: List[str] = []

    igp = RESULTS_DIR / f"eapig_{model_key}_{prompt_type}.json"
    eap = RESULTS_DIR / f"eap_{model_key}_{prompt_type}.json"
    # write_plain must mirror EXACTLY the condition the `_run(1, eap)` call
    # below uses -- eap.json is only ever written when also_plain AND
    # ig_steps != 1, so the skip-cache check has to agree, or `--ig-steps 1`
    # runs can never be considered "already done" (eap.json never exists for
    # that combination) and get recomputed from scratch on every invocation.
    write_plain = also_plain and ig_steps != 1
    want = [igp] + ([eap] if write_plain else [])
    if not force and all(p.exists() for p in want):
        print(f"[run_eap_ig] all outputs exist for {model_key}/{prompt_type}; skipping")
        return [str(p) for p in want]

    prompts = load_conflict_prompts(prompt_type=prompt_type)
    if subset is not None and subset < len(prompts):
        print(f"[run_eap_ig] subsetting {len(prompts)} -> {subset} prompts "
              f"(spec scopes edge attribution to a subset)")
        prompts = prompts[:subset]

    def _run(steps: int, out_path: Path) -> None:
        config = EdgeAttributionConfig(n_steps=steps, granularity=granularity, contract_device="cpu")
        meta: Dict = {}
        scores = compute_edge_attribution(model, prompts, config, _meta_out=meta)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        payload = build_result_json(
            scores, meta, model_key, prompt_type, tokenizer_name=repo, top_k=top_k,
        )
        import json
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[run_eap_ig] saved {out_path}")

    _run(ig_steps, igp)
    outputs.append(str(igp))

    if write_plain:
        _run(1, eap)
        outputs.append(str(eap))

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-model edge attribution (EAP / EAP-IG)")
    parser.add_argument(
        "--model", choices=list(MODEL_SPECS) + ["all"], default="all",
        help="model alias to run (default: all six)",
    )
    parser.add_argument("--prompt_type", choices=list(PROMPT_TYPES) + ["both"], default="both")
    parser.add_argument(
        "--granularity", choices=["full", "coarse"], default="full",
        help="'coarse' is the deliberate degraded mode for archs TL can't expand",
    )
    parser.add_argument("--ig-steps", type=int, default=5, help="IG integration steps (1 = plain EAP)")
    parser.add_argument("--subset", type=int, default=100, help="cap prompts per run; <=0 means use all")
    parser.add_argument("--no-plain", action="store_true", help="skip the extra plain-EAP baseline pass")
    parser.add_argument("--top-k", type=int, default=1000, help="number of top edges to persist per file")
    parser.add_argument("--dtype", default="bfloat16", help="model dtype; pass 'float32' to disable")
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dtype = None if args.dtype.lower() == "float32" else args.dtype
    aliases = list(MODEL_SPECS) if args.model == "all" else [args.model]
    ptypes = list(PROMPT_TYPES) if args.prompt_type == "both" else [args.prompt_type]
    subset = None if args.subset is not None and args.subset <= 0 else args.subset

    for alias in aliases:
        spec = MODEL_SPECS[alias]
        model_key, repo = spec["model_key"], spec["repo"]

        print(f"[run_eap_ig] loading {alias} ({repo})")
        try:
            model = load_hooked_model(repo, dtype=dtype, device=args.device)
        except Exception as exc:  # noqa: BLE001 -- keep going across models
            print(f"[run_eap_ig] FAILED to load {alias}: {exc}")
            continue

        try:
            for ptype in ptypes:
                try:
                    analyse(
                        model=model, model_key=model_key, repo=repo, prompt_type=ptype,
                        granularity=args.granularity, ig_steps=args.ig_steps, subset=subset,
                        also_plain=not args.no_plain, top_k=args.top_k, force=args.force,
                    )
                except Exception as exc:  # noqa: BLE001 -- keep going across prompt types
                    print(f"[run_eap_ig] FAILED {alias}/{ptype}: {exc}")
        finally:
            # Always release the model, even if analyse() raised for every
            # prompt type -- a mid-run failure must not leave a 3-4B model's
            # GPU memory unfreed before the next alias in this loop tries to
            # load its own model.
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
