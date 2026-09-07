"""
lds_attribution.py — Node-level gradient attribution for knowledge-conflict heads.

Three flavors (gradnorm, gradact, eap) of the signed margin:

    m = ctx_log_prob − mem_log_prob   (mean teacher-forced log-prob per answer)

Gradient target:
    ∂m/∂z = ∂(ctx_log_prob)/∂z − ∂(mem_log_prob)/∂z

Both backward passes are required because ctx_log_prob and mem_log_prob come
from two separate forward passes (different input texts). Dropping either term
would under-count the heads responsible for suppressing the other answer.

Four passes per prompt:
  1. Counterfactual (no grad): cache hook_z for the EAP contrastive baseline.
     The counterfactual is an in-place swap of the conflict text
     (context_answer -> memory_answer inside prompt.text, spec §A1
     extension) — NOT prompt.clean_text, which is a structurally different,
     shorter prompt and would confound the diff.
  2. Conflict + ctx answer (grad): saves z at PROMPT positions, computes
     ctx_log_prob, backward → grad_ctx.
  3. Conflict + mem answer (grad): same, → grad_mem.
  Gradient restricted to prompt positions before subtraction (avoids shape
  mismatch when ctx and mem answers have different token counts).

Flavors (all keyed to cache["blocks.{l}.attn.hook_z"], prompt positions only):

  GRAD_NORM  : ||∂m/∂z[L,H,last_prompt]||_2  — unsigned importance.
  GRAD_ACT   : ∂m/∂z[L,H,last_prompt] · z_conflict[L,H,last_prompt]  — signed.
  EAP        : Σ_p (∂m/∂z[L,H,p] · (z_conflict[L,H,p] − z_counterfactual[L,H,p]))
               summed over the aligned (prefix ∪ suffix, provably-identical-
               token) positions between the conflict and counterfactual runs.

Outputs: results/lds2_<tag>_<prompt_type>_<set>_<flavor>.json
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import functools

import torch
import torch.utils.checkpoint as _tc
from transformer_lens import HookedTransformer

from contract import ConflictPrompt, HeadScores
from foundation import _answer_ids, answer_log_prob, load_conflict_prompts, load_model
from singleton_fix.contrast_pair import build_counterfactual_text, build_contrast_pair, alignment_indices, swap_occurred
from singleton_fix.provenance import build_provenance

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def enable_gradient_checkpointing(model: HookedTransformer) -> None:
    """Enable gradient checkpointing on each transformer block's MLP.

    We checkpoint the MLP sub-module (not the full block) for two reasons:

    1. The hook at hook_z detaches-and-re-attaches the attention output as a
       leaf tensor with requires_grad=True.  After the output-projection residual
       add, the residual stream carries requires_grad=True.  The MLP receives
       this residual as its input, so torch.utils.checkpoint correctly marks the
       MLP output as requiring grad — the chain to the logits stays intact.

       Checkpointing the full BLOCK fails because the block's INPUT (the initial
       residual stream) has requires_grad=False (frozen params, no prior leaf),
       so torch.utils.checkpoint would mark the block output as not requiring
       grad, severing the chain before backward can reach the hook_z leaves.

    2. MLPs hold the largest intermediate activations (d_ffn ≈ 4 × d_model for
       the gate and up-projection tensors), so checkpointing them gives the
       greatest memory saving per unit of extra compute.

    Must be called after load_model (which freezes all params).
    """
    for block in model.blocks:
        orig_mlp_fwd = block.mlp.forward

        @functools.wraps(orig_mlp_fwd)
        def _ckpt_mlp(*args, _orig=orig_mlp_fwd, **kwargs):
            if kwargs:
                return _tc.checkpoint(
                    functools.partial(_orig, **kwargs), *args, use_reentrant=True
                )
            return _tc.checkpoint(_orig, *args, use_reentrant=True)

        block.mlp.forward = _ckpt_mlp

_HOOK_Z_FILTER  = lambda n: "hook_z" in n  # noqa: E731
_HOOK_Z_PATTERN = re.compile(r"blocks\.(\d+)\.attn\.hook_z")

HeadKey = Tuple[int, int]
Flavors = Dict[str, Dict[HeadKey, float]]


def _score_single_prompt(
    model: HookedTransformer,
    prompt: ConflictPrompt,
) -> Optional[Flavors]:
    """Return {flavor: {(layer,head): score}} for one prompt, or None on failure."""

    ctx_ids = _answer_ids(model, prompt.context_answer)
    mem_ids = _answer_ids(model, prompt.memory_answer)
    L_ctx_ans = len(ctx_ids)
    L_mem_ans = len(mem_ids)
    ctx_text = prompt.text + " " + prompt.context_answer.strip()
    mem_text = prompt.text + " " + prompt.memory_answer.strip()

    # ── Pass 1: counterfactual (no grad) — cache hook_z for EAP baseline ─────
    # Spec §A1 extension: the in-place swap replaces the old clean_text
    # contrast (structurally different prompt) with a text identical to
    # prompt.text except at the swapped distractor/memory-answer word.
    main_tokens, cf_tokens = build_contrast_pair(model, prompt)
    cf_text = build_counterfactual_text(prompt)
    if not swap_occurred(prompt, cf_text):
        # context_answer wasn't found verbatim in prompt.text -- the in-place
        # swap was a silent no-op. Scoring this row would produce a
        # degenerate zero-effect eap result; skip it (counted as a failure
        # by _run's n_fail, same as any other None return).
        return None
    with torch.no_grad():
        _, cf_cache = model.run_with_cache(cf_text, names_filter=_HOOK_Z_FILTER)
    z_clean: Dict[int, torch.Tensor] = {}
    for name, t in cf_cache.cache_dict.items():
        m = _HOOK_Z_PATTERN.match(name)
        if m:
            z_clean[int(m.group(1))] = t  # [1, L_cf, n_heads, d_head] (name kept for diff size)

    L_prompt = main_tokens.shape[0]
    aligned_prompt_idx, aligned_cf_idx = alignment_indices(main_tokens, cf_tokens)

    # ── Hook: save hook_z with grad, restricted to prompt positions ───────────
    def _make_save_hook(store: Dict[str, torch.Tensor]):
        def _hook(value: torch.Tensor, hook):
            # Detach and require grad only for the prompt positions.
            v = value.detach().requires_grad_(True)
            store[hook.name] = v
            return v
        return _hook

    fwd_hooks = [
        (f"blocks.{l}.attn.hook_z", _make_save_hook.__func__ if False else None)
        for l in range(model.cfg.n_layers)
    ]
    # Build hook list properly (avoid closure capture issues in loop)
    z_ctx: Dict[str, torch.Tensor] = {}
    z_mem: Dict[str, torch.Tensor] = {}

    def _save_ctx(value, hook):
        # Idempotent: if already saved (gradient-checkpointing recompute pass),
        # return the original leaf so backward flows to the same tensor.
        if hook.name in z_ctx:
            return z_ctx[hook.name]
        v = value.detach().requires_grad_(True)
        z_ctx[hook.name] = v
        return v

    def _save_mem(value, hook):
        if hook.name in z_mem:
            return z_mem[hook.name]
        v = value.detach().requires_grad_(True)
        z_mem[hook.name] = v
        return v

    fwd_hooks_ctx = [(f"blocks.{l}.attn.hook_z", _save_ctx) for l in range(model.cfg.n_layers)]
    fwd_hooks_mem = [(f"blocks.{l}.attn.hook_z", _save_mem) for l in range(model.cfg.n_layers)]

    # ── Pass 2: conflict + ctx answer → backward → grad_ctx ──────────────────
    with torch.enable_grad():
        ctx_logits = model.run_with_hooks(ctx_text, fwd_hooks=fwd_hooks_ctx)
        ctx_lp     = answer_log_prob(ctx_logits, ctx_ids)
        torch.tensor(ctx_lp).backward() if False else None  # keep for type-checker
        # ctx_lp is a Python float; we need to re-derive it as a tensor for backward
        log_probs_ctx = torch.log_softmax(ctx_logits[0, -L_ctx_ans - 1:-1, :], dim=-1)
        per_token_ctx = log_probs_ctx[
            torch.arange(L_ctx_ans, device=ctx_logits.device),
            ctx_ids.to(ctx_logits.device),
        ]
        ctx_lp_tensor = per_token_ctx.mean()
        ctx_lp_tensor.backward()

    # Restrict gradients to prompt positions before saving
    grad_ctx: Dict[str, torch.Tensor] = {}
    z_conflict_saved: Dict[str, torch.Tensor] = {}
    for name, z in z_ctx.items():
        if z.grad is None:
            return None
        grad_ctx[name]         = z.grad[:, :L_prompt, :, :].clone()
        z_conflict_saved[name] = z      [:, :L_prompt, :, :].detach().clone()

    # Free ctx forward-pass tensors before starting the mem pass.
    del z_ctx, ctx_logits
    torch.cuda.empty_cache()

    # ── Pass 3: conflict + mem answer → backward → grad_mem ──────────────────
    with torch.enable_grad():
        mem_logits = model.run_with_hooks(mem_text, fwd_hooks=fwd_hooks_mem)
        log_probs_mem = torch.log_softmax(mem_logits[0, -L_mem_ans - 1:-1, :], dim=-1)
        per_token_mem = log_probs_mem[
            torch.arange(L_mem_ans, device=mem_logits.device),
            mem_ids.to(mem_logits.device),
        ]
        mem_lp_tensor = per_token_mem.mean()
        mem_lp_tensor.backward()

    grad_mem: Dict[str, torch.Tensor] = {}
    for name, z in z_mem.items():
        if z.grad is None:
            return None
        grad_mem[name] = z.grad[:, :L_prompt, :, :].clone()

    del z_mem, mem_logits
    torch.cuda.empty_cache()

    # ── Gradient of the margin (both grads restricted to prompt positions) ────
    # ∂m/∂z = ∂(ctx_lp)/∂z − ∂(mem_lp)/∂z
    # Shapes are identical ([1, L_prompt, n_heads, d_head]) after the restriction.
    grad_margin: Dict[str, torch.Tensor] = {
        name: grad_ctx[name] - grad_mem[name] for name in grad_ctx
    }

    # ── Compute flavors ───────────────────────────────────────────────────────
    gradnorm: Dict[HeadKey, float] = {}
    gradact:  Dict[HeadKey, float] = {}
    eap:      Dict[HeadKey, float] = {}

    for name, grad in grad_margin.items():
        m = _HOOK_Z_PATTERN.match(name)
        if m is None:
            return None
        layer   = int(m.group(1))
        zc      = z_conflict_saved[name]   # [1, L_prompt, n_heads, d_head]
        zk      = z_clean[layer]           # [1, L_clean,  n_heads, d_head]
        n_heads = grad.shape[2]

        for h in range(n_heads):
            # Use last PROMPT position (not last token of the answer-appended run)
            g_final  = grad[0, L_prompt - 1, h, :]
            zc_final = zc  [0, L_prompt - 1, h, :]

            gradnorm[(layer, h)] = g_final.norm().item()
            gradact [(layer, h)] = (g_final * zc_final).sum().item()

            # EAP: aligned (prefix ∪ suffix, provably-identical-token) positions.
            # `grad`/`zc` are indexed by prompt position (0..L_prompt-1);
            # `zk` (counterfactual cache) is indexed by its own token positions,
            # aligned via aligned_cf_idx.
            g_aln  = grad[0, aligned_prompt_idx, h, :]
            zc_aln = zc  [0, aligned_prompt_idx, h, :]
            zk_aln = zk  [0, aligned_cf_idx, h, :]
            eap[(layer, h)] = (g_aln * (zc_aln - zk_aln)).sum().item()

    return {"gradnorm": gradnorm, "gradact": gradact, "eap": eap}


def _aggregate(per_prompt: List[Dict[HeadKey, float]]) -> HeadScores:
    if not per_prompt:
        return {}
    keys: set = set()
    for d in per_prompt:
        keys.update(d.keys())
    return {k: sum(d.get(k, 0.0) for d in per_prompt) / len(per_prompt) for k in keys}


def _run(
    model: HookedTransformer,
    prompts: List[ConflictPrompt],
    label: str,
) -> Dict[str, HeadScores]:
    acc = {"gradnorm": [], "gradact": [], "eap": []}
    n_fail = 0
    from tqdm import tqdm
    for p in tqdm(prompts, desc=f"[lds2] {label}", unit="prompt",
                  dynamic_ncols=True, ascii=True, file=sys.stdout):
        res = _score_single_prompt(model, p)
        if res is None:
            n_fail += 1
            continue
        for f in acc:
            acc[f].append(res[f])
    if n_fail:
        print(f"[lds2] {label}: {n_fail} prompts failed/skipped")
    return {
        "gradnorm": _aggregate(acc["gradnorm"]),
        "gradact":  _aggregate(acc["gradact"]),
        "eap":      _aggregate(acc["eap"]),
    }


def _shuffle_labels(prompts: List[ConflictPrompt], seed: int = 42) -> List[ConflictPrompt]:
    """Swap memory_answer/context_answer for ~half the prompts (noise-floor control)."""
    rng = random.Random(seed)
    out = []
    for p in prompts:
        if rng.random() > 0.5:
            out.append(ConflictPrompt(
                text=p.text, clean_text=p.clean_text, prompt_type=p.prompt_type,
                domain=p.domain, memory_aliases=p.memory_aliases,
                context_answer=p.memory_answer,   # swapped
                memory_answer=p.context_answer,   # swapped
            ))
        else:
            out.append(p)
    return out


def _save(scores: HeadScores, path: Path, provenance: Optional[Dict] = None) -> None:
    serial = {f"{l}_{h}": v for (l, h), v in scores.items()}
    serial = dict(sorted(serial.items(), key=lambda x: -x[1]))
    out = {"_provenance": provenance, "scores": serial} if provenance is not None else serial
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[lds2] saved -> {path}")


def run_and_save(
    model: HookedTransformer,
    prompts: List[ConflictPrompt],
    model_tag: str,
    prompt_type: str,
    set_tag: str,
    run_shuffle: bool = True,
    results_dir: Optional[Path] = None,
    alias_policy: str = "first_single_token",
) -> Optional[Dict[str, HeadScores]]:
    """
    results_dir: override the output directory (default: module-level
        RESULTS_DIR = "results/", used by run_lds.py's GPT-2 CLI, unchanged).
        The 3-model run scripts pass a per-family subfolder, e.g.
        "results/llama32_3b/".
    alias_policy: forwarded into each saved file's _provenance block --
        callers using A2.2 refinement should pass their real ALIAS_POLICY
        string so the label matches what actually ran.
    """
    out_dir = results_dir if results_dir is not None else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    main_eap = out_dir / f"lds2_{model_tag}_{prompt_type}_{set_tag}_eap.json"
    shuf_eap = out_dir / f"lds2_{model_tag}_{prompt_type}_{set_tag}_eap_shuffled.json"
    if main_eap.exists() and (not run_shuffle or shuf_eap.exists()):
        print(f"[lds2] skip {model_tag}/{prompt_type}/{set_tag} (already done)")
        return None
    print(f"\n[lds2] === {model_tag} / {prompt_type} / {set_tag} | n={len(prompts)} ===")
    scores = _run(model, prompts, label=f"{prompt_type}/{set_tag}/main")
    for flavor, s in scores.items():
        provenance = build_provenance(
            contrast="inplace_swap",
            dataset=f"single_token_{set_tag}" if set_tag != "full" else "full_multitoken",
            n_used=len(prompts), alias_policy=alias_policy,
            prompt_type=prompt_type, model_tag=model_tag,
        )
        _save(s, out_dir / f"lds2_{model_tag}_{prompt_type}_{set_tag}_{flavor}.json", provenance)

    if run_shuffle:
        shuf = _run(model, _shuffle_labels(prompts), label=f"{prompt_type}/{set_tag}/shuffled")
        for flavor, s in shuf.items():
            _save(s, out_dir / f"lds2_{model_tag}_{prompt_type}_{set_tag}_{flavor}_shuffled.json")
        def top(d, k=10):
            return {h for h, _ in sorted(d.items(), key=lambda kv: -abs(kv[1]))[:k]}
        ov = len(top(scores["eap"]) & top(shuf["eap"]))
        print(f"[lds2] eap noise floor: real-and-shuffled top10 overlap = {ov}/10 (chance ~0.7)")
    return scores


def _run_model(model_name: str, hf_model_name: str | None, model_tag: str) -> None:
    model = load_model(model_name, hf_model_name=hf_model_name)
    for prompt_type in ("substitution", "coherent"):
        prompts = load_conflict_prompts(prompt_type=prompt_type)

        if model_tag == "gpt2_base":
            run_and_save(model, prompts, model_tag, prompt_type, "full", run_shuffle=True)

        kept_path = RESULTS_DIR / f"pp_gpt2_{model_tag.replace('gpt2_', '')}_{prompt_type}_kept_indices.json"
        if not kept_path.exists():
            kept_path = RESULTS_DIR / f"pp_gpt2_{prompt_type}_kept_indices.json"
        if kept_path.exists():
            kept    = json.loads(kept_path.read_text())
            aligned = [prompts[i] for i in kept]
            run_and_save(model, aligned, model_tag, prompt_type, "aligned", run_shuffle=False)
        else:
            print(f"[lds2] no kept_indices for {model_tag}/{prompt_type}, skipping aligned set")


if __name__ == "__main__":
    _run_model("gpt2", None, "gpt2_base")
    _run_model("gpt2", "vicgalle/gpt2-open-instruct-v1", "gpt2_instruct")
    print("\n[lds2] Done. Run triangulation.py for Merge-A results.")
