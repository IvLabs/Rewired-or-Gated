from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformer_lens import HookedTransformer

from contract import ConflictPrompt, HeadScores
from foundation import answer_logits, run_with_cache

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# only cache hook_z, no need to store everything
_HOOK_Z_FILTER = r"hook_z"
_HOOK_Z_PATTERN = re.compile(r"blocks\.(\d+)\.attn\.hook_z")


def _score_single_prompt(
    model: HookedTransformer,
    prompt: ConflictPrompt,
) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
    # first pass just to get logits — not used for grad, just a sanity check
    logits, cache = run_with_cache(model, prompt, names_filter=_HOOK_Z_FILTER)
    mem_logit, ctx_logit = answer_logits(logits, prompt)

    hook_z_tensors: Dict[str, torch.Tensor] = {}

    def _save_hook(value: torch.Tensor, hook):
        # detach then re-attach to grad graph so backward works
        value = value.detach().requires_grad_(True)
        hook_z_tensors[hook.name] = value
        return value

    fwd_hooks = [
        (f"blocks.{layer}.attn.hook_z", _save_hook)
        for layer in range(model.cfg.n_layers)
    ]

    # second pass — this one actually goes through the grad graph
    with torch.enable_grad():
        logits2 = model.run_with_hooks(prompt.text, fwd_hooks=fwd_hooks)
        mem_l2 = logits2[0, -1, prompt.memory_token_id]
        ctx_l2 = logits2[0, -1, prompt.context_token_id]
        margin2 = ctx_l2 - mem_l2
        margin2.backward()

    grad_scores: Dict[Tuple[int, int], float] = {}
    grad_act_scores: Dict[Tuple[int, int], float] = {}

    for hook_name, z_tensor in hook_z_tensors.items():
        m = _HOOK_Z_PATTERN.match(hook_name)
        if m is None:
            continue
        layer = int(m.group(1))

        if z_tensor.grad is None:
            # shouldn't happen but just in case
            n_heads = z_tensor.shape[2]
            for head in range(n_heads):
                grad_scores[(layer, head)] = 0.0
                grad_act_scores[(layer, head)] = 0.0
            continue

        grad = z_tensor.grad  # [1, seq, n_heads, d_head]
        n_heads = z_tensor.shape[2]

        for head in range(n_heads):
            g = grad[0, :, head, :]      # [seq, d_head]
            a = z_tensor[0, :, head, :]  # [seq, d_head]

            # flavor 1: plain gradient magnitude
            grad_scores[(layer, head)] = g.abs().mean().item()

            # flavor 2: gradient * activation — catches heads that move logits a lot
            grad_act_scores[(layer, head)] = (g * a).abs().mean().item()

    return grad_scores, grad_act_scores


def _aggregate(
    per_prompt_scores: List[Dict[Tuple[int, int], float]],
) -> HeadScores:
    # mean-absolute across prompts — signed mean not used because
    # context-memory margin is already directional, mean-abs is more stable
    if not per_prompt_scores:
        return {}

    all_keys = set()
    for d in per_prompt_scores:
        all_keys.update(d.keys())

    aggregated: HeadScores = {}
    for key in all_keys:
        values = [d.get(key, 0.0) for d in per_prompt_scores]
        aggregated[key] = sum(abs(v) for v in values) / len(values)

    return aggregated


def _run_clean_baseline(
    model: HookedTransformer,
    prompts: List[ConflictPrompt],
) -> Tuple[HeadScores, HeadScores]:
    # run LDS on clean prompts — heads that score high here are just
    # doing normal next-token prediction, not conflict resolution
    print("[lds] Running clean-prompt sanity baseline...")
    clean_prompts = [
        ConflictPrompt(
            text=p.clean_text,
            clean_text=p.clean_text,
            prompt_type=p.prompt_type,
            domain=p.domain,
            memory_aliases=p.memory_aliases,
            context_answer=p.context_answer,
            memory_token_id=p.memory_token_id,
            context_token_id=p.context_token_id,
        )
        for p in prompts
    ]
    return _run_prompt_list(model, clean_prompts, label="clean_baseline")


def _run_shuffled_baseline(
    model: HookedTransformer,
    prompts: List[ConflictPrompt],
    seed: int = 42,
) -> Tuple[HeadScores, HeadScores]:
    # swap memory/context token ids randomly — heads that still score high
    # after shuffling are probably just responding to token frequency, not conflict
    print("[lds] Running shuffled-label sanity baseline...")
    rng = random.Random(seed)
    shuffled: List[ConflictPrompt] = []
    for p in prompts:
        if rng.random() > 0.5:
            shuffled.append(ConflictPrompt(
                text=p.text,
                clean_text=p.clean_text,
                prompt_type=p.prompt_type,
                domain=p.domain,
                memory_aliases=p.memory_aliases,
                context_answer=p.context_answer,
                memory_token_id=p.context_token_id,  # swapped
                context_token_id=p.memory_token_id,  # swapped
            ))
        else:
            shuffled.append(p)
    return _run_prompt_list(model, shuffled, label="shuffled_baseline")


def _run_prompt_list(
    model: HookedTransformer,
    prompts: List[ConflictPrompt],
    label: str = "",
) -> Tuple[HeadScores, HeadScores]:
    all_grad: List[Dict[Tuple[int, int], float]] = []
    all_grad_act: List[Dict[Tuple[int, int], float]] = []

    for i, prompt in enumerate(prompts):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"[lds] {label} scoring prompt {i+1}/{len(prompts)}...")
        try:
            g, ga = _score_single_prompt(model, prompt)
            all_grad.append(g)
            all_grad_act.append(ga)
        except Exception as e:
            print(f"[lds] WARNING: prompt {i} failed ({e}), skipping.")

    return _aggregate(all_grad), _aggregate(all_grad_act)


def _scores_to_json_serializable(scores: HeadScores) -> Dict[str, float]:
    return {f"{layer}_{head}": score for (layer, head), score in scores.items()}


def _save_scores(scores: HeadScores, path: Path) -> None:
    serializable = _scores_to_json_serializable(scores)
    sorted_scores = dict(sorted(serializable.items(), key=lambda x: -x[1]))
    with open(path, "w") as f:
        json.dump(sorted_scores, f, indent=2)
    print(f"[lds] Saved → {path}")


def load_scores(path: Path) -> HeadScores:
    with open(path) as f:
        raw = json.load(f)
    return {
        (int(k.split("_")[0]), int(k.split("_")[1])): v
        for k, v in raw.items()
    }


def _print_top_k(scores: HeadScores, label: str = "", k: int = 10) -> None:
    top = sorted(scores.items(), key=lambda x: -x[1])[:k]
    print(f"\n[lds] Top-{k} heads by {label}:")
    for (layer, head), score in top:
        print(f"  L{layer:02d}H{head:02d}  {score:.6f}")


def _print_noise_floor_check(
    real: HeadScores,
    clean: HeadScores,
    shuffled: HeadScores,
    k: int = 10,
) -> None:
    top_real  = {k for k, _ in sorted(real.items(),     key=lambda x: -x[1])[:k]}
    top_clean = {k for k, _ in sorted(clean.items(),    key=lambda x: -x[1])[:k]}
    top_shuf  = {k for k, _ in sorted(shuffled.items(), key=lambda x: -x[1])[:k]}

    overlap_clean = len(top_real & top_clean)
    overlap_shuf  = len(top_real & top_shuf)

    print(f"\n[lds] Noise-floor check (top-{k} overlap):")
    print(f"  real ∩ clean    = {overlap_clean}/{k}  {'⚠ HIGH — check for noise' if overlap_clean > k//2 else '✓ OK'}")
    print(f"  real ∩ shuffled = {overlap_shuf}/{k}   {'⚠ HIGH — labels may be flipped' if overlap_shuf > k//2 else '✓ OK'}")


def run_lds(
    model: HookedTransformer,
    prompts: List[ConflictPrompt],
    model_tag: str = "gpt2",
    run_sanity: bool = True,
    save: bool = True,
) -> Tuple[HeadScores, HeadScores]:
    prompt_type = prompts[0].prompt_type if prompts else "unknown"
    print(f"\n[lds] Starting LDS | model={model_tag} | prompt_type={prompt_type} | n={len(prompts)}")

    grad_scores, grad_act_scores = _run_prompt_list(model, prompts, label="main")

    if save:
        _save_scores(grad_scores,     RESULTS_DIR / f"lds_{model_tag}_{prompt_type}_grad.json")
        _save_scores(grad_act_scores, RESULTS_DIR / f"lds_{model_tag}_{prompt_type}_grad_act.json")

    if run_sanity:
        clean_grad, clean_grad_act = _run_clean_baseline(model, prompts)
        shuf_grad, shuf_grad_act   = _run_shuffled_baseline(model, prompts)

        if save:
            _save_scores(clean_grad,     RESULTS_DIR / f"lds_{model_tag}_{prompt_type}_sanity_clean_grad.json")
            _save_scores(clean_grad_act, RESULTS_DIR / f"lds_{model_tag}_{prompt_type}_sanity_clean_grad_act.json")
            _save_scores(shuf_grad,      RESULTS_DIR / f"lds_{model_tag}_{prompt_type}_sanity_shuffled_grad.json")
            _save_scores(shuf_grad_act,  RESULTS_DIR / f"lds_{model_tag}_{prompt_type}_sanity_shuffled_grad_act.json")

        _print_noise_floor_check(grad_scores, clean_grad, shuf_grad, k=10)

    _print_top_k(grad_scores,     label=f"grad [{model_tag}]",     k=10)
    _print_top_k(grad_act_scores, label=f"grad_act [{model_tag}]", k=10)

    return grad_scores, grad_act_scores


if __name__ == "__main__":
    from foundation import load_conflict_prompts, load_model

    print("Loading model...")
    model = load_model("gpt2")

    print("Loading prompts...")
    prompts = load_conflict_prompts("coherent") 

    grad_scores, grad_act_scores = run_lds(
    model, prompts,
    model_tag="gpt2_base_coherent",
        run_sanity=True,
        save=True,
    )

    print("\nDone. Check results/ directory for JSON outputs.")