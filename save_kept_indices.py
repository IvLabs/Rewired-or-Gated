"""Filter-only pass to save kept prompt indices.

Runs the same filter logic as run_path_patching (length, degenerate-margin)
but without any patching. Saves results/pp_gpt2_{prompt_type}_kept_indices.json
for reuse by an instruct-model run.
"""
import json, os, sys, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from foundation import load_model, load_conflict_prompts, _answer_ids, answer_log_prob
from path_patching.path_patching import _token_length

EPSILON   = 0.05   # log-prob units (was 0.5 logit units)
L_MAX_DIFF = 200
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

print("[filter] Loading base GPT-2 ...")
model = load_model("gpt2")

for prompt_type in ["substitution", "coherent"]:
    prompts = load_conflict_prompts(prompt_type=prompt_type)
    kept = []

    print(f"\n[filter] Filtering {len(prompts)} {prompt_type} prompts ...")
    n_skip_len, n_skip_margin = 0, 0

    for i, p in enumerate(prompts):
        L_c = _token_length(model, p.text)
        L_k = _token_length(model, p.clean_text)

        if abs(L_c - L_k) > L_MAX_DIFF:
            n_skip_len += 1
            continue

        ctx_ids = _answer_ids(model, p.context_answer)
        mem_ids = _answer_ids(model, p.memory_answer)

        with torch.no_grad():
            ctx_lp_c = answer_log_prob(model(p.text       + " " + p.context_answer.strip()), ctx_ids)
            mem_lp_c = answer_log_prob(model(p.text       + " " + p.memory_answer.strip()),  mem_ids)
            ctx_lp_k = answer_log_prob(model(p.clean_text + " " + p.context_answer.strip()), ctx_ids)
            mem_lp_k = answer_log_prob(model(p.clean_text + " " + p.memory_answer.strip()),  mem_ids)

        denom = (ctx_lp_c - mem_lp_c) - (ctx_lp_k - mem_lp_k)
        if abs(denom) < EPSILON or denom <= 0:
            n_skip_margin += 1
            continue

        kept.append(i)

    out = os.path.join(RESULTS_DIR, f"pp_gpt2_{prompt_type}_kept_indices.json")
    with open(out, "w") as f:
        json.dump(kept, f)

    print(f"[filter] {prompt_type}: kept {len(kept)}/{len(prompts)} "
          f"(skipped: len={n_skip_len}, margin={n_skip_margin})")
    print(f"[filter] Saved -> {out}")

print("\n[filter] Done.")
