"""Per-model single-token filter + A2 alias-selection policy.

"Single-token" is a property of a (word, tokenizer) pair -- a word that's one
token for Gemma may be two for Qwen/Llama. This module filters ParaConflict
prompts down to rows where BOTH the (possibly-corrected) memory answer and
the distractor tokenize to exactly one token under the GIVEN model's
tokenizer, and re-derives the canonical memory answer per spec §A2 instead
of blindly using `aliases[0]`.

Deliberately does NOT compute a cross-model intersection across the three
model families -- that tokenizer-intersection tooling is out of scope here
(spec's provenance notes it as a separate ~748-row artifact owned elsewhere).
This module only does the per-model filter each of the three run scripts
needs directly.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import List, Literal, Optional

import torch

from contract import ConflictPrompt
from foundation import load_conflict_prompts, _answer_ids, answer_log_prob

_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Spec §B4.3 — hold out a SMALL domain, not the dominant one. Athlete Sport
# (~71% of the single-token pool) and World Capital (~20 rows, too noisy) are
# both wrong choices; Company Headquarter (~88 rows in the intersection) is
# the recommended held-out domain for transfer testing.
RECOMMENDED_HELDOUT_DOMAIN = "Company Headquarter"


def _is_single_token(model, word: str) -> bool:
    """True if ' word' tokenizes to exactly one token under model's tokenizer."""
    return len(_answer_ids(model, word)) == 1


def _real_logprob_of(model, clean_text: str, answer: str) -> float:
    """Model's teacher-forced log-prob for `answer` immediately following
    `clean_text` -- one real forward pass. This is the A2.2 oracle: "how
    much does the model already want to say this word on the clean,
    no-conflict prompt" (independent of any injected distractor).
    """
    ids = _answer_ids(model, answer)
    text = clean_text + " " + answer.strip()
    with torch.no_grad():
        logits = model(text)
    return answer_log_prob(logits, ids)


def _attach_logprob_oracle(model) -> None:
    """Attach a `model.logprob_of(clean_text, answer) -> float` method if the
    model doesn't already have one (spec A2.2). Real HookedTransformer models
    reach this path; test doubles that already define their own
    `logprob_of` (e.g. to control the comparison without a real forward
    pass) are left alone.
    """
    if hasattr(model, "logprob_of"):
        return
    model.logprob_of = lambda clean_text, answer: _real_logprob_of(model, clean_text, answer)


def _select_memory_answer(
    model,
    aliases: List[str],
    use_logprob_refinement: bool,
    clean_text: Optional[str] = None,
) -> str:
    """Spec §A2.1 (+ §A2.2 refinement, decided ON for this pipeline).

    A2.1 -- deterministic default: the first alias in `aliases` that
    tokenizes to a single token under `model`'s tokenizer. Falls back to
    `aliases[0]` if none is single-token (the row then fails the downstream
    single-token filter, same as before).

    A2.2 -- within-family refinement: among the single-token aliases, pick
    the one with the highest model log-prob given `clean_text` (the model's
    actual preferred spelling of the fact). This matters because LDS/PP/
    EAP-IG read a single canonical token's log-prob -- if the chosen alias
    is one the model would rarely say even on the clean prompt (e.g.
    "football" when the model's native preference is "soccer"), the
    ctx-vs-mem margin these methods compute is biased by that surface-form
    mismatch, not by the injected conflict. Requires `clean_text`; falls
    back to A2.1's choice if `clean_text` is None or there's only one
    single-token candidate (no ambiguity to resolve).
    """
    single_token_aliases = [a for a in aliases if _is_single_token(model, a)]
    if not single_token_aliases:
        return aliases[0]

    if not use_logprob_refinement or clean_text is None or len(single_token_aliases) == 1:
        return single_token_aliases[0]

    _attach_logprob_oracle(model)
    return max(single_token_aliases, key=lambda a: model.logprob_of(clean_text, a))


def load_single_token_prompts(
    model,
    prompt_type: Literal["substitution", "coherent"] = "substitution",
    heldout_domain: Optional[str] = None,
    split: Literal["train", "heldout", "all"] = "all",
    use_logprob_refinement: bool = False,
) -> List[ConflictPrompt]:
    """Load ParaConflict, re-derive the canonical memory answer per §A2, and
    keep only rows where BOTH the memory answer and the distractor tokenize
    to a single token under `model`'s tokenizer.

    Logs how many rows A2.1's alias recovery adds over a naive
    aliases[0]-only filter, and the final survival count -- spec §A2.1's
    acceptance criterion.

    `use_logprob_refinement` defaults to False (A2.1-only, deterministic)
    since 2026-07-09: the A2.2 log-prob oracle runs independently per model
    and was found (via review) to pick different canonical aliases for base
    vs instruct on the same row (e.g. "football" vs "soccer"), confounding
    the base-vs-instruct comparison downstream methods rely on. A2.1's
    tokenizer-only choice is identical for base and instruct within a
    family by construction, so it carries no such risk. Pass
    `use_logprob_refinement=True` explicitly only for the standalone
    robustness check of whether results hold under each model's own
    preferred wording -- never as the canonical default.
    """
    all_prompts = load_conflict_prompts(
        prompt_type=prompt_type, heldout_domain=heldout_domain, split=split,
    )

    naive_survivors = 0
    kept: List[ConflictPrompt] = []
    for p in all_prompts:
        naive_ok = _is_single_token(model, p.memory_answer) and _is_single_token(model, p.context_answer)
        if naive_ok:
            naive_survivors += 1

        corrected_answer = _select_memory_answer(
            model, p.memory_aliases, use_logprob_refinement, clean_text=p.clean_text,
        )
        if not (_is_single_token(model, corrected_answer) and _is_single_token(model, p.context_answer)):
            continue

        kept.append(dataclasses.replace(p, memory_answer=corrected_answer))

    recovered = len(kept) - naive_survivors
    print(
        f"[single-token-loader] prompt_type={prompt_type} | kept {len(kept)}/{len(all_prompts)} "
        f"single-token rows (naive aliases[0]-only would keep {naive_survivors}; "
        f"alias-recovery added {recovered} rows)"
    )
    return kept


def load_cached_single_token_prompts(
    family: str,
    variant: str,
    prompt_type: Literal["substitution", "coherent"] = "substitution",
    heldout_domain: Optional[str] = None,
    split: Literal["train", "heldout", "all"] = "all",
) -> List[ConflictPrompt]:
    """Reconstruct the filtered ConflictPrompt list from a cached survival
    file written by singleton_fix/run_single_token_filter_all.py or
    singleton_fix/modal_single_token_filter.py -- NO model load, NO
    re-running the GPU-bound A2.2 oracle, and NO re-downloading ParaConflict.

    Use this instead of load_single_token_prompts() once a variant's
    survival file exists -- it guarantees the exact same prompt set (with
    the exact same A2-corrected memory_answer) that was verified and
    reported in the survival stats gets used for the real run, with no
    possibility of drift from re-deriving it.

    Raises FileNotFoundError with a clear message if the cache doesn't
    exist yet for this (family, variant) -- run the filter script first.
    """
    path = _RESULTS_DIR / family / f"single_token_survival_{family}_{variant}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No cached single-token survival file for {family}/{variant} at {path}. "
            f"Run 'singleton_fix/run_single_token_filter_all.py --families {family} "
            f"--variants {variant}' (or the Modal equivalent) first."
        )
    data = json.loads(path.read_text(encoding="utf-8"))

    out: List[ConflictPrompt] = []
    for row in data["rows"]:
        text = row["substitution_text"] if prompt_type == "substitution" else row["coherent_text"]
        out.append(ConflictPrompt(
            text=text,
            clean_text=row["clean_text"],
            prompt_type=prompt_type,
            domain=row["domain"],
            memory_aliases=row["memory_aliases"],
            context_answer=row["context_answer"],
            memory_answer=row["memory_answer"],
        ))

    if heldout_domain is not None and split != "all":
        if split == "train":
            out = [p for p in out if p.domain != heldout_domain]
        else:  # "heldout"
            out = [p for p in out if p.domain == heldout_domain]

    print(f"[cached-single-token-loader] {family}/{variant}/{prompt_type} | loaded {len(out)} rows from cache")
    return out


def print_domain_composition(prompts: List[ConflictPrompt]) -> None:
    """Spec §B4.1 -- print per-domain row counts so a sport-dominated pool is
    never silently presented as general knowledge-conflict."""
    counts: dict = {}
    for p in prompts:
        counts[p.domain] = counts.get(p.domain, 0) + 1
    total = len(prompts)
    print(f"[domain-composition] total={total}")
    for domain, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        pct = 100 * n / total if total else 0.0
        print(f"  {domain!r}: {n} ({pct:.1f}%)")
