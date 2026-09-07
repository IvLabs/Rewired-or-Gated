"""In-place counterfactual construction + provably-identical-token alignment.

Replaces the buggy `clean_text` contrast (a structurally different, shorter
prompt) and naive `min(len_a, len_b)` suffix alignment used by the original
Path Patching and LDS `eap` code. The fix: build the counterfactual by
swapping the injected distractor for the memory answer *inside* the conflict
text, so the two runs are identical except at the swapped word, then align by
provably-identical token positions (prefix ∪ suffix), not by length.

See docs/specs/2026-07-08-consolidated-interpretationB-and-validity-spec.md
Part A1 for the full rationale, and Part J for a worked example.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import torch

from contract import ConflictPrompt

TokenSeq = Union[Sequence[int], torch.Tensor]


def _as_list(tokens: TokenSeq) -> List[int]:
    if isinstance(tokens, torch.Tensor):
        return tokens.tolist()
    return list(tokens)


def _common_prefix_len(a: List[int], b: List[int]) -> int:
    """Length of the longest common prefix of two token-id sequences."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _common_suffix_len(a: List[int], b: List[int], prefix_len: int) -> int:
    """Length of the longest common suffix, not overlapping the already-counted prefix."""
    max_suffix = min(len(a), len(b)) - prefix_len
    i = 0
    while i < max_suffix and a[-1 - i] == b[-1 - i]:
        i += 1
    return i


def alignment_indices(
    main_tokens: TokenSeq,
    cf_tokens: TokenSeq,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Positions provably identical in BOTH sequences (prefix ∪ suffix).

    Args:
        main_tokens: token ids of the conflict prompt (`prompt.text`).
        cf_tokens: token ids of the in-place counterfactual
            (`build_counterfactual_text(prompt)`).

    Returns:
        (main_idx, cf_idx): equal-length 1-D LongTensors. Indexing
        `main_tokens[main_idx[i]]` and `cf_tokens[cf_idx[i]]` yields the same
        token id for every i. The window is generally non-contiguous when the
        swapped span changes token count (prefix block, then a gap for the
        swap, then a suffix block).

    Two regimes, both keeping ONLY provably-identical positions (never
    misaligns two positions holding different token ids):

    1. Length-preserving swap (len(main) == len(cf)) -- the common case for
       the single-token backbone, since swapping a single-token distractor
       for a single-token memory answer never changes sequence length. Here
       the two sequences correspond position-by-position, so EVERY index
       where the token id matches is safely alignable, including indices
       *between* scattered swaps. This recovers the whole middle that the
       prefix-suffix heuristic (regime 2) drops on coherent prompts, where
       the distractor repeats -- verified on all 3 real tokenizers
       (Llama/Gemma/Qwen): coherent swaps are length-preserving in 600/600
       sampled rows, keeping ~115/118 positions vs prefix-suffix's ~45.
       For substitution (a single contiguous swap) this yields the exact
       same index set as prefix-suffix, so Path Patching / LDS-eap scores on
       substitution are unchanged.

    2. Length-changing swap (len(main) != len(cf)) -- e.g. a multi-token
       answer, out of scope for the single-token backbone. Position-by-
       position diff is invalid (everything after the swap shifts), so fall
       back to the SAFE prefix-suffix window. This is lossy when the swap
       *also* repeats (spec Part G's LCS recovery would do better) but never
       incorrect.
    """
    a = _as_list(main_tokens)
    b = _as_list(cf_tokens)

    if len(a) == len(b):
        idx = [i for i in range(len(a)) if a[i] == b[i]]
        t = torch.tensor(idx, dtype=torch.long)
        return t, t

    prefix_len = _common_prefix_len(a, b)
    suffix_len = _common_suffix_len(a, b, prefix_len)

    main_idx = list(range(prefix_len)) + list(range(len(a) - suffix_len, len(a)))
    cf_idx = list(range(prefix_len)) + list(range(len(b) - suffix_len, len(b)))
    return (
        torch.tensor(main_idx, dtype=torch.long),
        torch.tensor(cf_idx, dtype=torch.long),
    )


def build_counterfactual_text(prompt: ConflictPrompt) -> str:
    """In-place swap: replace the injected distractor with the memory answer.

    e.g. "Lionel Messi plays the sport of basketball. Lionel Messi plays the
    sport of" -> "...soccer. Lionel Messi plays the sport of". Same subject,
    same structure — the two runs differ only at the swapped word(s), so any
    activation delta between them is attributable to that swap.

    Replaces ALL occurrences (matters for coherent prompts, where the
    distractor repeats — spec Part H); for substitution prompts there is
    normally exactly one occurrence.
    """
    return prompt.text.replace(
        prompt.context_answer.strip(), prompt.memory_answer.strip()
    )


def swap_occurred(prompt: ConflictPrompt, cf_text: str) -> bool:
    """True if build_counterfactual_text actually changed the text.

    `str.replace` is a silent no-op when `context_answer` isn't found
    verbatim in `prompt.text` (casing/surface-form mismatch, or a
    malformed row) -- without this check, `cf_text == prompt.text` and
    every downstream diff/patch for that row is zero, indistinguishable
    from a real "no effect" result. Callers should skip (and count) any
    prompt where this returns False rather than silently scoring it.
    """
    return cf_text != prompt.text


def build_contrast_pair(model, prompt: ConflictPrompt) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenize `prompt.text` and its in-place counterfactual with the model's tokenizer.

    Args:
        model: any object exposing `to_tokens(text) -> torch.Tensor[1, L]`
            (HookedTransformer, or a duck-typed test double).
        prompt: the ConflictPrompt to build the pair for.

    Returns:
        (main_tokens, cf_tokens): 1-D LongTensors of token ids, batch dim
        squeezed off. Feed these to `alignment_indices` to get the aligned
        index set for patching/diffing.
    """
    cf_text = build_counterfactual_text(prompt)
    main_tokens = model.to_tokens(prompt.text)[0]
    cf_tokens = model.to_tokens(cf_text)[0]
    return main_tokens, cf_tokens
