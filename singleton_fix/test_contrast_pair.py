"""Tests for contrast_pair.py's alignment primitives.

Pure-function unit tests — no model, no torch device dependency beyond
CPU tensors. See docs/specs/2026-07-08-consolidated-interpretationB-and-validity-spec.md
Part A1 for the alignment contract these implement.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from singleton_fix.contrast_pair import (
    _common_prefix_len,
    _common_suffix_len,
    alignment_indices,
)


class TestCommonPrefixLen:
    def test_identical_sequences(self):
        assert _common_prefix_len([1, 2, 3], [1, 2, 3]) == 3

    def test_no_overlap(self):
        assert _common_prefix_len([1, 2, 3], [9, 8, 7]) == 0

    def test_partial_prefix(self):
        assert _common_prefix_len([1, 2, 3, 4], [1, 2, 9, 9]) == 2

    def test_different_lengths_bounded_by_shorter(self):
        assert _common_prefix_len([1, 2, 3], [1, 2, 3, 4, 5]) == 3


class TestCommonSuffixLen:
    def test_identical_sequences_prefix_already_consumed(self):
        # prefix_len=3 (whole thing) leaves nothing for the suffix to double-count
        assert _common_suffix_len([1, 2, 3], [1, 2, 3], prefix_len=3) == 0

    def test_partial_suffix(self):
        assert _common_suffix_len([1, 2, 3, 4], [9, 9, 3, 4], prefix_len=0) == 2

    def test_suffix_does_not_overlap_prefix(self):
        # a = [1,2,1,2], b = [1,2,9,2]; prefix=2 ("1,2"); remaining tail is [1,2] vs [9,2]
        # only position -1 matches (2==2); -2 doesn't (1 != 9) -> suffix = 1
        assert _common_suffix_len([1, 2, 1, 2], [1, 2, 9, 2], prefix_len=2) == 1

    def test_no_suffix_overlap_when_prefix_consumes_everything(self):
        assert _common_suffix_len([1, 2], [1, 2], prefix_len=2) == 0


class TestAlignmentIndices:
    def test_single_middle_swap_aligns_everything_else(self):
        # "Lionel Messi plays the sport of basketball. Lionel Messi plays the sport of"
        # vs the in-place swap to "soccer" -- only the swapped token id differs.
        main = [10, 11, 12, 13, 99, 10, 11, 12, 13]   # 99 = basketball
        cf   = [10, 11, 12, 13, 55, 10, 11, 12, 13]   # 55 = soccer
        main_idx, cf_idx = alignment_indices(main, cf)
        assert isinstance(main_idx, torch.Tensor) and main_idx.dtype == torch.long
        assert isinstance(cf_idx, torch.Tensor) and cf_idx.dtype == torch.long
        assert main_idx.tolist() == [0, 1, 2, 3, 5, 6, 7, 8]
        assert cf_idx.tolist() == [0, 1, 2, 3, 5, 6, 7, 8]
        # the swapped position (index 4) must be excluded from both
        assert 4 not in main_idx.tolist()

    def test_identical_sequences_align_fully(self):
        seq = [1, 2, 3, 4, 5]
        main_idx, cf_idx = alignment_indices(seq, seq)
        assert main_idx.tolist() == [0, 1, 2, 3, 4]
        assert cf_idx.tolist() == [0, 1, 2, 3, 4]

    def test_completely_different_sequences_align_to_nothing(self):
        main_idx, cf_idx = alignment_indices([1, 2, 3], [9, 8, 7])
        assert main_idx.tolist() == []
        assert cf_idx.tolist() == []

    def test_accepts_tensor_input(self):
        main = torch.tensor([1, 2, 3])
        cf = torch.tensor([1, 2, 3])
        main_idx, cf_idx = alignment_indices(main, cf)
        assert main_idx.tolist() == [0, 1, 2]

    def test_length_changing_swap_still_aligns_ends(self):
        # distractor and memory answer are different token counts (e.g. multi-token
        # in the middle) -- prefix/suffix still align around it, middle is excluded.
        main = [10, 11, 99, 99, 12, 13]   # 2-token distractor
        cf   = [10, 11, 55, 12, 13]        # 1-token memory answer
        main_idx, cf_idx = alignment_indices(main, cf)
        # prefix = [10, 11] (len 2), suffix = [12, 13] (len 2)
        assert main_idx.tolist() == [0, 1, 4, 5]
        assert cf_idx.tolist() == [0, 1, 3, 4]

    def test_scattered_length_preserving_swaps_keep_the_middle(self):
        # Coherent-style prompt: the distractor (99) repeats at 3 SCATTERED
        # positions (2, 6, 9), each swapped to a single-token memory answer
        # (55) -- so the two sequences are the SAME length and differ at
        # exactly those 3 positions. The prefix-suffix heuristic would keep
        # only [0,1] (prefix, up to first diff) + [10] (suffix, after last
        # diff) = 3 positions, discarding positions 3,4,5,7,8 which are
        # IDENTICAL between the two runs. Positional-diff alignment must
        # recover all 8 identical positions -- this is what unlocks Path
        # Patching / LDS-eap on coherent prompts (verified on all 3 real
        # tokenizers: coherent swaps are length-preserving 600/600 rows).
        main = [10, 11, 99, 12, 13, 14, 99, 15, 16, 99, 17]
        cf   = [10, 11, 55, 12, 13, 14, 55, 15, 16, 55, 17]
        assert len(main) == len(cf)  # length-preserving swap
        main_idx, cf_idx = alignment_indices(main, cf)
        expected = [0, 1, 3, 4, 5, 7, 8, 10]  # everything except the 3 swapped positions
        assert main_idx.tolist() == expected
        assert cf_idx.tolist() == expected
        # the 3 swapped positions must be excluded from both
        assert all(p not in main_idx.tolist() for p in (2, 6, 9))


from singleton_fix.contrast_pair import build_counterfactual_text, build_contrast_pair, swap_occurred
from contract import ConflictPrompt


class _FakeTokenizerModel:
    """Duck-typed model exposing only to_tokens, word-level 'tokenization'
    (splits on whitespace, maps each distinct word to a stable id) so tests
    can reason about token positions without a real BPE tokenizer."""

    def __init__(self):
        self._vocab = {}

    def _id_for(self, word: str) -> int:
        return self._vocab.setdefault(word, len(self._vocab) + 1)

    def to_tokens(self, text: str) -> torch.Tensor:
        ids = [self._id_for(w) for w in text.split(" ")]
        return torch.tensor([ids], dtype=torch.long)


def _messi_prompt() -> ConflictPrompt:
    return ConflictPrompt(
        text="Lionel Messi plays the sport of basketball. Lionel Messi plays the sport of",
        clean_text="Lionel Messi plays the sport of",
        prompt_type="substitution",
        domain="Athlete Sport",
        memory_aliases=["soccer", "football"],
        context_answer="basketball",
        memory_answer="soccer",
    )


class TestBuildCounterfactualText:
    def test_swaps_distractor_for_memory_answer(self):
        p = _messi_prompt()
        cf = build_counterfactual_text(p)
        assert "basketball" not in cf
        assert cf.count("soccer") == 1
        assert cf == (
            "Lionel Messi plays the sport of soccer. "
            "Lionel Messi plays the sport of"
        )

    def test_replaces_all_occurrences_for_coherent_style_repeats(self):
        p = ConflictPrompt(
            text="X likes basketball. X plays basketball daily. X's sport is",
            clean_text="X's sport is",
            prompt_type="coherent",
            domain="Athlete Sport",
            memory_aliases=["soccer"],
            context_answer="basketball",
            memory_answer="soccer",
        )
        cf = build_counterfactual_text(p)
        assert "basketball" not in cf
        assert cf.count("soccer") == 2


class TestSwapOccurred:
    def test_true_when_distractor_present(self):
        p = _messi_prompt()
        cf = build_counterfactual_text(p)
        assert swap_occurred(p, cf) is True

    def test_false_when_distractor_absent_from_text(self):
        p = ConflictPrompt(
            text="Lionel Messi plays the sport of football.",  # no "basketball" substring
            clean_text="Lionel Messi plays the sport of",
            prompt_type="substitution",
            domain="Athlete Sport",
            memory_aliases=["soccer"],
            context_answer="basketball",
            memory_answer="soccer",
        )
        cf = build_counterfactual_text(p)
        assert cf == p.text  # naive .replace() was a silent no-op
        assert swap_occurred(p, cf) is False


class TestBuildContrastPair:
    def test_returns_1d_tensors_differing_only_at_swap(self):
        model = _FakeTokenizerModel()
        p = _messi_prompt()
        main_tokens, cf_tokens = build_contrast_pair(model, p)
        assert main_tokens.ndim == 1 and cf_tokens.ndim == 1
        assert main_tokens.shape[0] == cf_tokens.shape[0]  # same word count here
        diff_positions = [
            i for i in range(main_tokens.shape[0])
            if main_tokens[i].item() != cf_tokens[i].item()
        ]
        assert diff_positions == [6]  # "basketball" -> "soccer" is the 7th word (index 6)

    def test_alignment_indices_excludes_only_the_swap(self):
        from singleton_fix.contrast_pair import alignment_indices
        model = _FakeTokenizerModel()
        p = _messi_prompt()
        main_tokens, cf_tokens = build_contrast_pair(model, p)
        main_idx, cf_idx = alignment_indices(main_tokens, cf_tokens)
        assert 6 not in main_idx.tolist()
        assert main_idx.tolist() == [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
