"""Regression test for the PP prompt-type scope.

As of 2026-07-10, Path Patching runs on BOTH substitution and coherent
prompt types across all three per-model run scripts. The original spec
Part H gate (substitution-only) was lifted once alignment_indices gained a
position-by-position diff for length-preserving swaps -- which recovers the
coherent middle span the old prefix∪suffix heuristic dropped, while
remaining byte-identical to the old alignment on substitution.

This test pins two things so the change can never silently regress:
  1. PP is wired to both prompt types (the intended new behavior).
  2. alignment_indices is still byte-identical to the old prefix∪suffix
     window on a single contiguous (substitution-style) swap -- the safety
     property that makes enabling coherent a non-regression for
     substitution PP/LDS-eap scores.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestPPScopeIncludesBothPromptTypes:
    def test_run_llama32_pp_scope(self):
        import run_llama32
        assert run_llama32.PP_PROMPT_TYPES == ("substitution", "coherent")
        assert "coherent" in run_llama32.PROMPT_TYPES

    def test_run_gemma3_pp_scope(self):
        import run_gemma3
        assert run_gemma3.PP_PROMPT_TYPES == ("substitution", "coherent")
        assert "coherent" in run_gemma3.PROMPT_TYPES

    def test_run_qwen25_pp_scope(self):
        import singleton_fix.run_qwen25 as run_qwen25
        assert run_qwen25.PP_PROMPT_TYPES == ("substitution", "coherent")
        assert "coherent" in run_qwen25.PROMPT_TYPES


class TestSubstitutionAlignmentUnchanged:
    """The coherent-enabling change must NOT alter substitution alignment --
    a single contiguous swap must still yield the exact prefix∪suffix window,
    so PP/LDS-eap substitution scores are byte-identical to before the fix."""

    def _old_prefix_suffix(self, a, b):
        i = 0
        while i < min(len(a), len(b)) and a[i] == b[i]:
            i += 1
        pl = i
        j = 0
        while j < min(len(a), len(b)) - pl and a[-1 - j] == b[-1 - j]:
            j += 1
        sl = j
        main_idx = list(range(pl)) + list(range(len(a) - sl, len(a)))
        cf_idx = list(range(pl)) + list(range(len(b) - sl, len(b)))
        return main_idx, cf_idx

    def test_single_contiguous_swap_matches_old_window(self):
        from singleton_fix.contrast_pair import alignment_indices
        # substitution: exactly one swapped token in the middle
        main = [10, 11, 12, 13, 99, 10, 11, 12, 13]
        cf = [10, 11, 12, 13, 55, 10, 11, 12, 13]
        new_main, new_cf = alignment_indices(main, cf)
        old_main, old_cf = self._old_prefix_suffix(main, cf)
        assert new_main.tolist() == old_main
        assert new_cf.tolist() == old_cf
