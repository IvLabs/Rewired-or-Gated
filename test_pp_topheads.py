"""Tests for run_pp_topheads.py — select_top_heads and CLI smoke."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import pytest

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from run_pp_topheads import select_top_heads


# ---------------------------------------------------------------------------
# Unit tests for select_top_heads (pure function, no model)
# ---------------------------------------------------------------------------

class TestSelectTopHeads:
    def _scores(self, values: list) -> Dict[Tuple[int, int], float]:
        return {(0, i): v for i, v in enumerate(values)}

    def test_returns_empty_for_k_zero(self):
        scores = self._scores([1.0, -1.0, 2.0])
        assert select_top_heads(scores, 0) == []

    def test_returns_empty_for_empty_scores(self):
        assert select_top_heads({}, 5) == []

    def test_ctx_heads_are_most_positive(self):
        # scores: h0=3, h1=2, h2=-1, h3=-2
        scores = {(0, 0): 3.0, (0, 1): 2.0, (0, 2): -1.0, (0, 3): -2.0}
        result = select_top_heads(scores, k=1)
        # ctx head = (0,0); mem head = (0,3)
        assert (0, 0) in result
        assert (0, 3) in result

    def test_mem_heads_are_most_negative(self):
        scores = {(0, 0): 5.0, (0, 1): 3.0, (0, 2): -4.0, (0, 3): -6.0}
        result = select_top_heads(scores, k=1)
        assert (0, 3) in result  # most negative

    def test_no_duplicates(self):
        scores = {(0, 0): 5.0, (0, 1): -5.0}
        result = select_top_heads(scores, k=2)
        assert len(result) == len(set(result)), "Duplicates in result"

    def test_union_size_at_most_2k(self):
        scores = {(0, i): float(i - 5) for i in range(10)}
        result = select_top_heads(scores, k=3)
        assert len(result) <= 6

    def test_k_clamps_to_available(self):
        scores = {(0, 0): 1.0, (0, 1): -1.0}
        result = select_top_heads(scores, k=100)
        assert len(result) <= 2

    def test_all_positive_scores(self):
        scores = {(0, i): float(i + 1) for i in range(5)}
        result = select_top_heads(scores, k=2)
        # mem heads are least positive, still valid to select
        assert len(result) > 0

    def test_all_negative_scores(self):
        scores = {(0, i): float(-i - 1) for i in range(5)}
        result = select_top_heads(scores, k=2)
        assert len(result) > 0

    def test_ctx_heads_come_first_in_output(self):
        scores = {(0, 0): 5.0, (0, 1): 4.0, (0, 2): -4.0, (0, 3): -5.0}
        result = select_top_heads(scores, k=2)
        # First two should be ctx heads (most positive)
        assert result[0] in {(0, 0), (0, 1)}
        assert result[1] in {(0, 0), (0, 1)}

    def test_known_gpt2_base_substitution(self):
        """Top ctx head should be L10H0 from the known EAP results."""
        eap_path = Path(_ROOT) / "results" / "lds2_gpt2_base_substitution_full_eap.json"
        if not eap_path.exists():
            pytest.skip("EAP results not present; run run_lds.py first")
        raw = json.loads(eap_path.read_text(encoding="utf-8"))
        scores = {(int(k.split("_")[0]), int(k.split("_")[1])): float(v)
                  for k, v in raw.items()}
        result = select_top_heads(scores, k=10)
        assert len(result) <= 20
        # L10H0 is the top context-following head from Merge-A
        assert (10, 0) in result, f"Expected (10,0) in top heads; got {result[:5]}"


# ---------------------------------------------------------------------------
# Integration smoke (mock model via path_patching test helpers)
# ---------------------------------------------------------------------------

class TestPPTopHeadsIntegration:
    """Verify that run_pp_topheads.select_top_heads + run_path_patching(head_subset=...)
    produce a valid HeadScores dict restricted to the selected heads."""

    def test_pp_on_subset_has_correct_keys(self):
        sys.path.insert(0, os.path.join(_ROOT, "path_patching"))
        from path_patching import run_path_patching
        # Reuse the MockModel from the pp test module
        from test_pp import MockModel, _make_valid_prompt

        model = MockModel()
        prompts = [_make_valid_prompt(model, str(i)) for i in range(5)]

        # Build a tiny EAP score dict over the mock model's 2×2 heads
        eap: Dict[Tuple[int, int], float] = {
            (0, 0): 0.8,
            (0, 1): 0.3,
            (1, 0): -0.6,
            (1, 1): -0.1,
        }
        top_heads = select_top_heads(eap, k=1)  # expect [(0,0), (1,0)]
        assert (0, 0) in top_heads
        assert (1, 0) in top_heads

        scores = run_path_patching(model, prompts, head_subset=top_heads)
        assert set(scores.keys()) == set(top_heads)
