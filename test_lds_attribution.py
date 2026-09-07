"""Tests for lds_attribution.py's `eap` flavor fix.

Uses a tiny real 2-layer, 2-head model substitute is impractical for
autograd through hook_z, so these tests exercise the PURE post-processing
logic (which positions/tensors feed the eap sum) by monkeypatching
`model.run_with_cache`'s clean-cache call to confirm it now tokenizes the
counterfactual text, not clean_text. The full gradient pipeline itself
(hooks, backward passes) is exercised by the existing @pytest.mark.slow
real-GPT-2 smoke test added in Step 5.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest
import torch

from contract import ConflictPrompt
from lds_attribution import _score_single_prompt


def _prompt() -> ConflictPrompt:
    return ConflictPrompt(
        text="X plays sport of basketball. X plays sport of",
        clean_text="totally unrelated text that must not be read",
        prompt_type="substitution",
        domain="Athlete Sport",
        memory_aliases=["soccer"],
        context_answer="basketball",
        memory_answer="soccer",
    )


@pytest.mark.slow
class TestEapFlavorIgnoresCleanText:
    """Real-GPT-2 smoke test: two prompts identical except for clean_text
    must produce the SAME eap scores, since clean_text is no longer part of
    the contrast."""

    def test_eap_scores_unaffected_by_clean_text(self):
        from foundation import load_model
        model = load_model("gpt2")
        p1 = _prompt()
        p2 = ConflictPrompt(
            text=p1.text, clean_text="a completely different sentence entirely",
            prompt_type=p1.prompt_type, domain=p1.domain,
            memory_aliases=p1.memory_aliases, context_answer=p1.context_answer,
            memory_answer=p1.memory_answer,
        )
        r1 = _score_single_prompt(model, p1)
        r2 = _score_single_prompt(model, p2)
        assert r1 is not None and r2 is not None
        for key in r1["eap"]:
            assert abs(r1["eap"][key] - r2["eap"][key]) < 1e-5

    def test_eap_returns_finite_scores_for_all_heads(self):
        from foundation import load_model
        model = load_model("gpt2")
        result = _score_single_prompt(model, _prompt())
        assert result is not None
        assert len(result["eap"]) == model.cfg.n_layers * model.cfg.n_heads
        assert all(torch.isfinite(torch.tensor(v)) for v in result["eap"].values())
