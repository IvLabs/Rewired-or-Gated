"""End-to-end backbone smoke test: a <=5-prompt run
of the single-token loader -> LDS (gradnorm/gradact/eap) -> Path Patching on
one real model (GPT-2, for speed), confirming no shape assertions fire and
every stage produces provenance-tagged, finite output. NOT a scientific
result -- just the "does the wiring hold together" gate.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


@pytest.mark.slow
class TestBackboneSmoke:
    def test_single_token_loader_produces_rows_for_gpt2(self):
        from foundation import load_model
        from singleton_fix.single_token_loader import load_single_token_prompts

        model = load_model("gpt2")
        prompts = load_single_token_prompts(model, prompt_type="substitution")
        assert len(prompts) > 0
        for p in prompts[:5]:
            assert p.memory_answer
            assert p.context_answer

    def test_lds_eap_runs_on_five_single_token_prompts(self):
        from foundation import load_model
        from singleton_fix.single_token_loader import load_single_token_prompts
        from lds_attribution import _score_single_prompt

        model = load_model("gpt2")
        prompts = load_single_token_prompts(model, prompt_type="substitution")[:5]
        assert len(prompts) > 0
        for p in prompts:
            result = _score_single_prompt(model, p)
            assert result is not None
            assert len(result["eap"]) == model.cfg.n_layers * model.cfg.n_heads

    def test_path_patching_runs_on_five_single_token_prompts(self):
        from foundation import load_model
        from singleton_fix.single_token_loader import load_single_token_prompts
        from path_patching import run_path_patching

        model = load_model("gpt2")
        prompts = load_single_token_prompts(model, prompt_type="substitution")[:5]
        meta = {}
        scores = run_path_patching(model, prompts, head_subset=[(0, 0), (1, 1), (5, 5)], _meta_out=meta)
        assert len(scores) == 3
        assert meta["noise_floor"]["identity_patch_max_abs"] < 1e-4

    def test_behavioral_crr_runs_on_five_single_token_prompts(self):
        from foundation import load_model, compute_crr
        from singleton_fix.single_token_loader import load_single_token_prompts

        model = load_model("gpt2")
        prompts = load_single_token_prompts(model, prompt_type="substitution")[:5]
        result = compute_crr(model, prompts, max_new_tokens=5)
        assert result["total"] == len(prompts)
        assert result["context"] + result["memory"] + result["neither"] == result["total"]
