"""Tests for the behavioral CRR (spec §B1). Uses a fake model exposing
generate() so tests don't need a real forward pass."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest
import torch

from contract import ConflictPrompt
from foundation import compute_crr, compute_crr_logprob


class _FakeGenModel:
    """to_tokens(text) -> a 1-token-per-word LongTensor (BOS included as an
    extra leading token, mirroring TransformerLens convention).
    generate(tokens, max_new_tokens, ...) appends `self._continuation` token
    ids, ignoring the actual prompt content -- tests control the "generated"
    text entirely via `self._continuation`."""

    class tokenizer:
        @staticmethod
        def decode(ids) -> str:
            return " ".join(f"tok{i}" for i in ids)

    def __init__(self, continuation_text: str):
        self._continuation_text = continuation_text

    def to_tokens(self, text: str, prepend_bos: bool = True) -> torch.Tensor:
        n = len(text.split(" ")) + (1 if prepend_bos else 0)
        return torch.zeros(1, n, dtype=torch.long)

    def generate(self, tokens: torch.Tensor, max_new_tokens: int, **kwargs) -> torch.Tensor:
        # Simulate appended generated tokens by returning a longer tensor;
        # decoding is monkeypatched at the call site via to_string, see below.
        extra = torch.zeros(1, max_new_tokens, dtype=torch.long)
        return torch.cat([tokens, extra], dim=1)

    def to_string(self, tokens: torch.Tensor) -> str:
        return self._continuation_text


def _prompt() -> ConflictPrompt:
    return ConflictPrompt(
        text="X plays sport of basketball. X plays sport of",
        clean_text="X plays sport of",
        prompt_type="substitution",
        domain="Athlete Sport",
        memory_aliases=["soccer", "football"],
        context_answer="basketball",
        memory_answer="soccer",
    )


class TestBehavioralCrr:
    def test_context_following_generation_counted_as_context(self):
        model = _FakeGenModel(continuation_text=" basketball, obviously.")
        result = compute_crr(model, [_prompt()])
        assert result["context"] == 1
        assert result["memory"] == 0
        assert result["neither"] == 0
        assert result["crr"] == 1.0

    def test_memory_following_generation_matches_any_alias(self):
        model = _FakeGenModel(continuation_text=" football is the answer")
        result = compute_crr(model, [_prompt()])
        assert result["memory"] == 1
        assert result["context"] == 0

    def test_neither_bucket_is_reachable(self):
        model = _FakeGenModel(continuation_text=" tennis, actually")
        result = compute_crr(model, [_prompt()])
        assert result["neither"] == 1
        assert result["context"] == 0
        assert result["memory"] == 0

    def test_total_matches_prompt_count(self):
        model = _FakeGenModel(continuation_text=" basketball")
        result = compute_crr(model, [_prompt(), _prompt(), _prompt()])
        assert result["total"] == 3
        assert result["context"] + result["memory"] + result["neither"] == 3

    def test_empty_prompts_returns_zeroed_result(self):
        model = _FakeGenModel(continuation_text="")
        result = compute_crr(model, [])
        assert result["total"] == 0
        assert result["crr"] == 0.0


class TestLogprobCrrStillWorks:
    """compute_crr_logprob is the renamed old proxy -- confirm the rename
    didn't change behavior, using the same real-model smoke path the old
    compute_crr test would have used."""

    @pytest.mark.slow
    def test_logprob_crr_runs_on_real_gpt2(self):
        from foundation import load_model
        model = load_model("gpt2")
        crr = compute_crr_logprob(model, [_prompt()])
        assert 0.0 <= crr <= 1.0
