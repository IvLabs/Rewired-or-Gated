"""Tests for foundation.py's multi-token-aware CRR."""
from __future__ import annotations

import pytest
import torch

from foundation import compute_crr, _matches_answer
from contract import ConflictPrompt


def test_matches_answer_word_boundary():
    """Unit test for _matches_answer: verify word boundary checking.

    The function must not match a strict prefix when it's followed by
    alphanumeric characters (mid-word continuation). This prevents false
    positives like 'football' matching 'footballers'.
    """
    # Exact match: should match
    assert _matches_answer("football", "football") is True
    assert _matches_answer("football is great", "football") is True

    # Prefix match with trailing space: should match
    assert _matches_answer("football players", "football") is True

    # Prefix match with trailing punctuation: should match
    assert _matches_answer("football!", "football") is True
    assert _matches_answer("football, the sport", "football") is True

    # Mid-word continuation (alnum after prefix): should NOT match
    assert _matches_answer("footballers is great", "football") is False
    assert _matches_answer("footballing is fun", "football") is False
    assert _matches_answer("football123", "football") is False

    # No match at start: should NOT match
    assert _matches_answer("soccer is fun", "football") is False


@pytest.fixture(scope="module")
def gpt2():
    from foundation import load_model
    return load_model("gpt2")


@pytest.mark.slow
def test_crr_matches_multi_token_context_answer(gpt2):
    """A prompt whose context_answer is multiple GPT-2 tokens must still be
    classified as context-following when the model actually says it --
    the old single-next-token check could never match this."""
    p = ConflictPrompt(
        text="Dak Prescott plays the sport of American football. "
             "Dak Prescott plays the sport of",
        clean_text="Dak Prescott plays the sport of",
        prompt_type="substitution", domain="Athelete Sport",
        memory_aliases=["football"], context_answer="American football",
        memory_answer="football",
    )
    crr = compute_crr(gpt2, [p])
    # Not asserting a specific direction (that depends on GPT-2's actual
    # behavior) -- asserting the function runs and returns a valid fraction,
    # which the old single-token argmax version could still do, so the real
    # regression check is test_crr_never_misses_multi_token_match below.
    assert 0.0 <= crr <= 1.0


@pytest.mark.slow
def test_crr_never_misses_multi_token_match(gpt2):
    """Construct a prompt where GPT-2 is essentially certain to continue
    with the multi-token context answer verbatim, and confirm CRR counts it
    as context-following (not 'neither', which is what the old single-token
    argmax check would silently do for any 2+ token answer)."""
    p = ConflictPrompt(
        text="Repeat this exact phrase: New York City. Repeat this exact phrase:",
        clean_text="Repeat this exact phrase:",
        prompt_type="substitution", domain="test",
        memory_aliases=["Los Angeles"], context_answer="New York City",
        memory_answer="Los Angeles",
    )
    crr = compute_crr(gpt2, [p])
    assert crr == 1.0, (
        "expected the multi-token context answer 'New York City' to be "
        "detected as context-following; got crr={crr} -- if this is 0.0, "
        "compute_crr is still only checking a single next token"
    )


def test_degenerate_filter_checks_all_aliases(monkeypatch):
    """A distractor equal to a NON-first memory alias must still be dropped
    -- checking only aliases[0] lets a true synonym slip through as a fake
    'conflict'."""
    import foundation

    fake_row = {
        "Category": "Athlete Sport",
        "Answer": ["football", "soccer"],
        "Distracted Token": "soccer",  # matches aliases[1], not aliases[0]
        "Clean Prompt": "X plays the sport of",
        "Substitution Conflict": "X plays the sport of soccer. X plays the sport of",
        "Coherent Conflict": "X plays the sport of soccer. X plays the sport of",
    }

    class _FakeDataset:
        def __iter__(self):
            return iter([fake_row])
        def __len__(self):
            return 1

    monkeypatch.setattr(foundation, "load_dataset", lambda *a, **k: _FakeDataset())
    prompts = foundation.load_conflict_prompts(prompt_type="substitution")
    assert len(prompts) == 0, (
        "distractor 'soccer' equals memory_aliases[1] -- this row is not a "
        "real conflict and must be dropped, but the old check only compared "
        "against aliases[0] ('football')"
    )


def test_empty_answer_string_is_dropped(monkeypatch):
    """aliases=[''] is truthy as a list but the canonical answer is empty --
    must be dropped, not silently produce memory_answer=''."""
    import foundation

    fake_row = {
        "Category": "Athlete Sport",
        "Answer": [""],
        "Distracted Token": "basketball",
        "Clean Prompt": "X plays the sport of",
        "Substitution Conflict": "X plays the sport of basketball. X plays the sport of",
        "Coherent Conflict": "X plays the sport of basketball. X plays the sport of",
    }

    class _FakeDataset:
        def __iter__(self):
            return iter([fake_row])
        def __len__(self):
            return 1

    monkeypatch.setattr(foundation, "load_dataset", lambda *a, **k: _FakeDataset())
    prompts = foundation.load_conflict_prompts(prompt_type="substitution")
    assert len(prompts) == 0


def test_empty_non_canonical_alias_is_stripped_from_memory_aliases(monkeypatch):
    """aliases=["football", ""] has a non-empty canonical answer (aliases[0]),
    so the row must NOT be dropped -- but the empty string must never end up
    in memory_aliases, since compute_crr's word-boundary matcher would treat
    "" as a match for any generation starting with punctuation/whitespace."""
    import foundation

    fake_row = {
        "Category": "Athlete Sport",
        "Answer": ["football", ""],
        "Distracted Token": "basketball",
        "Clean Prompt": "X plays the sport of",
        "Substitution Conflict": "X plays the sport of basketball. X plays the sport of",
        "Coherent Conflict": "X plays the sport of basketball. X plays the sport of",
    }

    class _FakeDataset:
        def __iter__(self):
            return iter([fake_row])
        def __len__(self):
            return 1

    monkeypatch.setattr(foundation, "load_dataset", lambda *a, **k: _FakeDataset())
    prompts = foundation.load_conflict_prompts(prompt_type="substitution")
    assert len(prompts) == 1
    assert prompts[0].memory_aliases == ["football"]
    assert "" not in prompts[0].memory_aliases
    assert prompts[0].memory_answer == "football"


def test_answer_log_prob_raises_on_empty_answer_ids():
    """answer_log_prob must raise a clear ValueError on empty answer_ids,
    not crash inside .mean() or silently return NaN."""
    from foundation import answer_log_prob

    logits = torch.randn(1, 5, 100)
    empty_ids = torch.tensor([], dtype=torch.long)
    with pytest.raises(ValueError, match="empty"):
        answer_log_prob(logits, empty_ids)


def test_load_model_survives_bad_hf_token(monkeypatch):
    """A stale/invalid HF_TOKEN must not block loading a PUBLIC model like
    plain gpt2 -- huggingface_hub.login() failing should warn, not crash."""
    import foundation

    monkeypatch.setenv("HF_TOKEN", "not-a-real-token-12345")

    def _boom(*a, **k):
        raise Exception("simulated invalid token")

    # load_model does `import huggingface_hub as _hfh` lazily inside the
    # function body; patching the attribute on the real module object still
    # affects that lazy import, since Python caches and reuses the same
    # module object across every import of it.
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "login", _boom)

    model = foundation.load_model("gpt2")  # must NOT raise
    assert model is not None
