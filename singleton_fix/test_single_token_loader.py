"""Tests for single_token_loader.py -- the per-model single-token filter and
A2 alias-selection policy. Uses a fake model exposing to_tokens() with a
tiny hand-built vocabulary so single- vs multi-token words are controllable,
and (for the logprob refinement) a fake log-prob oracle.
"""
from __future__ import annotations

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from contract import ConflictPrompt
from singleton_fix.single_token_loader import (
    _select_memory_answer,
    load_single_token_prompts,
)


class _FakeForwardModel:
    """Real-forward-pass-shaped fake (no native logprob_of): to_tokens gives
    1 token for known single-token words (2 otherwise); __call__ returns
    logits where the position before the last token carries a controllable
    raw logit for whichever known word the text ends with, so log-prob
    ordering between candidate aliases is deterministic. Exercises
    _attach_logprob_oracle's real-forward-pass code path (spec A2.2)."""

    def __init__(self, single_token_words, raw_logit_by_word=None):
        self._single = {w.lower(): i + 1 for i, w in enumerate(single_token_words)}
        self._raw = {w.lower(): v for w, v in (raw_logit_by_word or {}).items()}

    def to_tokens(self, text: str, prepend_bos: bool = False) -> torch.Tensor:
        word = text.strip().lower()
        if word in self._single:
            return torch.tensor([[self._single[word]]], dtype=torch.long)
        return torch.zeros(1, 2, dtype=torch.long)

    def __call__(self, text: str, **kwargs) -> torch.Tensor:
        vocab = 50
        stripped = text.strip().lower()
        for word, raw in self._raw.items():
            if stripped.endswith(word):
                logits = torch.zeros(1, 2, vocab)
                logits[0, -2, self._single[word]] = raw
                return logits
        return torch.zeros(1, 2, vocab)


class _FakeModel:
    """to_tokens(' word') returns 1 token for words in _single_token_words,
    2 tokens otherwise. Also supports a log-prob oracle for A2.2."""

    def __init__(self, single_token_words, logprob_table=None):
        self._single = set(w.lower() for w in single_token_words)
        self._logprob = logprob_table or {}

    def to_tokens(self, text: str, prepend_bos: bool = False) -> torch.Tensor:
        word = text.strip().lower()
        n = 1 if word in self._single else 2
        return torch.zeros(1, n, dtype=torch.long)

    def logprob_of(self, clean_text: str, answer: str) -> float:
        return self._logprob.get((clean_text, answer.strip().lower()), -10.0)


def _prompt(aliases, distractor="basketball", domain="Athlete Sport") -> ConflictPrompt:
    return ConflictPrompt(
        text=f"X plays sport of {distractor}. X plays sport of",
        clean_text="X plays sport of",
        prompt_type="substitution",
        domain=domain,
        memory_aliases=aliases,
        context_answer=distractor,
        memory_answer=aliases[0],
    )


class TestSelectMemoryAnswerA21:
    def test_first_alias_used_when_single_token(self):
        model = _FakeModel(single_token_words=["soccer", "football"])
        chosen = _select_memory_answer(model, ["soccer", "football"], use_logprob_refinement=False)
        assert chosen == "soccer"

    def test_skips_to_first_single_token_alias(self):
        model = _FakeModel(single_token_words=["football"])  # "mixed martial arts" is multi-word/multi-token
        chosen = _select_memory_answer(
            model, ["mixed martial arts", "football"], use_logprob_refinement=False
        )
        assert chosen == "football"

    def test_falls_back_to_first_alias_when_none_single_token(self):
        model = _FakeModel(single_token_words=[])
        chosen = _select_memory_answer(model, ["mixed martial arts", "American football"], use_logprob_refinement=False)
        assert chosen == "mixed martial arts"  # aliases[0], will fail the single-token filter downstream


class TestSelectMemoryAnswerA22Refinement:
    def test_picks_highest_logprob_among_single_token_aliases(self):
        model = _FakeModel(
            single_token_words=["soccer", "football"],
            logprob_table={("X plays sport of", "soccer"): -2.0, ("X plays sport of", "football"): -0.5},
        )
        chosen = _select_memory_answer(
            model, ["soccer", "football"], use_logprob_refinement=True, clean_text="X plays sport of",
        )
        assert chosen == "football"

    def test_refinement_only_considers_single_token_aliases(self):
        model = _FakeModel(
            single_token_words=["soccer"],
            logprob_table={("X plays sport of", "soccer"): -5.0, ("X plays sport of", "rugby league"): -0.1},
        )
        chosen = _select_memory_answer(
            model, ["soccer", "rugby league"], use_logprob_refinement=True, clean_text="X plays sport of",
        )
        assert chosen == "soccer"  # rugby league is multi-token, excluded despite better logprob


class TestAttachLogprobOracleRealForwardPass:
    """spec A2.2: for a model with NO native logprob_of (the real case --
    HookedTransformer has none), _select_memory_answer must attach a working
    real-forward-pass oracle and use it, not silently fall back to A2.1."""

    def test_real_oracle_picks_higher_logprob_alias(self):
        model = _FakeForwardModel(
            single_token_words=["soccer", "football"],
            raw_logit_by_word={"soccer": 5.0, "football": 1.0},
        )
        assert not hasattr(model, "logprob_of")
        chosen = _select_memory_answer(
            model, ["football", "soccer"], use_logprob_refinement=True,
            clean_text="Lionel Messi plays the sport of",
        )
        assert chosen == "soccer"
        assert hasattr(model, "logprob_of")  # oracle got attached as a side effect

    def test_oracle_not_attached_when_only_one_single_token_candidate(self):
        """No ambiguity to resolve -- must not touch the model (no forward pass)."""
        model = _FakeForwardModel(single_token_words=["soccer"])
        chosen = _select_memory_answer(
            model, ["soccer"], use_logprob_refinement=True,
            clean_text="X plays sport of",
        )
        assert chosen == "soccer"
        assert not hasattr(model, "logprob_of")


class TestLoadSingleTokenPrompts:
    def test_filters_out_multi_token_context_answer(self, monkeypatch):
        model = _FakeModel(single_token_words=["soccer"])  # "basketball" is 2 tokens here
        import singleton_fix.single_token_loader as sl
        monkeypatch.setattr(sl, "load_conflict_prompts", lambda **kw: [_prompt(["soccer"])])
        kept = load_single_token_prompts(model, prompt_type="substitution")
        assert kept == []

    def test_keeps_rows_where_both_sides_become_single_token(self, monkeypatch):
        model = _FakeModel(single_token_words=["soccer", "basketball"])
        import singleton_fix.single_token_loader as sl
        monkeypatch.setattr(sl, "load_conflict_prompts", lambda **kw: [_prompt(["soccer"])])
        kept = load_single_token_prompts(model, prompt_type="substitution")
        assert len(kept) == 1
        assert kept[0].memory_answer == "soccer"

    def test_rewrites_memory_answer_per_a21(self, monkeypatch):
        model = _FakeModel(single_token_words=["football", "basketball"])  # "soccer" NOT single-token here
        import singleton_fix.single_token_loader as sl
        monkeypatch.setattr(
            sl, "load_conflict_prompts",
            lambda **kw: [_prompt(["soccer", "football"])],
        )
        kept = load_single_token_prompts(model, prompt_type="substitution", use_logprob_refinement=False)
        assert len(kept) == 1
        assert kept[0].memory_answer == "football"

    def test_survival_count_logged(self, monkeypatch, capsys):
        model = _FakeModel(single_token_words=["soccer", "basketball"])
        import singleton_fix.single_token_loader as sl
        monkeypatch.setattr(
            sl, "load_conflict_prompts",
            lambda **kw: [_prompt(["soccer"]), _prompt(["mixed martial arts"], distractor="basketball")],
        )
        kept = load_single_token_prompts(model, prompt_type="substitution")
        assert len(kept) == 1
        captured = capsys.readouterr()
        assert "single-token" in captured.out.lower()
        assert "1" in captured.out  # survived count appears somewhere in the log line


class TestHeldoutDomainPassthrough:
    def test_heldout_split_filters_to_named_domain(self, monkeypatch):
        model = _FakeModel(single_token_words=["soccer", "basketball", "paris", "london"])
        import singleton_fix.single_token_loader as sl

        def fake_load(prompt_type, heldout_domain=None, split="all"):
            rows = [
                _prompt(["soccer"], domain="Athlete Sport"),
                _prompt(["paris"], distractor="london", domain="Company Headquarter"),
            ]
            if heldout_domain is not None and split != "all":
                rows = (
                    [r for r in rows if r.domain != heldout_domain] if split == "train"
                    else [r for r in rows if r.domain == heldout_domain]
                )
            return rows

        monkeypatch.setattr(sl, "load_conflict_prompts", fake_load)
        train = load_single_token_prompts(
            model, heldout_domain="Company Headquarter", split="train",
        )
        heldout = load_single_token_prompts(
            model, heldout_domain="Company Headquarter", split="heldout",
        )
        assert all(p.domain == "Athlete Sport" for p in train)
        assert all(p.domain == "Company Headquarter" for p in heldout)


class TestPrintDomainComposition:
    def test_prints_per_domain_counts(self, capsys):
        from singleton_fix.single_token_loader import print_domain_composition
        prompts = [
            _prompt(["soccer"], domain="Athlete Sport"),
            _prompt(["soccer"], domain="Athlete Sport"),
            _prompt(["paris"], distractor="london", domain="Company Headquarter"),
        ]
        print_domain_composition(prompts)
        out = capsys.readouterr().out
        assert "Athlete Sport" in out
        assert "Company Headquarter" in out
        assert "total=3" in out


class TestLoadCachedSingleTokenPrompts:
    def _write_cache(self, tmp_path, monkeypatch, family="fam", variant="base"):
        import singleton_fix.single_token_loader as sl
        monkeypatch.setattr(sl, "_RESULTS_DIR", tmp_path)
        family_dir = tmp_path / family
        family_dir.mkdir(parents=True)
        cache = {
            "family": family, "variant": variant, "model": "fake/model",
            "n_survived": 2,
            "domain_counts": {"Athlete Sport": 1, "Company Headquarter": 1},
            "n_a22_changed_vs_a21": 0,
            "rows": [
                {
                    "row_index": 0, "domain": "Athlete Sport",
                    "clean_text": "X plays sport of",
                    "substitution_text": "X plays sport of basketball. X plays sport of",
                    "coherent_text": "X likes basketball. X plays sport of",
                    "memory_aliases": ["soccer", "football"],
                    "context_answer": "basketball", "memory_answer": "football",
                    "a21_naive_answer": "soccer", "a22_changed": True,
                },
                {
                    "row_index": 5, "domain": "Company Headquarter",
                    "clean_text": "Y is headquartered in",
                    "substitution_text": "Y is headquartered in london. Y is headquartered in",
                    "coherent_text": "Y is based in london. Y is headquartered in",
                    "memory_aliases": ["paris"],
                    "context_answer": "london", "memory_answer": "paris",
                    "a21_naive_answer": "paris", "a22_changed": False,
                },
            ],
        }
        (family_dir / f"single_token_survival_{family}_{variant}.json").write_text(
            json.dumps(cache), encoding="utf-8",
        )

    def test_reconstructs_conflict_prompts_from_cache(self, tmp_path, monkeypatch):
        from singleton_fix.single_token_loader import load_cached_single_token_prompts
        self._write_cache(tmp_path, monkeypatch)
        prompts = load_cached_single_token_prompts("fam", "base", prompt_type="substitution")
        assert len(prompts) == 2
        assert prompts[0].text == "X plays sport of basketball. X plays sport of"
        assert prompts[0].memory_answer == "football"  # the A2-corrected answer, not the naive one
        assert prompts[0].memory_aliases == ["soccer", "football"]

    def test_prompt_type_selects_coherent_text(self, tmp_path, monkeypatch):
        from singleton_fix.single_token_loader import load_cached_single_token_prompts
        self._write_cache(tmp_path, monkeypatch)
        prompts = load_cached_single_token_prompts("fam", "base", prompt_type="coherent")
        assert prompts[0].text == "X likes basketball. X plays sport of"
        assert prompts[0].prompt_type == "coherent"

    def test_heldout_split_filters_by_domain(self, tmp_path, monkeypatch):
        from singleton_fix.single_token_loader import load_cached_single_token_prompts
        self._write_cache(tmp_path, monkeypatch)
        train = load_cached_single_token_prompts(
            "fam", "base", heldout_domain="Company Headquarter", split="train",
        )
        heldout = load_cached_single_token_prompts(
            "fam", "base", heldout_domain="Company Headquarter", split="heldout",
        )
        assert [p.domain for p in train] == ["Athlete Sport"]
        assert [p.domain for p in heldout] == ["Company Headquarter"]

    def test_missing_cache_raises_clear_error(self, tmp_path, monkeypatch):
        import singleton_fix.single_token_loader as sl
        from singleton_fix.single_token_loader import load_cached_single_token_prompts
        monkeypatch.setattr(sl, "_RESULTS_DIR", tmp_path)
        try:
            load_cached_single_token_prompts("nonexistent_family", "base")
            assert False, "expected FileNotFoundError"
        except FileNotFoundError as e:
            assert "nonexistent_family" in str(e)
