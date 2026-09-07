"""Tests for path_patching.py.

Unit tests use MockModel (no GPU required).  The mock gives each answer a fixed
raw logit at the second-to-last position of the answer-appended sequence.
answer_log_prob then reads log_softmax at that position, giving controllable
log-prob margins without a real model.

Smoke tests (real GPT-2, @pytest.mark.slow) are at the bottom.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Optional, Tuple

import pytest
import torch
import numpy as np

from contract import ConflictPrompt, HeadScores
from path_patching import run_path_patching, _trimmed_mean  # noqa: E402


# ---------------------------------------------------------------------------
# Mock vocabulary
# ---------------------------------------------------------------------------
VOCAB_SIZE  = 256
MEM_TOKEN   = 10   # token ID for "soccer" in the mock
CTX_TOKEN   = 20   # token ID for "basketball" in the mock

MOCK_VOCAB = {"soccer": MEM_TOKEN, "basketball": CTX_TOKEN}


# ---------------------------------------------------------------------------
# MockModel
# ---------------------------------------------------------------------------

class MockModel:
    """Lightweight duck-type for HookedTransformer.

    set_logits(base_text, mem_raw, ctx_raw) registers raw logit values so that:
      - model(base_text + " soccer")     → logits with logits[0,-2,MEM_TOKEN]=mem_raw
      - model(base_text + " basketball") → logits with logits[0,-2,CTX_TOKEN]=ctx_raw

    This drives answer_log_prob (which reads log_softmax at position -2) to
    return controllable log-prob margins without a real model.

    run_with_hooks ignores hooks (identity), so identity-patch scores are 0.
    """

    class cfg:
        n_layers = 2
        n_heads  = 2

    class tokenizer:
        @staticmethod
        def decode(ids: List[int]) -> str:
            _map = {MEM_TOKEN: " soccer", CTX_TOKEN: " basketball"}
            return _map.get(ids[0] if ids else 0, " unknown")

    def __init__(self) -> None:
        self._d_head    = 4
        self._lengths:    Dict[str, int]                      = {}
        self._logit_spec: Dict[str, Tuple[float, float]]      = {}
        self._token_ids:  Dict[str, List[int]]                = {}
        self._next_word_id = 1000  # avoid colliding with MEM_TOKEN=10 / CTX_TOKEN=20
        self.run_with_cache_calls: List[str] = []  # every text passed to run_with_cache, in order

    def set_length(self, text: str, n: int) -> None:
        self._lengths[text] = n

    def set_tokens(self, text: str, ids: List[int]) -> None:
        """Register exact token ids for `text` (used by alignment-sensitive tests).
        Also sets the length to len(ids) for consistency with existing filters."""
        self._token_ids[text] = list(ids)
        self._lengths[text] = len(ids)

    def set_logits(self, base_text: str, mem: float, ctx: float) -> None:
        """Register (mem_raw, ctx_raw) logit values for a base text.

        Effective for model(base_text + " soccer") → mem raw at (-2, MEM_TOKEN)
                     and model(base_text + " basketball") → ctx raw at (-2, CTX_TOKEN).

        For log-prob margins (typical mock values):
          Raw logit 5.0 → log_softmax ≈ -1.0   (high probability token)
          Raw logit 1.0 → log_softmax ≈ -4.6   (low probability token)

        So set_logits(corrupt, mem=1.0, ctx=5.0) gives ctx_margin_c ≈ +3.6
           set_logits(clean,   mem=5.0, ctx=1.0) gives ctx_margin_k ≈ -3.6
           → denominator ≈ 7.2 > epsilon=0.05 ✓
        """
        self._logit_spec[base_text] = (mem, ctx)

    def _get_answer_info(self, text: str) -> Optional[Tuple[str, int, float]]:
        """If text ends with a known answer, return (base, token_id, raw_logit)."""
        for answer, token_id in MOCK_VOCAB.items():
            suffix = " " + answer
            if text.endswith(suffix):
                base = text[:-len(suffix)]
                if base in self._logit_spec:
                    mem_raw, ctx_raw = self._logit_spec[base]
                    raw = mem_raw if token_id == MEM_TOKEN else ctx_raw
                    return base, token_id, raw
        return None

    def _build_logit_tensor(self, text: str) -> torch.Tensor:
        info = self._get_answer_info(text)
        if info is not None:
            base, token_id, raw = info
            base_len = self._lengths.get(base, 5)
            n = base_len + 1                     # base tokens + 1 answer token
            logits = torch.zeros(1, n, VOCAB_SIZE)
            logits[0, -2, token_id] = raw        # position predicting the answer token
            return logits
        n = self._lengths.get(text, 5)
        return torch.zeros(1, n, VOCAB_SIZE)

    def to_tokens(self, text: str, prepend_bos: bool = True) -> torch.Tensor:
        # Exact override takes priority (alignment-sensitive tests).
        if text in self._token_ids:
            return torch.tensor([self._token_ids[text]], dtype=torch.long)
        # Return a single-token tensor for known answer words (for _answer_ids).
        stripped = text.strip().lower()
        if stripped in MOCK_VOCAB:
            return torch.tensor([[MOCK_VOCAB[stripped]]])
        # Deterministic per-word ids so distinct texts get distinguishable,
        # reproducible token sequences (needed now that alignment is
        # token-identity-based, not length-based).
        words = text.split(" ")
        ids = []
        for w in words:
            key = f"__word__{w}"
            if key not in self._token_ids:
                self._token_ids[key] = [self._next_word_id]
                self._next_word_id += 1
            ids.append(self._token_ids[key][0])
        n = self._lengths.get(text, len(ids))
        if n != len(ids):
            # A caller pre-registered an explicit length via set_length():
            # pad/truncate deterministically to honor it (existing
            # length-mismatch tests rely on set_length alone, no set_tokens).
            if n > len(ids):
                ids = ids + [0] * (n - len(ids))
            else:
                ids = ids[:n]
        return torch.tensor([ids], dtype=torch.long)

    def __call__(self, text: str, **kwargs) -> torch.Tensor:
        return self._build_logit_tensor(text)

    def run_with_cache(self, text: str, names_filter=None):
        self.run_with_cache_calls.append(text)
        logits = self._build_logit_tensor(text)
        n      = logits.shape[1]
        cache: Dict[str, torch.Tensor] = {
            f"blocks.{l}.attn.hook_z": torch.zeros(1, n, self.cfg.n_heads, self._d_head)
            for l in range(self.cfg.n_layers)
        }
        return logits, cache

    def run_with_hooks(self, text: str, fwd_hooks=None) -> torch.Tensor:
        # Ignore hooks → identity behaviour → identity-patch recovery = 0.
        return self._build_logit_tensor(text)


# ---------------------------------------------------------------------------
# Prompt factories
# ---------------------------------------------------------------------------

def _make_prompt_with_registered_cf(
    model: MockModel,
    tag: str,
    corrupt_len: int,
    cf_len: int,
    corrupt_logits: Tuple[float, float],  # (mem, ctx)
    cf_logits: Tuple[float, float],
) -> ConflictPrompt:
    """Build a prompt whose text embeds the distractor LITERALLY (so
    build_counterfactual_text produces a genuinely distinct counterfactual --
    the new contrast source, spec §A1), with independently-controlled
    lengths/logits for the corrupt and counterfactual texts.

    Prior to spec §A1, these fixtures registered a "clean_<tag>" string that
    the implementation read directly; post-fix, clean_text is never read, so
    the fixture must instead register the actual in-place-swap
    counterfactual text (corrupt text with "basketball" -> "soccer").
    """
    corrupt = f"subject_{tag} plays sport of basketball. subject_{tag} plays sport of"
    cf = f"subject_{tag} plays sport of soccer. subject_{tag} plays sport of"
    model.set_length(corrupt, corrupt_len)
    model.set_length(cf, cf_len)
    model.set_logits(corrupt, mem=corrupt_logits[0], ctx=corrupt_logits[1])
    model.set_logits(cf, mem=cf_logits[0], ctx=cf_logits[1])
    return ConflictPrompt(
        text=corrupt,
        clean_text=f"subject_{tag} plays sport of",  # unused by the fixed implementation
        prompt_type="substitution",
        domain="Athlete Sport",
        memory_aliases=["soccer"],
        context_answer="basketball",
        memory_answer="soccer",
    )


def _make_valid_prompt(model: MockModel, tag: str = "a") -> ConflictPrompt:
    """Create a prompt that passes ALL filters with comfortable margins."""
    # corrupt: ctx_raw=5.0 → ctx_lp≈-1.0, mem_raw=1.0 → mem_lp≈-4.6 → margin_c≈+3.6
    # cf:      mem_raw=5.0 → mem_lp≈-1.0, ctx_raw=1.0 → ctx_lp≈-4.6 → margin_k≈-3.6
    # denominator ≈ 3.6-(-3.6) = 7.2 > 0.05 ✓, denominator > 0 ✓
    return _make_prompt_with_registered_cf(
        model, tag, corrupt_len=9, cf_len=9,
        corrupt_logits=(1.0, 5.0), cf_logits=(5.0, 1.0),
    )


# ---------------------------------------------------------------------------
# Pure-function unit tests
# ---------------------------------------------------------------------------

class TestTrimmedMean:
    def test_empty_list_returns_nan(self):
        assert np.isnan(_trimmed_mean([], trim=0.10))

    def test_exact_values_no_trim(self):
        assert abs(_trimmed_mean([1.0, 2.0, 3.0, 4.0, 5.0], trim=0.0) - 3.0) < 1e-6

    def test_trimmed_ignores_extremes(self):
        scores  = [1.0] * 10 + [2.0] * 8 + [100.0, 200.0]
        assert _trimmed_mean(scores, trim=0.10) < float(np.mean(scores))

    def test_single_element(self):
        assert abs(_trimmed_mean([7.0], trim=0.10) - 7.0) < 1e-6

    def test_preserves_sign(self):
        assert abs(_trimmed_mean([-1.0, -2.0, 1.0, 2.0], trim=0.10)) < 1.0


# ---------------------------------------------------------------------------
# Sign convention (pure formula, no model)
# ---------------------------------------------------------------------------

class TestSignConvention:
    def test_context_driving_head_positive(self):
        recovery    = 4.0 - 1.0   # ctx_margin_c=4, ctx_margin_p=1
        denominator = 4.0 - (-2.0)
        assert recovery / denominator > 0

    def test_memory_suppressing_head_negative(self):
        recovery    = 4.0 - 5.0
        denominator = 4.0 - (-2.0)
        assert recovery / denominator < 0

    def test_no_effect_head_is_near_zero(self):
        recovery    = 4.0 - 4.0
        denominator = 4.0 - (-2.0)
        assert abs(recovery / denominator) < 1e-9


# ---------------------------------------------------------------------------
# Output shape and types
# ---------------------------------------------------------------------------

class TestOutputShapeAndTypes:
    def test_returns_correct_number_of_heads(self):
        model   = MockModel()
        prompts = [_make_valid_prompt(model, str(i)) for i in range(3)]
        scores  = run_path_patching(model, prompts, n_subset=None)
        assert len(scores) == model.cfg.n_layers * model.cfg.n_heads

    def test_keys_are_int_tuples(self):
        model   = MockModel()
        prompts = [_make_valid_prompt(model, str(i)) for i in range(3)]
        for key in run_path_patching(model, prompts):
            assert isinstance(key, tuple) and len(key) == 2
            assert all(isinstance(x, int) for x in key)

    def test_layer_and_head_indices_in_range(self):
        model   = MockModel()
        prompts = [_make_valid_prompt(model, str(i)) for i in range(3)]
        for l, h in run_path_patching(model, prompts):
            assert 0 <= l < model.cfg.n_layers
            assert 0 <= h < model.cfg.n_heads


# ---------------------------------------------------------------------------
# Identity-patch scores are zero
# ---------------------------------------------------------------------------

class TestIdentityPatch:
    def test_identity_patch_max_abs_near_zero(self):
        model   = MockModel()
        prompts = [_make_valid_prompt(model, str(i)) for i in range(5)]
        meta: Dict = {}
        run_path_patching(model, prompts, _meta_out=meta)
        assert meta["noise_floor"]["identity_patch_max_abs"] < 1e-4


# ---------------------------------------------------------------------------
# Length-mismatch filter
# ---------------------------------------------------------------------------

class TestLengthMismatchFilter:
    def test_large_mismatch_skipped(self):
        model = MockModel()
        p_bad = _make_prompt_with_registered_cf(
            model, "bad", corrupt_len=10, cf_len=3,
            corrupt_logits=(1.0, 5.0), cf_logits=(5.0, 1.0),
        )
        meta: Dict = {}
        run_path_patching(model, [p_bad], l_max_diff=4, _meta_out=meta)
        assert meta["n_skipped_length_mismatch"] == 1
        assert meta["n_prompts_used"] == 0

    def test_small_mismatch_kept(self):
        model  = MockModel()
        p_ok   = _make_valid_prompt(model, "ok")
        meta: Dict = {}
        run_path_patching(model, [p_ok], l_max_diff=4, _meta_out=meta)
        assert meta["n_skipped_length_mismatch"] == 0
        assert meta["n_prompts_used"] == 1

    def test_boundary_mismatch_exactly_4_kept(self):
        model = MockModel()
        p = _make_prompt_with_registered_cf(
            model, "b4", corrupt_len=9, cf_len=5,  # |9-5|=4 ≤ 4 → kept
            corrupt_logits=(1.0, 5.0), cf_logits=(5.0, 1.0),
        )
        meta: Dict = {}
        run_path_patching(model, [p], l_max_diff=4, _meta_out=meta)
        assert meta["n_skipped_length_mismatch"] == 0

    def test_mismatch_5_skipped(self):
        model = MockModel()
        p = _make_prompt_with_registered_cf(
            model, "b5", corrupt_len=10, cf_len=5,  # |10-5|=5 > 4 → skipped
            corrupt_logits=(1.0, 5.0), cf_logits=(5.0, 1.0),
        )
        meta: Dict = {}
        run_path_patching(model, [p], l_max_diff=4, _meta_out=meta)
        assert meta["n_skipped_length_mismatch"] == 1


# ---------------------------------------------------------------------------
# No-op swap guard
# ---------------------------------------------------------------------------

class TestNoSwapGuard:
    def test_prompt_missing_distractor_is_skipped_not_scored(self):
        """context_answer not present verbatim in p.text -> build_counterfactual_text
        is a silent no-op (cf_text == p.text). Must be skipped, not scored."""
        model = MockModel()
        corrupt = "subject_noswap plays sport of football."  # no "basketball" substring
        model.set_tokens(corrupt, [1, 2, 3, 4, 5])
        p = ConflictPrompt(
            text=corrupt,
            clean_text="subject_noswap plays sport of",
            prompt_type="substitution",
            domain="Athlete Sport",
            memory_aliases=["soccer"],
            context_answer="basketball",   # not actually in `corrupt`
            memory_answer="soccer",
        )
        meta: Dict = {}
        run_path_patching(model, [p], _meta_out=meta)
        assert meta["n_skipped_no_swap"] == 1
        assert meta["n_prompts_used"] == 0

    def test_normal_prompt_not_affected_by_guard(self):
        model = MockModel()
        p_ok = _make_valid_prompt(model, "swap_ok")
        meta: Dict = {}
        run_path_patching(model, [p_ok], _meta_out=meta)
        assert meta["n_skipped_no_swap"] == 0
        assert meta["n_prompts_used"] == 1


# ---------------------------------------------------------------------------
# Degenerate-margin filter
# ---------------------------------------------------------------------------

class TestDegenerateMarginFilter:
    def test_zero_denominator_skipped(self):
        """Equal raw logits on both corrupt and cf → margin_c = margin_k = 0 → skip."""
        model = MockModel()
        p = _make_prompt_with_registered_cf(
            model, "dm", corrupt_len=9, cf_len=9,
            corrupt_logits=(3.0, 3.0), cf_logits=(3.0, 3.0),
        )
        meta: Dict = {}
        run_path_patching(model, [p], _meta_out=meta)
        assert meta["n_skipped_degenerate_margin"] == 1
        assert meta["n_prompts_used"] == 0

    def test_negative_denominator_skipped(self):
        """corrupt prefers memory, cf prefers context → denominator < 0 → skip."""
        model = MockModel()
        p = _make_prompt_with_registered_cf(
            model, "neg", corrupt_len=9, cf_len=9,
            corrupt_logits=(5.0, 1.0),  # margin_c < 0
            cf_logits=(1.0, 5.0),       # margin_k > 0 → denom < 0
        )
        meta: Dict = {}
        run_path_patching(model, [p], _meta_out=meta)
        assert meta["n_skipped_degenerate_margin"] == 1

    def test_sufficient_denominator_kept(self):
        model  = MockModel()
        p_ok   = _make_valid_prompt(model, "dm_ok")
        meta: Dict = {}
        run_path_patching(model, [p_ok], _meta_out=meta)
        assert meta["n_skipped_degenerate_margin"] == 0


# ---------------------------------------------------------------------------
# Subset reproducibility
# ---------------------------------------------------------------------------

class TestSubsetReproducibility:
    def test_same_seed_gives_identical_results(self):
        model   = MockModel()
        prompts = [_make_valid_prompt(model, str(i)) for i in range(10)]
        s1 = run_path_patching(model, prompts, n_subset=5, seed=42)
        s2 = run_path_patching(model, prompts, n_subset=5, seed=42)
        for key in s1:
            if np.isnan(s1[key]):
                assert np.isnan(s2[key])
            else:
                assert abs(s1[key] - s2[key]) < 1e-9

    def test_different_seeds_give_different_indices(self):
        model   = MockModel()
        prompts = [_make_valid_prompt(model, str(i)) for i in range(20)]
        m1, m2  = {}, {}
        run_path_patching(model, prompts, n_subset=5, seed=42,  _meta_out=m1)
        run_path_patching(model, prompts, n_subset=5, seed=999, _meta_out=m2)
        assert m1.get("subset_indices") != m2.get("subset_indices")

    def test_no_subset_uses_all_prompts(self):
        model   = MockModel()
        prompts = [_make_valid_prompt(model, str(i)) for i in range(5)]
        meta: Dict = {}
        run_path_patching(model, prompts, n_subset=None, _meta_out=meta)
        assert meta["n_prompts_input"] == 5
        assert meta.get("subset_indices") is None


# ---------------------------------------------------------------------------
# head_subset
# ---------------------------------------------------------------------------

class TestHeadSubset:
    def test_subset_scores_equal_full_run_for_same_heads(self):
        model   = MockModel()
        prompts = [_make_valid_prompt(model, str(i)) for i in range(5)]
        subset  = [(0, 0), (1, 1)]
        full    = run_path_patching(model, prompts)
        sub     = run_path_patching(model, prompts, head_subset=subset)
        for h in subset:
            fv, sv = full[h], sub[h]
            if np.isnan(fv):
                assert np.isnan(sv)
            else:
                assert abs(fv - sv) < 1e-9

    def test_subset_output_contains_only_subset_heads(self):
        model   = MockModel()
        prompts = [_make_valid_prompt(model, str(i)) for i in range(3)]
        scores  = run_path_patching(model, prompts, head_subset=[(0, 1)])
        assert set(scores.keys()) == {(0, 1)}

    def test_none_subset_gives_all_heads(self):
        model   = MockModel()
        prompts = [_make_valid_prompt(model, str(i)) for i in range(3)]
        assert len(run_path_patching(model, prompts)) == model.cfg.n_layers * model.cfg.n_heads

    def test_identity_patch_still_near_zero_with_subset(self):
        model   = MockModel()
        prompts = [_make_valid_prompt(model, str(i)) for i in range(5)]
        meta: Dict = {}
        run_path_patching(model, prompts, head_subset=[(0, 0), (1, 1)], _meta_out=meta)
        assert meta["noise_floor"]["identity_patch_max_abs"] < 1e-4

    def test_meta_records_head_subset_and_count(self):
        model   = MockModel()
        prompts = [_make_valid_prompt(model, str(i)) for i in range(3)]
        subset  = [(0, 0), (1, 0)]
        meta: Dict = {}
        run_path_patching(model, prompts, head_subset=subset, _meta_out=meta)
        assert meta["head_subset"]    == ["0_0", "1_0"]
        assert meta["n_heads_scored"] == 2


# ---------------------------------------------------------------------------
# Aligned (in-place-swap) contrast — spec §A1
# ---------------------------------------------------------------------------

def _make_aligned_prompt(model: MockModel, tag: str = "a") -> ConflictPrompt:
    """A prompt whose corrupt text is IDENTICAL to its clean text except at the
    swapped word -- the new contrast this task wires in. `text` embeds the
    distractor once; the counterfactual (via contrast_pair) swaps it in-place."""
    corrupt = f"subject_{tag} plays sport of basketball. subject_{tag} plays sport of"
    model.set_tokens(corrupt, [1, 2, 3, 4, 99, 1, 2, 3, 5])
    cf = f"subject_{tag} plays sport of soccer. subject_{tag} plays sport of"
    model.set_tokens(cf, [1, 2, 3, 4, 55, 1, 2, 3, 5])
    model.set_logits(corrupt, mem=1.0, ctx=5.0)   # margin_c ≈ +3.6
    model.set_logits(cf,      mem=5.0, ctx=1.0)   # margin_k ≈ -3.6 (denominator ≈ 7.2)
    return ConflictPrompt(
        text=corrupt,
        clean_text=f"subject_{tag} plays sport of",   # present but must NOT be read (§A1)
        prompt_type="substitution",
        domain="Athlete Sport",
        memory_aliases=["soccer"],
        context_answer="basketball",
        memory_answer="soccer",
    )


class TestAlignedContrastDoesNotReadCleanText:
    def test_clean_text_is_never_run_with_cache(self):
        """The patch-source cache must come from the counterfactual text, not
        clean_text. Register a DISTINCT, poisoned clean_text (with its own
        set_tokens/set_logits) so a regression that reads it is detectable via
        the call log, not just via output equality (which this mock can't
        distinguish once run_with_hooks ignores patch content)."""
        model = MockModel()
        p = _make_aligned_prompt(model, "ct")
        model.set_tokens(p.clean_text, [777, 778, 779])
        model.set_logits(p.clean_text, mem=9.0, ctx=9.0)

        run_path_patching(model, [p], head_subset=[(0, 0)])

        cf_text = "subject_ct plays sport of soccer. subject_ct plays sport of"
        assert p.clean_text not in model.run_with_cache_calls
        assert cf_text in model.run_with_cache_calls

    def test_scores_unaffected_by_clean_text_content(self):
        """Two prompts identical except for clean_text must score identically,
        since clean_text is no longer part of the contrast. Each clean_text is
        registered with WILDLY different tokens/logits so a regression that
        reads clean_text would visibly change the computed margin/denominator
        (this mock's run_with_hooks ignores patch content, but margin_k/
        denominator are computed via direct model(...) calls, which DO read
        _logit_spec/_lengths keyed by the exact text -- including clean_text
        under the old, buggy implementation)."""
        model = MockModel()
        p1 = _make_aligned_prompt(model, "same")
        p2 = ConflictPrompt(
            text=p1.text, clean_text="totally different unrelated clean prompt",
            prompt_type=p1.prompt_type, domain=p1.domain,
            memory_aliases=p1.memory_aliases, context_answer=p1.context_answer,
            memory_answer=p1.memory_answer,
        )
        # Poison each clean_text distinctly -- if the implementation reads
        # clean_text for its counterfactual margin, s1 and s2 will diverge.
        model.set_tokens(p1.clean_text, [901, 902])
        model.set_logits(p1.clean_text, mem=0.1, ctx=8.0)
        model.set_tokens(p2.clean_text, [903, 904, 905, 906])
        model.set_logits(p2.clean_text, mem=8.0, ctx=0.1)

        s1 = run_path_patching(model, [p1], head_subset=[(0, 0), (1, 1)])
        s2 = run_path_patching(model, [p2], head_subset=[(0, 0), (1, 1)])
        for h in s1:
            assert abs(s1[h] - s2[h]) < 1e-9


class TestIdentityCheckUsesCorruptNotClean:
    def test_identity_recovery_still_near_zero(self):
        model = MockModel()
        prompts = [_make_aligned_prompt(model, str(i)) for i in range(3)]
        meta: Dict = {}
        run_path_patching(model, prompts, head_subset=[(0, 0), (1, 1)], _meta_out=meta)
        assert meta["noise_floor"]["identity_patch_max_abs"] < 1e-4


class TestMetaKeyRenameStaysBackwardCompatible:
    def test_L_clean_distribution_key_still_present(self):
        """A3.1/A1.6: the legacy key name is kept (downstream run_pp_topheads.py
        reads it) but now holds counterfactual-text length stats."""
        model = MockModel()
        prompts = [_make_aligned_prompt(model, str(i)) for i in range(2)]
        meta: Dict = {}
        run_path_patching(model, prompts, head_subset=[(0, 0)], _meta_out=meta)
        assert "L_clean_distribution" in meta


# ---------------------------------------------------------------------------
# Error isolation + crash-resilient checkpointing (long-run robustness)
# ---------------------------------------------------------------------------

from path_patching.path_patching import _save_pp_checkpoint, _load_pp_checkpoint  # noqa: E402


class TestErrorIsolation:
    def test_one_failing_prompt_is_isolated_not_fatal(self):
        """A single prompt that raises mid-processing must be counted and
        skipped, NOT crash the whole cell -- the other prompts still score."""
        class FlakyModel(MockModel):
            def run_with_cache(self, text, names_filter=None):
                if "subject_BOOM plays sport of soccer" in text:
                    raise RuntimeError("simulated transient CUDA error")
                return super().run_with_cache(text, names_filter)

        model = FlakyModel()
        prompts = [
            _make_valid_prompt(model, "a"),
            _make_valid_prompt(model, "b"),
            _make_valid_prompt(model, "BOOM"),   # raises in the main patching loop
            _make_valid_prompt(model, "c"),
        ]
        meta: Dict = {}
        scores = run_path_patching(model, prompts, n_subset=None, _meta_out=meta)

        # Run completed and returned a full head grid despite the bad prompt.
        assert len(scores) == model.cfg.n_layers * model.cfg.n_heads
        assert meta["n_skipped_error"] == 1
        assert meta["n_prompts_used"] == 3   # the 3 good prompts scored

    def test_failing_prompt_leaves_no_partial_scores(self):
        """The atomic per-prompt commit means a mid-head-loop failure appends
        NOTHING for that prompt -- every head's score list stays equal-length."""
        class FailMidHeads(MockModel):
            def __init__(self):
                super().__init__()
                self._hook_calls = 0
            def run_with_hooks(self, text, fwd_hooks=None):
                self._hook_calls += 1
                # fail deep into the run, after some heads of some prompt scored
                if self._hook_calls == 5:
                    raise RuntimeError("boom mid head-loop")
                return super().run_with_hooks(text, fwd_hooks)

        model = FailMidHeads()
        prompts = [_make_valid_prompt(model, str(i)) for i in range(3)]
        scores = run_path_patching(model, prompts, n_subset=None)
        # No crash; all heads present.
        assert len(scores) == model.cfg.n_layers * model.cfg.n_heads


class TestCheckpointHelpers:
    def _heads(self):
        return [(l, h) for l in range(2) for h in range(2)]

    def test_save_load_roundtrip(self, tmp_path):
        ckpt = str(tmp_path / "pp.ckpt.json")
        heads = self._heads()
        hsl = {k: [0.1, 0.2] for k in heads}
        _save_pp_checkpoint(
            ckpt, next_prompt_idx=4, head_scores_list=hsl,
            kept_prompt_indices=[0, 1], l_corrupts=[9, 9], l_cleans=[9, 9],
            counters={"n_skipped_length": 1, "n_skipped_degenerate_margin": 2,
                      "n_skipped_no_swap": 3, "n_skipped_error": 4},
            n_total=10,
        )
        loaded = _load_pp_checkpoint(ckpt, n_total=10, heads_iter=heads)
        assert loaded is not None
        assert loaded["next_prompt_idx"] == 4
        assert loaded["head_scores_list"] == hsl
        assert loaded["kept_prompt_indices"] == [0, 1]
        assert loaded["n_skipped_length"] == 1
        assert loaded["n_skipped_error"] == 4

    def test_load_returns_none_on_ntotal_mismatch(self, tmp_path):
        ckpt = str(tmp_path / "pp.ckpt.json")
        heads = self._heads()
        _save_pp_checkpoint(ckpt, 4, {k: [] for k in heads}, [], [], [],
                            {"n_skipped_length": 0, "n_skipped_degenerate_margin": 0,
                             "n_skipped_no_swap": 0, "n_skipped_error": 0}, n_total=10)
        # different n_total → stale checkpoint must be ignored
        assert _load_pp_checkpoint(ckpt, n_total=999, heads_iter=heads) is None

    def test_load_returns_none_on_head_mismatch(self, tmp_path):
        ckpt = str(tmp_path / "pp.ckpt.json")
        heads = self._heads()
        _save_pp_checkpoint(ckpt, 4, {k: [] for k in heads}, [], [], [],
                            {"n_skipped_length": 0, "n_skipped_degenerate_margin": 0,
                             "n_skipped_no_swap": 0, "n_skipped_error": 0}, n_total=10)
        # different head set → must be ignored
        assert _load_pp_checkpoint(ckpt, n_total=10, heads_iter=[(0, 0)]) is None

    def test_load_returns_none_when_absent(self, tmp_path):
        assert _load_pp_checkpoint(str(tmp_path / "nope.json"), 10, self._heads()) is None


class TestCheckpointResume:
    def test_resume_skips_done_and_includes_checkpointed_scores(self, tmp_path):
        ckpt = str(tmp_path / "pp.ckpt.json")
        model = MockModel()
        prompts = [_make_valid_prompt(model, str(i)) for i in range(6)]
        heads = [(l, h) for l in range(2) for h in range(2)]

        # Pre-seed a checkpoint saying prompts 0..3 are already scored (4 values
        # per head), so the resumed run should only process prompts 4 and 5.
        _save_pp_checkpoint(
            ckpt, next_prompt_idx=4,
            head_scores_list={k: [0.5, 0.5, 0.5, 0.5] for k in heads},
            kept_prompt_indices=[0, 1, 2, 3], l_corrupts=[9, 9, 9, 9], l_cleans=[9, 9, 9, 9],
            counters={"n_skipped_length": 0, "n_skipped_degenerate_margin": 0,
                      "n_skipped_no_swap": 0, "n_skipped_error": 0},
            n_total=6,
        )
        meta: Dict = {}
        run_path_patching(model, prompts, n_subset=None,
                          checkpoint_path=ckpt, checkpoint_every=100, _meta_out=meta)

        # prompts 0..3 came from the checkpoint, 4..5 scored fresh → all 6 kept
        assert set(meta["kept_prompt_indices"]) == {0, 1, 2, 3, 4, 5}
        assert meta["n_prompts_used"] == 6

    def test_checkpoint_removed_after_successful_completion(self, tmp_path):
        ckpt = str(tmp_path / "pp.ckpt.json")
        model = MockModel()
        prompts = [_make_valid_prompt(model, str(i)) for i in range(6)]
        run_path_patching(model, prompts, n_subset=None,
                          checkpoint_path=ckpt, checkpoint_every=2)
        # Successful cell cleans up its checkpoint (the real output file replaces it).
        assert not os.path.exists(ckpt)

    def test_checkpoint_absent_path_is_a_noop(self, tmp_path):
        """checkpoint_path=None (default) must behave exactly as before."""
        model = MockModel()
        prompts = [_make_valid_prompt(model, str(i)) for i in range(3)]
        scores_no_ckpt = run_path_patching(model, prompts, n_subset=None)
        assert len(scores_no_ckpt) == model.cfg.n_layers * model.cfg.n_heads


# ---------------------------------------------------------------------------
# Smoke test — real GPT-2 (slow, requires HF download)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestSmokeGPT2:
    @pytest.fixture(scope="class")
    def model_and_prompts(self):
        from foundation import load_model, load_conflict_prompts
        torch.manual_seed(0)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        model   = load_model("gpt2")
        prompts = load_conflict_prompts(prompt_type="substitution")
        return model, prompts

    def test_returns_144_finite_entries(self, model_and_prompts):
        model, prompts = model_and_prompts
        scores = run_path_patching(model, prompts, n_subset=50, seed=42)
        assert len(scores) == 144
        assert sum(1 for v in scores.values() if np.isfinite(v)) > 0

    def test_identity_patch_noise_below_threshold(self, model_and_prompts):
        model, prompts = model_and_prompts
        meta: Dict = {}
        run_path_patching(model, prompts, n_subset=50, seed=42, _meta_out=meta)
        assert meta["noise_floor"]["identity_patch_max_abs"] < 1e-4

    def test_signal_exceeds_noise_floor(self, model_and_prompts):
        model, prompts = model_and_prompts
        meta: Dict = {}
        scores = run_path_patching(model, prompts, n_subset=50, seed=42, _meta_out=meta)
        finite      = [v for v in scores.values() if np.isfinite(v)]
        top_score   = max(finite) if finite else 0.0
        p95         = meta["noise_floor"].get("random_pair_p95", 0.0) or 0.0
        threshold   = max(0.05, 2 * (p95 if np.isfinite(p95) else 0.0))
        assert top_score > threshold

    def test_not_all_scores_identical(self, model_and_prompts):
        model, prompts = model_and_prompts
        scores = run_path_patching(model, prompts, n_subset=50, seed=42)
        finite = [v for v in scores.values() if np.isfinite(v)]
        assert len(set(round(v, 6) for v in finite)) > 1
