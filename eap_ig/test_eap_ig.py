"""Tests for EAP-IG edge attribution.

Fast unit tests (no model download) cover the pure logic: IG coefficients, the
node graph, topological edge validity, and the contrast-pair construction.

Slow tests (marked `slow`, real GPT-2) cover the actual TransformerLens hook
plumbing: split_qkv correctness, per-prompt scoring, end-to-end structure, the
metric-sign sanity, and plain-EAP vs EAP-IG agreement.

Run fast only:   pytest eap_ig/test_eap_ig.py -m "not slow"
Run everything:  pytest eap_ig/test_eap_ig.py
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from contract import ConflictPrompt
from eap_ig.eap_ig import (
    EdgeAttributionConfig,
    _alignment_window,
    _answer_log_prob_diff,
    _common_suffix_len,
    _ig_alphas,
    _validity_mask,
    build_contrast_pair,
    build_graph,
    compute_edge_attribution,
    configure_model_for_edges,
    score_prompt,
)


# ---------------------------------------------------------------------------
# Fast unit tests
# ---------------------------------------------------------------------------

def test_ig_alphas_plain_vs_ig():
    # n_steps == 1 is plain EAP: gradient at the corrupted baseline (alpha = 0).
    assert _ig_alphas(1) == [0.0]
    # n_steps >= 2 is the midpoint Riemann rule, strictly inside (0, 1).
    a = _ig_alphas(5)
    assert len(a) == 5
    assert all(0.0 < x < 1.0 for x in a)
    assert a == pytest.approx([0.1, 0.3, 0.5, 0.7, 0.9])


def test_common_suffix_len_stops_at_first_mismatch():
    a = torch.tensor([1, 2, 3, 4, 5])
    b = torch.tensor([9, 9, 3, 4, 5])
    assert _common_suffix_len(a, b) == 3  # only [3, 4, 5] match from the end

    # No overlap at all.
    assert _common_suffix_len(torch.tensor([1, 2]), torch.tensor([3, 4])) == 0

    # Full match when one is a suffix of the other at the id level.
    assert _common_suffix_len(torch.tensor([7, 8, 9]), torch.tensor([8, 9])) == 2


def _fake_model(n_layers=2, n_heads=2):
    return SimpleNamespace(cfg=SimpleNamespace(n_layers=n_layers, n_heads=n_heads))


def test_build_graph_counts_full():
    g = build_graph(_fake_model(2, 2), "full")
    # sources: input + n_layers*(n_heads heads + 1 mlp) = 1 + 2*(2+1) = 7
    assert len(g.sources) == 1 + 2 * (2 + 1)
    # dests: n_layers*(3 letters * n_heads + 1 mlp_in) + logits = 2*(3*2+1) + 1 = 15
    assert len(g.dests) == 2 * (3 * 2 + 1) + 1
    assert g.sources[0].label == "input"
    assert g.dests[-1].label == "logits"


def test_build_graph_counts_coarse():
    g = build_graph(_fake_model(2, 2), "coarse")
    assert len(g.sources) == 1 + 2 * (2 + 1)
    # dests: n_layers resid_pre + logits = 2 + 1
    assert len(g.dests) == 2 + 1


def test_validity_mask_topology():
    g = build_graph(_fake_model(2, 2), "full")
    mask = _validity_mask(g)
    labels_src = [n.label for n in g.sources]
    labels_dst = [n.label for n in g.dests]

    def idx(labels, lbl):
        return labels.index(lbl)

    # input feeds every destination.
    assert mask[idx(labels_src, "input")].all()
    # logits is fed by every source.
    assert mask[:, idx(labels_dst, "logits")].all()
    # A layer-0 head must NOT feed its own layer's q/k/v input.
    s = idx(labels_src, "blocks.0.attn.hook_result[h0]")
    d_same = idx(labels_dst, "blocks.0.attn.hook_q_input[h1]")
    assert not mask[s, d_same]
    # ...but it DOES feed its own layer's MLP and a later layer's q input.
    assert mask[s, idx(labels_dst, "blocks.0.hook_mlp_in")]
    assert mask[s, idx(labels_dst, "blocks.1.attn.hook_q_input[h0]")]
    # An MLP must not feed its own input.
    sm = idx(labels_src, "blocks.0.hook_mlp_out")
    assert not mask[sm, idx(labels_dst, "blocks.0.hook_mlp_in")]


def test_contrast_pair_substitutes_context_for_memory_in_place():
    """The counterfactual must be the SAME sentence with only the disputed
    fact swapped -- not a different (shorter) sentence."""
    lookup = {
        "Lionel Messi plays the sport of basketball. Lionel Messi plays the sport of":
            torch.tensor([[50256, 1, 2, 3, 4, 5, 6, 7]]),
        "Lionel Messi plays the sport of soccer. Lionel Messi plays the sport of":
            torch.tensor([[50256, 1, 2, 3, 4, 8, 6, 7]]),
    }
    model = SimpleNamespace(to_tokens=lambda text: lookup[text])
    p = ConflictPrompt(
        text="Lionel Messi plays the sport of basketball. Lionel Messi plays the sport of",
        clean_text="Lionel Messi plays the sport of",
        prompt_type="substitution", domain="Athlete Sport",
        memory_aliases=["soccer"], context_answer="basketball", memory_answer="soccer",
    )
    pair = build_contrast_pair(model, p)
    assert pair is not None
    main, cf = pair
    assert torch.equal(main, lookup[
        "Lionel Messi plays the sport of basketball. Lionel Messi plays the sport of"])
    assert torch.equal(cf, lookup[
        "Lionel Messi plays the sport of soccer. Lionel Messi plays the sport of"])


def test_contrast_pair_returns_none_when_context_answer_not_in_text():
    model = SimpleNamespace(to_tokens=lambda text: torch.tensor([[50256, 1, 2, 3]]))
    p = ConflictPrompt(
        text="some unrelated text", clean_text="c", prompt_type="substitution", domain="d",
        memory_aliases=["mem"], context_answer="not present anywhere", memory_answer="mem",
    )
    assert build_contrast_pair(model, p) is None


def test_common_prefix_len_stops_at_first_mismatch():
    from eap_ig.eap_ig import _common_prefix_len

    a = torch.tensor([1, 2, 3, 4, 5])
    b = torch.tensor([1, 2, 3, 9, 9])
    assert _common_prefix_len(a, b) == 3  # only [1, 2, 3] match from the start
    assert _common_prefix_len(torch.tensor([1, 2]), torch.tensor([9, 9])) == 0
    assert _common_prefix_len(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3, 4])) == 3


def test_answer_log_prob_diff_raises_on_empty_answer_ids():
    """_answer_log_prob_diff must raise a clear ValueError on empty answer_ids,
    not crash inside .mean() or silently return NaN."""
    logits = torch.randn(1, 5, 100)
    empty_ids = torch.tensor([], dtype=torch.long)
    with pytest.raises(ValueError, match="empty"):
        _answer_log_prob_diff(logits, empty_ids)


# ---------------------------------------------------------------------------
# Slow tests (real GPT-2)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gpt2():
    from foundation import load_model
    return load_model("gpt2")


def _toy_prompt() -> ConflictPrompt:
    """A synthetic conflict prompt: conflict text carries the distractor
    sentence, clean text is the trailing query only -- the same shape as a
    real ParaConflict row. Avoids downloading the dataset."""
    return ConflictPrompt(
        text="The capital of France is Lyon. The capital of France is",
        clean_text="The capital of France is",
        prompt_type="substitution", domain="World Capital",
        memory_aliases=["Paris"], context_answer="Lyon", memory_answer="Paris",
    )


@pytest.mark.slow
def test_split_qkv_does_not_change_logits(gpt2):
    """Enabling the edge-attribution flags must not alter the forward output."""
    tokens = gpt2.to_tokens("The capital of France is Lyon. The capital of France is")
    with torch.no_grad():
        logits_plain = gpt2(tokens)
        configure_model_for_edges(gpt2, "full")
        logits_flags = gpt2(tokens)
    assert torch.allclose(logits_plain, logits_flags, atol=1e-4), (
        "use_split_qkv_input / use_attn_result changed the logits -- TL bug; "
        "fall back to coarse granularity."
    )


@pytest.mark.slow
def test_metric_sign_clean_gt_corrupt(gpt2):
    """ctx_log_prob > mem_log_prob on the conflict prompt: context-following
    should score the injected distractor higher than the memorized answer."""
    from foundation import answer_log_probs
    p = _toy_prompt()
    mem_lp, ctx_lp = answer_log_probs(gpt2, p.text, p.memory_answer, p.context_answer)
    assert ctx_lp > mem_lp, f"sign sanity failed: ctx_lp={ctx_lp} mem_lp={mem_lp}"


@pytest.mark.slow
@pytest.mark.parametrize("granularity", ["coarse", "full"])
def test_score_prompt_shape_and_finite(gpt2, granularity):
    configure_model_for_edges(gpt2, granularity)
    graph = build_graph(gpt2, granularity)
    cfg = EdgeAttributionConfig(n_steps=3, granularity=granularity, contract_device="cpu")
    edge = score_prompt(gpt2, graph, _toy_prompt(), cfg)
    assert edge.shape == (len(graph.sources), len(graph.dests))
    assert torch.isfinite(edge).all()


@pytest.mark.slow
def test_end_to_end_structure_and_validity(gpt2):
    cfg = EdgeAttributionConfig(n_steps=3, granularity="coarse", contract_device="cpu")
    prompts = [_toy_prompt() for _ in range(3)]
    scores = compute_edge_attribution(gpt2, prompts, cfg)
    assert isinstance(scores, dict) and len(scores) > 0
    for (s, d), v in scores.items():
        assert isinstance(s, str) and isinstance(d, str)
        assert isinstance(v, float) and torch.isfinite(torch.tensor(v))
    # invalid edges must be absent (validity enforced before serialisation)
    assert ("logits", "input") not in scores


def _spearman(a, b) -> float:
    """Spearman rank correlation without a scipy dependency."""
    import numpy as np
    ra = np.argsort(np.argsort(np.asarray(a, dtype=float)))
    rb = np.argsort(np.argsort(np.asarray(b, dtype=float)))
    return float(np.corrcoef(ra, rb)[0, 1])


@pytest.mark.slow
def test_determinism(gpt2):
    """Same prompts + same config -> identical edge scores (reproducibility DoD)."""
    prompts = [_toy_prompt() for _ in range(2)]
    cfg = EdgeAttributionConfig(n_steps=3, granularity="coarse", contract_device="cpu")
    a = compute_edge_attribution(gpt2, prompts, cfg)
    b = compute_edge_attribution(gpt2, prompts, cfg)
    assert a.keys() == b.keys()
    for k in a:
        assert a[k] == pytest.approx(b[k], abs=1e-6)


@pytest.mark.slow
def test_coherent_multi_occurrence_prompt(gpt2):
    """Coherent prompts repeat the distractor many times (~130 tok) -- the
    prefix+suffix-aligned scoring must not OOM/crash when the substitution
    touches multiple, widely-separated occurrences in the same text."""
    text = (
        "Dante plays the sport of basketball. Recognized widely, Dante's journey "
        "in basketball has been highlighted in many basketball publications. "
        "Question: What sport does Dante play? Answer: Dante plays the sport of"
    )
    p = ConflictPrompt(
        text=text, clean_text="Dante plays the sport of",
        prompt_type="coherent", domain="Athelete Sport",
        memory_aliases=["soccer"], context_answer="basketball", memory_answer="soccer",
    )
    main, cf = build_contrast_pair(gpt2, p)
    # Same sentence with "basketball" (1 token) replaced by "soccer" (1 token)
    # -- lengths match exactly since both are single GPT-2 tokens.
    assert main.shape[1] == cf.shape[1]
    assert not torch.equal(main, cf)  # but the tokens genuinely differ
    scores = compute_edge_attribution(
        gpt2, [p], EdgeAttributionConfig(n_steps=3, granularity="full"))
    assert len(scores) > 0 and all(isinstance(v, float) for v in scores.values())


@pytest.mark.slow
def test_substitution_window_covers_most_of_a_single_token_swap(gpt2):
    """When context_answer and memory_answer are both single tokens (the
    common case in this dataset), the ENTIRE sentence except that one word
    should land in the common prefix+suffix window -- this is the concrete
    fix for the old suffix-only design's signal loss on long prompts."""
    from eap_ig.eap_ig import _common_prefix_len, _common_suffix_len

    p = _toy_prompt()
    main, cf = build_contrast_pair(gpt2, p)
    main_ids, cf_ids = main[0], cf[0]
    L_pre = _common_prefix_len(main_ids, cf_ids)
    L_suf = _common_suffix_len(main_ids, cf_ids)
    total_window = L_pre + L_suf
    shorter_len = min(main_ids.shape[0], cf_ids.shape[0])
    # Only the one substituted token should be excluded (assuming "Lyon" and
    # "Paris" both tokenize to 1 token each for gpt2).
    assert total_window >= shorter_len - 1, (
        f"expected almost the entire {shorter_len}-token sequence to be in "
        f"the aligned window, got only {total_window} (L_pre={L_pre}, L_suf={L_suf})"
    )


def test_alignment_window_includes_disputed_span_when_lengths_match():
    """When main and cf tokenize to the SAME total length (the common
    single-token-swap case), the disputed position is provably at the same
    absolute index in both sequences -- _alignment_window must return the
    full sequence as the window instead of carving out that one position,
    recovering signal that _common_prefix_len/_common_suffix_len alone would
    still (correctly, for their own narrower contract) exclude."""
    # ids: [0,1,2, 99, 4,5,6] vs [0,1,2, 77, 4,5,6] -- same length, only
    # position 3 (the disputed token) differs.
    main_ids = torch.tensor([0, 1, 2, 99, 4, 5, 6])
    cf_ids = torch.tensor([0, 1, 2, 77, 4, 5, 6])
    L_pre, L_suf = _alignment_window(main_ids, cf_ids)
    assert (L_pre, L_suf) == (7, 0), (
        f"expected the full 7-token sequence to be used as the window "
        f"(equal lengths => no index shift), got L_pre={L_pre}, L_suf={L_suf}"
    )


def test_alignment_window_excludes_middle_when_lengths_differ():
    """When main and cf have DIFFERENT total lengths (a multi-token swap
    where context_answer/memory_answer tokenize to different lengths), the
    substitution shifts every later position -- the middle must still be
    excluded, and the window must fall back to the verified prefix/suffix
    runs, never overlapping."""
    # main: [0,1,2, 99,98, 4,5,6] (8 tokens, 2-token disputed span)
    # cf:   [0,1,2, 77, 4,5,6]    (7 tokens, 1-token disputed span)
    main_ids = torch.tensor([0, 1, 2, 99, 98, 4, 5, 6])
    cf_ids = torch.tensor([0, 1, 2, 77, 4, 5, 6])
    L_pre, L_suf = _alignment_window(main_ids, cf_ids)
    assert (L_pre, L_suf) == (3, 3), (
        f"expected prefix=3 (before the swap) and suffix=3 (after it), "
        f"got L_pre={L_pre}, L_suf={L_suf}"
    )
    assert L_pre + L_suf <= min(main_ids.shape[0], cf_ids.shape[0])


@pytest.mark.slow
def test_common_suffix_stops_before_leading_space_mismatch(gpt2):
    """Regression test for a real bug found on real ParaConflict data: a naive
    min(len(main), len(base)) suffix window is WRONG, because GPT-2's BPE
    tokenizes the same word differently at the start of a standalone string
    (no leading space, e.g. "Lionel" -> ["L","ion","el"]) vs mid-sentence
    (leading space, e.g. " Lionel" -> [" Lionel"], one token). clean_text is a
    literal string-suffix of text, but NOT a token-id suffix at the seam.
    """
    text = "Lionel Messi plays the sport of basketball. Lionel Messi plays the sport of"
    clean_text = "Lionel Messi plays the sport of"
    main = gpt2.to_tokens(text)[0]
    base = gpt2.to_tokens(clean_text)[0]

    L_naive = min(main.shape[0], base.shape[0])
    L_real = _common_suffix_len(main, base)

    # The naive window is wrong: it includes "Lionel" tokenized two different
    # ways (start-of-string "L"/"ion"/"el" vs mid-sentence " Lionel"), so it
    # is NOT actually identical token-for-token.
    naive_main_ids = main[-L_naive:].tolist()
    naive_base_ids = base[-L_naive:].tolist()
    assert naive_main_ids != naive_base_ids, (
        "expected the naive min-length window to be misaligned on this example "
        "-- if this now passes, the tokenizer's leading-space behavior changed "
        "and this regression test needs a new example"
    )

    # The real common-suffix window IS identical token-for-token, and is
    # strictly shorter than the naive (wrong) window.
    assert 0 < L_real < L_naive
    assert main[-L_real:].tolist() == base[-L_real:].tolist()


@pytest.mark.slow
def test_plain_vs_ig_agreement(gpt2):
    """Plain EAP (n_steps=1) and EAP-IG (n_steps=5) should rank edges similarly."""
    prompts = [_toy_prompt() for _ in range(3)]
    plain = compute_edge_attribution(
        gpt2, prompts, EdgeAttributionConfig(n_steps=1, granularity="coarse"))
    ig = compute_edge_attribution(
        gpt2, prompts, EdgeAttributionConfig(n_steps=5, granularity="coarse"))
    keys = list(plain)
    rho = _spearman([plain[k] for k in keys], [ig[k] for k in keys])
    assert rho > 0.5, f"plain-EAP vs EAP-IG rank agreement too low: rho={rho:.3f}"
