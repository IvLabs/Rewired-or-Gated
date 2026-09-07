"""Tests for single-token EAP-IG (eap_ig/eap_ig_single_token.py).

Fast tests use SimpleNamespace mocks (no model download) to check the
single-token invariant accounting and the provenance block. Slow tests
(marked `slow`, real GPT-2) check end-to-end equivalence to the full-window
path.
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
from eap_ig.eap_ig import EdgeAttributionConfig
from eap_ig.eap_ig_single_token import (
    SET_TAG,
    ALIAS_POLICY,
    build_result_json_single_token,
    compute_edge_attribution_single_token,
)


def _mock_graph_model():
    """A model whose to_tokens returns equal-length pairs for the swap and an
    UNequal-length pair for a poisoned prompt, so we can exercise the
    length-mismatch sanity skip without a real tokenizer."""
    lookup = {
        # equal length (single-token swap) -> should be scored
        "A plays basketball. A plays": torch.tensor([[0, 1, 2, 3, 4, 1, 5]]),
        "A plays soccer. A plays":     torch.tensor([[0, 1, 2, 9, 4, 1, 5]]),
        # UNequal length (mem tokenizes to 2) -> should be skipped + counted
        "B plays basketball. B plays": torch.tensor([[0, 1, 2, 3, 4, 1, 5]]),
        "B plays pingpong. B plays":   torch.tensor([[0, 1, 2, 9, 9, 4, 1, 5]]),
    }
    return SimpleNamespace(to_tokens=lambda t: lookup[t])


def test_length_mismatch_is_counted_and_skipped(monkeypatch):
    """A single-token run must treat an unequal-length pair as a sanity
    failure: skip it and count it in n_skipped_length_mismatch, never score
    it through the multi-token prefix/suffix path."""
    import eap_ig.eap_ig_single_token as st

    # Stub the heavy pieces: build_graph/score_prompt/configure/_validity_mask.
    fake_graph = SimpleNamespace(sources=[SimpleNamespace(label="input")],
                                 dests=[SimpleNamespace(label="logits")])
    monkeypatch.setattr(st, "configure_model_for_edges", lambda m, g: None)
    monkeypatch.setattr(st, "build_graph", lambda m, g: fake_graph)
    monkeypatch.setattr(st, "_validity_mask", lambda g: torch.ones((1, 1), dtype=torch.bool))
    monkeypatch.setattr(st, "score_prompt", lambda m, g, p, c: torch.ones((1, 1)))

    model = _mock_graph_model()
    good = ConflictPrompt(text="A plays basketball. A plays", clean_text="A plays",
                          prompt_type="substitution", domain="d",
                          memory_aliases=["soccer"], context_answer="basketball",
                          memory_answer="soccer")
    bad = ConflictPrompt(text="B plays basketball. B plays", clean_text="B plays",
                         prompt_type="substitution", domain="d",
                         memory_aliases=["pingpong"], context_answer="basketball",
                         memory_answer="pingpong")
    meta = {}
    scores = compute_edge_attribution_single_token(
        model, [good, bad], EdgeAttributionConfig(granularity="coarse"), _meta_out=meta)
    assert meta["n_prompts_used"] == 1
    assert meta["n_skipped_length_mismatch"] == 1
    assert ("input", "logits") in scores


def _mock_four_prompt_model():
    """4 distinct single-token prompts (equal-length main/cf pairs), each mapped
    to a DIFFERENT constant score so a checkpoint/resume test can prove the
    resumed run summed the exact same per-prompt contributions as an
    uninterrupted one (not just "some" score)."""
    lookup = {}
    for i in range(4):
        lookup[f"P{i} main"] = torch.tensor([[0, 1, 2, 10 + i, 4]])
        lookup[f"P{i} cf"] = torch.tensor([[0, 1, 2, 20 + i, 4]])
    return SimpleNamespace(to_tokens=lambda t: lookup[t])


def _four_prompts():
    return [
        ConflictPrompt(text=f"P{i} main", clean_text=f"P{i} clean",
                       prompt_type="substitution", domain="d",
                       memory_aliases=[f"mem{i}"], context_answer=f"ctx{i}",
                       memory_answer=f"mem{i}")
        for i in range(4)
    ]


def test_checkpoint_resume_matches_uninterrupted_run(tmp_path, monkeypatch):
    """A run interrupted mid-way (simulating a killed container) must, once
    resumed from its checkpoint, produce EXACTLY the same edge scores as an
    uninterrupted run over the same prompts -- this is the guarantee that makes
    it safe to kill and restart a multi-hour Modal job without losing GPU time."""
    import eap_ig.eap_ig_single_token as st

    fake_graph = SimpleNamespace(sources=[SimpleNamespace(label="input")],
                                 dests=[SimpleNamespace(label="logits")])
    monkeypatch.setattr(st, "configure_model_for_edges", lambda m, g: None)
    monkeypatch.setattr(st, "build_graph", lambda m, g: fake_graph)
    monkeypatch.setattr(st, "_validity_mask", lambda g: torch.ones((1, 1), dtype=torch.bool))

    def _cf_pair(model, prompt):
        i = int(prompt.text[1])
        return model.to_tokens(f"P{i} main"), model.to_tokens(f"P{i} cf")
    monkeypatch.setattr(st, "build_contrast_pair", _cf_pair)

    # Each prompt P{i} contributes a distinct score (i+1), so summing must equal
    # 1+2+3+4=10 -- proves per-prompt contributions, not just presence, survive resume.
    def _score(model, graph, prompt, config):
        i = int(prompt.text[1])
        return torch.tensor([[float(i + 1)]])
    monkeypatch.setattr(st, "score_prompt", _score)

    model = _mock_four_prompt_model()
    prompts = _four_prompts()
    cfg = EdgeAttributionConfig(granularity="coarse")

    # --- control: uninterrupted run, no checkpoint ---
    control = compute_edge_attribution_single_token(model, prompts, cfg)

    # --- interrupted run: crash after the first checkpoint (2 prompts in) ---
    ckpt_path = tmp_path / "eapig_ckpt.pt"
    call_count = {"n": 0}
    real_score = st.score_prompt

    def _score_then_crash(model, graph, prompt, config):
        call_count["n"] += 1
        if call_count["n"] > 2:
            raise RuntimeError("simulated container kill")
        return real_score(model, graph, prompt, config)
    monkeypatch.setattr(st, "score_prompt", _score_then_crash)

    with pytest.raises(RuntimeError, match="simulated container kill"):
        compute_edge_attribution_single_token(
            model, prompts, cfg, checkpoint_path=ckpt_path, checkpoint_every=2)

    assert ckpt_path.exists(), "checkpoint must survive the crash"
    ckpt = torch.load(ckpt_path)
    assert ckpt["next_start_index"] == 2
    assert ckpt["n_used"] == 2

    # --- resume: fresh call, same checkpoint_path, score_prompt no longer crashes ---
    monkeypatch.setattr(st, "score_prompt", real_score)
    resumed = compute_edge_attribution_single_token(
        model, prompts, cfg, checkpoint_path=ckpt_path, checkpoint_every=2)

    assert resumed == control, "resumed result must exactly match an uninterrupted run"
    assert not ckpt_path.exists(), "checkpoint must be deleted after a successful completion"


def test_result_json_has_single_token_provenance():
    edge_scores = {("input", "logits"): 0.5}
    meta = {"n_prompts_used": 700, "n_ig_steps": 5, "granularity": "full",
            "n_skipped_length_mismatch": 0}
    out = build_result_json_single_token(edge_scores, meta, "qwen25_3b_base",
                                         tokenizer_name="Qwen/Qwen2.5-3B")
    prov = out["_provenance"]
    assert prov["contrast"] == "inplace_swap"
    assert prov["dataset"] == f"single_token_{SET_TAG}"
    assert prov["alias_policy"] == ALIAS_POLICY
    assert prov["prompt_type"] == "substitution"
    assert prov["model_tag"] == "qwen25_3b_base"
    assert prov["n_used"] == 700
    assert out["prompt_type"] == "substitution"
    assert out["n_skipped_length_mismatch"] == 0


def test_load_prompts_reads_cache_without_model():
    """_load_prompts must reconstruct the exact cached single-token set with
    NO model load (cache-first path). Qwen base cache has 764 rows."""
    from eap_ig.run_eapig import _load_prompts, FAMILY_SPECS

    assert set(FAMILY_SPECS) == {"llama32_3b", "gemma3_4b", "qwen25_3b"}
    for spec in FAMILY_SPECS.values():
        assert set(spec) >= {"base", "instruct"}

    prompts = _load_prompts(model=None, family="qwen25_3b", variant="base")
    assert len(prompts) == 764
    assert all(p.prompt_type == "substitution" for p in prompts)
    # single-token cache: memory + context answers present as strings
    assert all(p.context_answer and p.memory_answer for p in prompts)


@pytest.fixture(scope="module")
def gpt2():
    from foundation import load_model
    return load_model("gpt2")


def _single_token_prompt():
    # "basketball" and "soccer" are each one GPT-2 token with a leading space,
    # so the swap keeps main and cf equal length -> full-window path.
    return ConflictPrompt(
        text="Lionel Messi plays the sport of basketball. Lionel Messi plays the sport of",
        clean_text="Lionel Messi plays the sport of",
        prompt_type="substitution", domain="Athlete Sport",
        memory_aliases=["soccer"], context_answer="basketball", memory_answer="soccer",
    )


@pytest.mark.slow
def test_single_token_matches_core_full_window(gpt2):
    """On an equal-length single-token pair, the single-token driver must
    produce the SAME edge scores as the multi-token core (which takes its
    full-window branch), and count zero length mismatches."""
    from eap_ig.eap_ig import compute_edge_attribution
    prompts = [_single_token_prompt()]
    cfg = EdgeAttributionConfig(n_steps=3, granularity="coarse")

    meta = {}
    st_scores = compute_edge_attribution_single_token(gpt2, prompts, cfg, _meta_out=meta)
    core_scores = compute_edge_attribution(gpt2, prompts, cfg)

    assert meta["n_skipped_length_mismatch"] == 0
    assert meta["n_prompts_used"] == 1
    assert set(st_scores) == set(core_scores)
    for k in st_scores:
        assert st_scores[k] == pytest.approx(core_scores[k], abs=1e-5)


def test_multi_token_runner_is_untouched():
    """The multi-token RUNNER (run_eap_ig.py) stays additive-only.

    NOTE: the numeric core eap_ig.py is intentionally NOT guarded here. It was
    deliberately fixed to make build_graph GQA-aware (see the two
    test_build_graph_* tests below) so that full-granularity EAP-IG runs on
    Qwen/Llama/Gemma at all -- the previous n_heads-for-everything graph crashed
    on GQA models. The GPT-2 equivalence test above proves that fix leaves MHA
    (GPT-2) behaviour byte-identical."""
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    diff = subprocess.run(
        ["git", "diff", "eapIG", "--", "eap_ig/run_eap_ig.py"],
        cwd=root, capture_output=True, text=True,
    )
    assert diff.stdout.strip() == "", (
        "Multi-token runner changed -- it must stay additive-only:\n" + diff.stdout)


def test_build_graph_gqa_uses_n_kv_heads_for_kv_inputs():
    """build_graph must create q_input destinations for all n_heads but
    k_input/v_input destinations for only n_key_value_heads (GQA). TransformerLens
    shapes hook_k_input/hook_v_input with n_kv_heads, so iterating them over
    n_heads (as for MHA) makes score_prompt slice a nonexistent KV head and
    IndexError on Qwen (16q/2kv), Llama (24q/8kv), Gemma (GQA)."""
    from eap_ig.eap_ig import build_graph
    model = SimpleNamespace(cfg=SimpleNamespace(n_layers=2, n_heads=16, n_key_value_heads=2))
    graph = build_graph(model, "full")

    def count(letter):
        return sum(1 for n in graph.dests if f"hook_{letter}_input" in n.hook)

    assert count("q") == 2 * 16   # n_layers * n_heads
    assert count("k") == 2 * 2    # n_layers * n_kv_heads
    assert count("v") == 2 * 2
    kv_heads = [n.head for n in graph.dests
                if "hook_k_input" in n.hook or "hook_v_input" in n.hook]
    assert max(kv_heads) == 1     # only heads 0..n_kv_heads-1 exist


def test_build_graph_mha_unchanged():
    """MHA (n_key_value_heads is None) must be identical to before the GQA fix --
    k/v get n_heads endpoints, so GPT-2's graph and results are unaffected."""
    from eap_ig.eap_ig import build_graph
    model = SimpleNamespace(cfg=SimpleNamespace(n_layers=2, n_heads=12, n_key_value_heads=None))
    graph = build_graph(model, "full")
    for letter in ("q", "k", "v"):
        assert sum(1 for n in graph.dests if f"hook_{letter}_input" in n.hook) == 2 * 12
