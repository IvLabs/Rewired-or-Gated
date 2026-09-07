"""Tests for run_eap_ig.py's model-loading error handling."""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest

from eap_ig.run_eap_ig import _is_unsupported_arch_error, analyse, load_hooked_model


def test_is_unsupported_arch_error_excludes_auth_failures():
    """A missing/invalid HF_TOKEN triggers huggingface_hub.RepositoryNotFoundError,
    which transformers wraps as '...is not a local folder and is not a valid
    model identifier...' -- this contains the arch-unsupported marker "not a
    valid", but must NOT be classified as an arch problem, since a plain auth
    failure needs to propagate untouched with its own actionable message."""
    real_missing_token_message = (
        "meta-llama/Llama-3.2-3B is not a local folder and is not a valid "
        "model identifier listed on 'https://huggingface.co/models'\n"
        "If this is a private repository, make sure to pass a token having "
        "permission to this repo."
    )
    assert _is_unsupported_arch_error(RuntimeError(real_missing_token_message)) is False

    # A genuine gated-repo access failure must also propagate, not be lumped
    # into "unsupported architecture".
    assert _is_unsupported_arch_error(
        RuntimeError("You are trying to access a gated repo.")
    ) is False

    # A real "TL doesn't know this architecture" message must still be
    # correctly classified as True (this is the one case that SHOULD fall
    # back to the HF bridge).
    assert _is_unsupported_arch_error(
        RuntimeError("this architecture is not officially supported by TransformerLens")
    ) is True


def test_load_hooked_model_reports_the_real_bridge_failure(monkeypatch):
    """When native TL loading fails as 'unsupported', and the HF-bridge
    fallback ALSO fails for a reason that happens to match the same keyword
    patterns (e.g. a message containing 'not supported'), the final error
    must mention the HF-bridge's actual failure, not just the native one."""
    def _fake_load_model(*a, **k):
        raise RuntimeError("architecture is not officially supported by native TL")

    def _fake_auto_model_from_pretrained(*a, **k):
        raise RuntimeError("gated repo: access not supported for this token")

    # `load_hooked_model` does `from foundation import load_model` lazily
    # inside its own body (not a module-level import in run_eap_ig.py), so
    # patching that name on `eap_ig.run_eap_ig` itself has no effect -- the
    # local import always resolves against the live `foundation` module at
    # call time. Patch it there instead.
    import foundation
    monkeypatch.setattr(foundation, "load_model", _fake_load_model)

    import transformers
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained",
        staticmethod(_fake_auto_model_from_pretrained),
    )

    with pytest.raises(RuntimeError, match="gated repo"):
        load_hooked_model("fake/repo", dtype=None, device="cpu")


def test_analyse_skip_cache_matches_write_plain_condition(monkeypatch, tmp_path):
    """With also_plain=True (the default) and ig_steps=1, eap_*.json is never
    written (the plain-EAP _run(1, eap) call is gated on ig_steps != 1) -- the
    skip-cache `want` list must reflect that, or a re-run with the same args
    can never be considered 'already done' and gets recomputed from scratch
    on every single invocation."""
    import eap_ig.run_eap_ig as run_eap_ig_mod

    monkeypatch.setattr(run_eap_ig_mod, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run_eap_ig_mod, "load_conflict_prompts", lambda prompt_type: ["prompt"])
    monkeypatch.setattr(
        run_eap_ig_mod, "compute_edge_attribution",
        lambda model, prompts, config, _meta_out=None: {("a", "b"): 1.0},
    )
    monkeypatch.setattr(run_eap_ig_mod, "build_result_json", lambda *a, **k: {"ok": True})

    # First call with ig_steps=1: only eapig_*.json should be written.
    analyse(
        model=object(), model_key="test_model", repo="test/repo", prompt_type="substitution",
        granularity="coarse", ig_steps=1, subset=None, also_plain=True, top_k=10, force=False,
    )
    igp = tmp_path / "eapig_test_model_substitution.json"
    eap = tmp_path / "eap_test_model_substitution.json"
    assert igp.exists()
    assert not eap.exists()

    # Second call, identical args: must be recognized as already-done and
    # skipped -- prove it by making compute_edge_attribution raise if called.
    def _boom(*a, **k):
        raise AssertionError(
            "compute_edge_attribution should not be called -- skip-cache should have fired"
        )
    monkeypatch.setattr(run_eap_ig_mod, "compute_edge_attribution", _boom)

    outputs = run_eap_ig_mod.analyse(
        model=object(), model_key="test_model", repo="test/repo", prompt_type="substitution",
        granularity="coarse", ig_steps=1, subset=None, also_plain=True, top_k=10, force=False,
    )
    assert outputs == [str(igp)]
