from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from singleton_fix.provenance import build_provenance


class TestBuildProvenance:
    def test_contains_all_required_keys(self):
        prov = build_provenance(
            contrast="inplace_swap", dataset="single_token_st", n_used=731,
            alias_policy="first_single_token", prompt_type="substitution",
            model_tag="qwen25_3b_base",
        )
        for key in ("contrast", "dataset", "n_used", "alias_policy",
                    "code_commit", "prompt_type", "model_tag"):
            assert key in prov, f"missing provenance key: {key}"

    def test_values_pass_through(self):
        prov = build_provenance(
            contrast="inplace_swap", dataset="single_token_st", n_used=42,
            alias_policy="first_single_token", prompt_type="coherent",
            model_tag="gemma3_4b_instruct",
        )
        assert prov["n_used"] == 42
        assert prov["prompt_type"] == "coherent"
        assert prov["model_tag"] == "gemma3_4b_instruct"

    def test_code_commit_is_nonempty_string(self):
        prov = build_provenance(
            contrast="inplace_swap", dataset="d", n_used=1,
            alias_policy="p", prompt_type="substitution", model_tag="t",
        )
        assert isinstance(prov["code_commit"], str) and len(prov["code_commit"]) > 0
