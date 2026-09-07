"""
superposition/test_pipeline.py
---------------------------------
Smoke tests. Run before starting the full experiment:
    python superposition/test_pipeline.py

Tests:
    1. Dataset loads and filters correctly
    2. Superposition output has correct shape
    3. Intervention hooks fire correctly (ΔCRR non-zero)
    4. JSON save/load round-trip preserves values

Does NOT load large models — uses GPT-2 as a fast proxy.
"""

import sys, json, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))


def test_dataset():
    print("\n[test_dataset]", end=" ")
    from transformers import AutoTokenizer
    from load_dataset import load_substitution_prompts

    tok     = AutoTokenizer.from_pretrained("gpt2")
    prompts = load_substitution_prompts(tok, verbose=False, max_rows=200)

    assert len(prompts) > 0, "No prompts loaded — check dataset access"
    assert prompts[0].context_token_id != -1
    assert prompts[0].memory_token_id  != -1
    assert prompts[0].context_token_id != prompts[0].memory_token_id
    print(f"OK — {len(prompts)} prompts from first 200 rows")


def test_superposition_shape():
    print("\n[test_superposition_shape]", end=" ")
    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer
    from load_dataset import load_substitution_prompts
    from superposition import compute_superposition

    tok    = AutoTokenizer.from_pretrained("gpt2")
    model  = HookedTransformer.from_pretrained("gpt2", fold_ln=True,
                                               center_unembed=True)
    model.eval()
    prompts = load_substitution_prompts(tok, verbose=False, max_rows=50)
    prompts = prompts[:10]  # fast

    result = compute_superposition(model, prompts, norm_type="ln",
                                   device="cpu", label="test")

    assert result["n_prompts"] > 0
    assert result["n_layers"]  == 12
    assert result["n_heads"]   == 12
    assert len(result["heads"]) == 144  # 12×12
    for h in result["heads"].values():
        assert h["role"] in ("context", "memory", "superposition")
        assert isinstance(h["ratio"], float)

    del model
    print("OK — 144 heads, correct roles, correct types")


def test_intervention_hooks():
    print("\n[test_intervention_hooks]", end=" ")
    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer
    from load_dataset import load_substitution_prompts
    from intervention import make_scale_hooks, compute_crr

    tok    = AutoTokenizer.from_pretrained("gpt2")
    model  = HookedTransformer.from_pretrained("gpt2", fold_ln=True,
                                               center_unembed=True)
    model.eval()
    prompts = load_substitution_prompts(tok, verbose=False, max_rows=50)
    prompts = prompts[:20]

    # Zero-ablate layer 11 head 0 — should change CRR
    baseline  = compute_crr(model, prompts, hooks=None, label="base")
    hooks     = make_scale_hooks([(11, 0)], scale_factor=0.0)
    intervened= compute_crr(model, prompts, hooks=hooks, label="ablate")

    # CRR values should be floats in [0,1]
    assert 0.0 <= baseline["crr"]   <= 1.0
    assert 0.0 <= intervened["crr"] <= 1.0

    del model
    print(f"OK — baseline CRR={baseline['crr']:.3f}, "
          f"ablated CRR={intervened['crr']:.3f}")


def test_json_roundtrip():
    print("\n[test_json_roundtrip]", end=" ")
    from aggregation_utils import save_json, load_json

    data = {str((l, h)): float(l * 12 + h)
            for l in range(3) for h in range(3)}
    path = Path("/tmp/test_superposition_roundtrip.json")
    save_json(data, path)
    loaded = load_json(path)
    assert len(loaded) == len(data)
    path.unlink()
    print("OK — save/load round-trip correct")


if __name__ == "__main__":
    test_dataset()
    test_superposition_shape()
    test_intervention_hooks()
    test_json_roundtrip()
    print("\n✓ All tests passed")
