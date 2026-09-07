"""run_lds.py — CLI front-end for LDS-as-EAP attribution.

Thin wrapper over lds_attribution.run_and_save. Computes all three flavors
(gradnorm, gradact, eap) plus a shuffled-label noise-floor control on the full
prompt set for the given model and prompt type(s).

Usage:
  # GPT-2 base, both prompt types
  python run_lds.py --model gpt2 --model-tag gpt2_base --prompt-type both

  # Fine-tuned variant sharing GPT-2 architecture
  python run_lds.py --model gpt2 \
      --hf-model vicgalle/gpt2-open-instruct-v1 \
      --model-tag gpt2_instruct \
      --prompt-type both

  # Larger model (e.g. gpt2-xl or a fine-tune on top)
  python run_lds.py --model gpt2-xl --model-tag gpt2xl_base --prompt-type substitution

Outputs (saved to results/):
  lds2_<model-tag>_<ptype>_full_{gradnorm,gradact,eap}.json
  lds2_<model-tag>_<ptype>_full_{gradnorm,gradact,eap}_shuffled.json  (unless --no-shuffle)
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from foundation import load_model, load_conflict_prompts
from lds_attribution import run_and_save, enable_gradient_checkpointing

PROMPT_TYPES = ("substitution", "coherent")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LDS-as-EAP head attribution and save results."
    )
    parser.add_argument(
        "--model", default="gpt2",
        help="TransformerLens architecture name (e.g. 'gpt2', 'gpt2-xl', 'llama-3-8b').",
    )
    parser.add_argument(
        "--hf-model", default=None,
        dest="hf_model",
        help="Optional HF repo for a fine-tune sharing the base architecture "
             "(e.g. 'vicgalle/gpt2-open-instruct-v1'). Passed as hf_model_name to load_model.",
    )
    parser.add_argument(
        "--model-tag", default=None,
        dest="model_tag",
        help="Output filename tag (e.g. 'gpt2_base'). "
             "Defaults to the --model value with '/' replaced by '_'.",
    )
    parser.add_argument(
        "--prompt-type", default="both",
        choices=["substitution", "coherent", "both"],
        dest="prompt_type",
        help="Which prompt type(s) to run. Default: both.",
    )
    parser.add_argument(
        "--no-shuffle", action="store_true",
        help="Skip the shuffled-label noise-floor control run.",
    )
    parser.add_argument(
        "--n-subset", type=int, default=None,
        dest="n_subset",
        help="If set, run on a random N-prompt subset (useful for smoke tests).",
    )
    parser.add_argument(
        "--dtype", default="bfloat16",
        help="Model dtype: 'bfloat16' (default, recommended for 3-4B+) or 'float32' (GPT-2).",
    )
    args = parser.parse_args()

    model_tag = args.model_tag or args.model.replace("/", "_")
    prompt_types = PROMPT_TYPES if args.prompt_type == "both" else (args.prompt_type,)
    run_shuffle = not args.no_shuffle

    print(f"[run_lds] Loading model: arch={args.model!r}, hf_model={args.hf_model!r}, tag={model_tag!r}, dtype={args.dtype!r}")
    kwargs = {}
    if args.hf_model:
        kwargs["hf_model_name"] = args.hf_model
    model = load_model(args.model, dtype=args.dtype, **kwargs)
    enable_gradient_checkpointing(model)

    for ptype in prompt_types:
        prompts = load_conflict_prompts(prompt_type=ptype)
        if args.n_subset is not None:
            import random as _random
            rng = _random.Random(42)
            prompts = rng.sample(prompts, min(args.n_subset, len(prompts)))
            print(f"[run_lds] Using subset of {len(prompts)} prompts for {ptype}.")

        run_and_save(
            model, prompts,
            model_tag=model_tag,
            prompt_type=ptype,
            set_tag="full",
            run_shuffle=run_shuffle,
        )

    print("[run_lds] Done.")


if __name__ == "__main__":
    main()
