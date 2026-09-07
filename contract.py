"""Shared contract for the prototype phase.

Every method (LDS, path patching, EAP-IG, superposition, ablation) consumes
ConflictPrompt and produces HeadScores or EdgeScores. Drift here = silent
disagreement across methods at integration time.

ConflictPrompt no longer stores single-int token ids (GPT-2-specific). Scoring
targets (memory_answer, context_answer strings) are tokenized by each method at
run time using the loaded model's own tokenizer — making the contract
tokenizer-agnostic across all model families.
"""
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple


@dataclass(frozen=True)
class ConflictPrompt:
    # --- text ---
    text: str                       # the CONFLICT prompt (substitution or coherent)
    clean_text: str                 # the NO-CONFLICT prompt (ParaConflict "Clean Prompt")
    prompt_type: Literal["substitution", "coherent"]
    domain: str                     # ParaConflict "Category" — for held-out transfer

    # --- answers (string-based; token ids computed per model at scoring time) ---
    memory_aliases: List[str]       # full alias list, e.g. ["soccer", "football", ...]
    context_answer: str             # the injected distractor, e.g. "basketball"
    memory_answer: str              # canonical memory answer = memory_aliases[0]


# Every NODE attribution method (LDS, path patching) returns this shape.
HeadScores = Dict[Tuple[int, int], float]      # (layer, head) -> importance

# Every EDGE attribution method (EAP-IG) returns this shape.
EdgeScores = Dict[Tuple[str, str], float]      # (src_hook, dst_hook) -> importance
