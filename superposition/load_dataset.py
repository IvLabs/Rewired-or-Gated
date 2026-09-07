"""
src/load_dataset.py
-------------------
Loads the gaotang/ParaConflict dataset from HuggingFace and filters rows
to those where BOTH the memory answer and context (distractor) answer
tokenize to exactly ONE token in the given model's tokenizer.

This must be run once per model family because different tokenizers
produce different token IDs for the same surface strings.

Dataset: gaotang/ParaConflict  (split="test")
Prompt type used: substitution only (column: "Substitution Conflict")

Columns used:
  "Substitution Conflict"  — the conflict prompt text
  "Clean Prompt"           — same question, no conflict (used as clean run)
  "Answer"                 — stringified list of memory answer aliases
  "Distracted Token"       — the context (conflicting) answer string

Usage:
  from src.load_dataset import load_substitution_prompts
  prompts = load_substitution_prompts(tokenizer, verbose=True)
"""

from __future__ import annotations
import ast
import logging
from dataclasses import dataclass
from datasets import load_dataset
from transformers import PreTrainedTokenizer

log = logging.getLogger(__name__)


# ── Data container ─────────────────────────────────────────────────────────────

@dataclass
class ConflictRow:
    """One usable row from ParaConflict (substitution type)."""
    row_idx:          int    # original dataset index for traceability
    prompt_conflict:  str    # substitution conflict prompt text
    prompt_clean:     str    # clean prompt (no conflict)
    memory_answer:    str    # string surface form of the memory answer
    context_answer:   str    # string surface form of the context answer
    memory_token_id:  int    # vocab id in this tokenizer
    context_token_id: int    # vocab id in this tokenizer
    domain:           str    # Category column (e.g. "World Capital")


# ── Token helpers ──────────────────────────────────────────────────────────────

def _single_token_id(text: str, tokenizer: PreTrainedTokenizer) -> int | None:
    """
    Return the single vocab token ID for `text`, or None if multi-token.

    Tries two variants:
      1. " " + text.strip()  (leading space — standard for mid-sentence tokens)
      2. text.strip()        (no leading space — catches BOS-position tokens)

    Returns the first variant that encodes to exactly one token.
    """
    for variant in (" " + text.strip(), text.strip()):
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    return None


def _best_token_from_answer_field(
    answer_field: str,
    tokenizer:    PreTrainedTokenizer,
) -> tuple[int, str] | tuple[None, None]:
    """
    The Answer column is a Python-stringified list of aliases, e.g.:
      "['mixed martial arts', 'MMA', 'mma', 'Mixed Martial Arts']"

    Try each alias in order. Return (token_id, alias_string) for the
    first alias that encodes to exactly one token, or (None, None).
    """
    try:
        aliases = ast.literal_eval(answer_field)
        if isinstance(aliases, str):
            aliases = [aliases]
    except Exception:
        aliases = [answer_field]

    for alias in aliases:
        tid = _single_token_id(str(alias), tokenizer)
        if tid is not None:
            return tid, str(alias)
    return None, None


# ── Main loader ────────────────────────────────────────────────────────────────

def load_substitution_prompts(
    tokenizer: PreTrainedTokenizer,
    verbose:   bool = True,
    max_rows:  int | None = None,
) -> list[ConflictRow]:
    """
    Load ParaConflict substitution prompts, filtered to single-token answers.

    Parameters
    ----------
    tokenizer : The model family's tokenizer. Token IDs are family-specific.
    verbose   : Print progress and filter statistics.
    max_rows  : Cap the number of rows to process (useful for debugging).

    Returns
    -------
    List[ConflictRow] — ready for superposition and intervention analysis.

    Filter criteria (a row is KEPT only if ALL of these hold):
      1. "Substitution Conflict" column is non-empty
      2. "Clean Prompt" column is non-empty
      3. "Answer" field contains at least one alias that is single-token
      4. "Distracted Token" is single-token
      5. memory_token_id != context_token_id  (otherwise no conflict)
    """
    if verbose:
        print("[load_dataset] Loading gaotang/ParaConflict (split=test)...")

    ds = load_dataset("gaotang/ParaConflict", split="test")

    if verbose:
        print(f"[load_dataset] Total rows: {len(ds)}")
        print(f"[load_dataset] Columns: {ds.column_names}")

    kept:    list[ConflictRow] = []
    skipped: dict[str, int]   = {
        "empty_prompt":      0,
        "empty_clean":       0,
        "memory_multi_token":0,
        "context_multi_token":0,
        "same_token":        0,
    }

    iterable = ds if max_rows is None else ds.select(range(min(max_rows, len(ds))))

    for idx, row in enumerate(iterable):
        # -- Extract fields ------------------------------------------
        conflict_text = str(row.get("Substitution Conflict", "") or "").strip()
        clean_text    = str(row.get("Clean Prompt",          "") or "").strip()
        answer_field  = str(row.get("Answer",                "") or "").strip()
        distract_text = str(row.get("Distracted Token",      "") or "").strip()
        domain        = str(row.get("Category",              "") or "").strip()

        # -- Filter 1: non-empty text --------------------------------
        if not conflict_text:
            skipped["empty_prompt"] += 1
            continue
        if not clean_text:
            skipped["empty_clean"] += 1
            continue

        # -- Filter 2: single-token memory answer --------------------
        mid, mem_surface = _best_token_from_answer_field(answer_field, tokenizer)
        if mid is None:
            skipped["memory_multi_token"] += 1
            continue

        # -- Filter 3: single-token context answer -------------------
        did = _single_token_id(distract_text, tokenizer)
        if did is None:
            skipped["context_multi_token"] += 1
            continue

        # -- Filter 4: different token IDs ---------------------------
        if mid == did:
            skipped["same_token"] += 1
            continue

        kept.append(ConflictRow(
            row_idx          = idx,
            prompt_conflict  = conflict_text,
            prompt_clean     = clean_text,
            memory_answer    = mem_surface,
            context_answer   = distract_text,
            memory_token_id  = mid,
            context_token_id = did,
            domain           = domain,
        ))

    if verbose:
        print(f"[load_dataset] Kept: {len(kept)}  |  Skipped: {sum(skipped.values())}")
        for reason, count in skipped.items():
            if count > 0:
                print(f"               {reason}: {count}")
        # Domain breakdown
        from collections import Counter
        domains = Counter(r.domain for r in kept)
        print(f"[load_dataset] Domain breakdown:")
        for domain, count in domains.most_common():
            print(f"               {domain}: {count}")

    return kept
