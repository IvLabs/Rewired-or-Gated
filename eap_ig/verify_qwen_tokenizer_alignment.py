"""Empirically test whether Qwen2.5-3B's tokenizer exhibits the same
leading-space-sensitive BPE splitting that broke the old GPT-2 alignment code.

Background
----------
An earlier (now-fixed, on branch `eapIG`) version of the EAP-IG alignment
logic computed:

    L_min = min(len(main_tokens), len(clean_tokens))

and then assumed `main_tokens[-L_min:] == clean_tokens[-L_min:]` -- i.e. that
the LAST L_min tokens of the "conflict" text and the "clean" text were the
same words. This is wrong for GPT-2's BPE tokenizer because the same word
tokenizes differently depending on whether it is the very first token of a
standalone string (no leading space, e.g. "Lionel" -> ["L","ion","el"]) vs.
mid-sentence (leading space, e.g. " Lionel" -> [" Lionel"]). On 10 real
ParaConflict rows, the naive last-N-tokens alignment failed on all 10 under
GPT-2's tokenizer.

This script checks whether Qwen2.5-3B's tokenizer (a different byte-level BPE
vocabulary, used for the team's actual experiments) has the same failure mode,
rather than assuming a teammate's "GPT-2 quirk" claim is correct.

Usage:
    cd BBNLP
    python -m eap_ig.verify_qwen_tokenizer_alignment
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-3B"


def step1_direct_mechanism_test(tokenizer, out):
    """For each test word, tokenize it as the first word of a standalone
    string (no leading space) vs. mid-sentence (leading space) and compare."""
    out.append("=" * 78)
    out.append("STEP 1 -- Direct mechanism test: does Qwen split words differently")
    out.append("          based on leading space / sentence position?")
    out.append("=" * 78)

    words = ["Lionel", "Ronaldo", "basketball", "football"]
    any_differ = False
    for w in words:
        standalone = f"{w} Messi plays the sport of soccer."
        mid = f"Someone scored a goal. {w} Messi plays the sport of soccer."

        ids_standalone = tokenizer.encode(standalone, add_special_tokens=False)
        ids_mid = tokenizer.encode(mid, add_special_tokens=False)

        # Grab just the sub-token-ids that correspond to the word `w` itself.
        # For `standalone`, w is the first word -> first k tokens until we
        # reach a token that decodes to something starting with the next
        # word boundary. We instead just show a fixed-size window around the
        # word for clarity, since word boundaries are exactly what we're
        # testing.
        n_show_standalone = min(5, len(ids_standalone))
        n_show_mid = min(8, len(ids_mid))
        standalone_window_ids = ids_standalone[:n_show_standalone]
        mid_window_ids = ids_mid[:n_show_mid]

        standalone_window_toks = [tokenizer.decode([i]) for i in standalone_window_ids]
        mid_window_toks = [tokenizer.decode([i]) for i in mid_window_ids]

        out.append(f"\nWord: {w!r}")
        out.append(f"  standalone (position 0, no leading space): {standalone!r}")
        out.append(f"    token ids  : {standalone_window_ids}")
        out.append(f"    token strs : {standalone_window_toks}")
        out.append(f"  mid-sentence (preceded by '. ', leading space): {mid!r}")
        out.append(f"    token ids  : {mid_window_ids}")
        out.append(f"    token strs : {mid_window_toks}")

        # Does the standalone encoding of the word match a substring of the
        # mid-sentence encoding? Compare token-id sequences for the word span.
        # standalone: the word occupies ids_standalone[0 : k] where k is the
        # number of tokens before " Messi" appears (i.e. before the shared
        # continuation). We detect this by finding where " Messi" first
        # appears in each decoded stream.
        def word_token_span(ids):
            for k in range(1, len(ids) + 1):
                if tokenizer.decode([ids[k - 1]]).strip() == "" :
                    continue
                # stop once decoding ids[:k] followed by decode of next token
                # reaches " Messi"
                if tokenizer.decode([ids[k]]) == " Messi":
                    return ids[:k]
            return ids

        standalone_word_ids = word_token_span(ids_standalone)
        mid_word_ids = word_token_span(ids_mid)

        match = standalone_word_ids == mid_word_ids
        if not match:
            any_differ = True
        out.append(f"  standalone word-span ids : {standalone_word_ids} "
                    f"-> {[tokenizer.decode([i]) for i in standalone_word_ids]}")
        out.append(f"  mid-sentence word-span ids: {mid_word_ids} "
                    f"-> {[tokenizer.decode([i]) for i in mid_word_ids]}")
        out.append(f"  SAME token-ids for the word in both positions? {match}")

    out.append("")
    out.append(f"Step 1 summary: at least one word tokenized differently "
                f"depending on position/leading-space = {any_differ}")
    return any_differ


def step2_reproduce_original_test(tokenizer, out, n=10):
    """Load 10 real ParaConflict rows and apply the exact old (buggy)
    last-L_min-tokens alignment formula using Qwen's tokenizer."""
    out.append("")
    out.append("=" * 78)
    out.append("STEP 2 -- Reproduce the exact original alignment test with Qwen's tokenizer")
    out.append("=" * 78)

    print("[verify] loading gaotang/ParaConflict test split...")
    ds = load_dataset("gaotang/ParaConflict", split="test")

    rows = []
    for row in ds:
        text = row.get("Substitution Conflict")
        clean_text = row.get("Clean Prompt")
        if not text or not clean_text:
            continue
        if not str(text).strip() or not str(clean_text).strip():
            continue
        rows.append(row)
        if len(rows) >= n:
            break

    n_pass = 0
    n_fail = 0
    detail_pass_shown = 0
    detail_fail_shown = 0

    for i, row in enumerate(rows):
        text = row["Substitution Conflict"]
        clean_text = row["Clean Prompt"]

        # add_special_tokens=False: Qwen2 has no BOS token by default, and we
        # want a clean, consistent comparison of the raw word-piece ids
        # (mirrors the original GPT-2 test's use of tokenizer.encode()).
        main_tokens = tokenizer.encode(text, add_special_tokens=False)
        clean_tokens = tokenizer.encode(clean_text, add_special_tokens=False)

        L_min = min(len(main_tokens), len(clean_tokens))
        main_suffix = main_tokens[-L_min:]
        clean_suffix = clean_tokens[-L_min:]
        aligned = main_suffix == clean_suffix

        status = "PASS" if aligned else "FAIL"
        if aligned:
            n_pass += 1
        else:
            n_fail += 1

        out.append(f"\n--- row {i} (Category={row.get('Category')!r}) ---")
        out.append(f"TEXT (conflict): {text!r}")
        out.append(f"CLEAN_TEXT:      {clean_text!r}")
        out.append(f"L_main={len(main_tokens)}  L_clean={len(clean_tokens)}  L_min={L_min}")
        out.append(f"RESULT: {status}")

        show_detail = (aligned and detail_pass_shown < 3) or (not aligned and detail_fail_shown < 3)
        if show_detail:
            main_toks_str = [tokenizer.decode([t]) for t in main_tokens]
            clean_toks_str = [tokenizer.decode([t]) for t in clean_tokens]
            out.append(f"  main tokens  : {main_toks_str}")
            out.append(f"  clean tokens : {clean_toks_str}")
            out.append(f"  main[-L_min:]  : {[tokenizer.decode([t]) for t in main_suffix]}")
            out.append(f"  clean[-L_min:] : {[tokenizer.decode([t]) for t in clean_suffix]}")
            if aligned:
                detail_pass_shown += 1
            else:
                detail_fail_shown += 1

    out.append("")
    out.append(f"Step 2 final count: {n_pass}/{len(rows)} PASSED (naive alignment valid), "
                f"{n_fail}/{len(rows)} FAILED (misaligned -- same bug class as GPT-2)")
    return n_pass, n_fail, len(rows)


def step3_interpretation(out, step1_differs, n_pass, n_fail, n_total):
    out.append("")
    out.append("=" * 78)
    out.append("STEP 3 -- Interpretation")
    out.append("=" * 78)
    same_bug = step1_differs or n_fail > 0
    if same_bug:
        out.append(
            "Qwen2.5-3B's tokenizer exhibits the SAME class of leading-space-sensitive "
            "behavior as GPT-2's BPE tokenizer, even though the two use completely "
            "different vocabularies. This is NOT a 'GPT-2 quirk' -- it is a structural "
            "property of byte-level BPE tokenizers in general (both GPT-2's tokenizer "
            "and Qwen's tokenizer are byte-level BPE, trained so that a leading space is "
            "folded into the token itself, e.g. ' Lionel' as one token, rather than "
            "treated as a separate whitespace token). Because whether a word is preceded "
            "by a space depends entirely on its position in the string, the BPE merge "
            "rules select a different token sequence for the *same word* depending on "
            "context. Any code that assumes 'same word => same trailing token ids, "
            "regardless of position in the string' is at risk of this bug with Qwen's "
            "tokenizer too, and the alignment must be done by finding the actual longest "
            "common prefix/suffix on token ids (or by aligning on text spans and "
            "re-tokenizing), not by naively slicing the last L_min = min(len_a, len_b) "
            "tokens."
        )
    else:
        out.append(
            "Qwen2.5-3B's tokenizer did NOT reproduce the failure: it appears genuinely "
            "immune to this class of bug in the tested cases."
        )
    return same_bug


def main():
    out = []
    print(f"[verify] loading {MODEL_NAME} tokenizer (first run downloads tokenizer files only)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    out.append(f"Tokenizer under test: {MODEL_NAME} ({type(tokenizer).__name__})")
    out.append("")

    step1_differs = step1_direct_mechanism_test(tokenizer, out)
    n_pass, n_fail, n_total = step2_reproduce_original_test(tokenizer, out, n=10)
    step3_interpretation(out, step1_differs, n_pass, n_fail, n_total)

    out_path = os.path.join(_ROOT, "results", "qwen_tokenizer_alignment_check.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")

    print(f"[verify] wrote report -> {out_path}")
    print(f"[verify] Step 2 result: {n_pass}/{n_total} PASS, {n_fail}/{n_total} FAIL")


if __name__ == "__main__":
    main()
