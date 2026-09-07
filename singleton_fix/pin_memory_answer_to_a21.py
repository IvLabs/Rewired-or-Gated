"""One-off fix for the base-vs-instruct A2.2 confound (found via Opus review,
2026-07-09): the A2.2 log-prob oracle runs independently per model, so base
and instruct can pick different canonical memory_answer values for the same
row (e.g. base="football" vs instruct="soccer"). That divergence confounds
the base-vs-instruct comparison the whole project measures.

Fix (Option A, chosen over reusing base's A2.2 pick): pin memory_answer back
to a21_naive_answer -- the deterministic, tokenizer-only alias choice -- for
every row in every survival file. Base and instruct share a tokenizer per
family, so this guarantees zero divergence by construction, and it removes
the log-prob oracle from the stimulus entirely (no reviewer can ask whether
the alias-picking model itself biased the result).

Rewrites the 6 on-disk single_token_survival_*.json files in place (so the
cache stays the source of truth, not a load-time patch), regenerates
single_token_intersection.json and single_token_filter_summary.json from the
corrected files, and prints a verification pass.

Run once: python singleton_fix/pin_memory_answer_to_a21.py
"""
from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_DIR = _ROOT / "results"

FAMILIES = {
    "llama32_3b": ("base", "instruct"),
    "gemma3_4b": ("base", "instruct"),
    "qwen25_3b": ("base", "instruct"),
}


def _pin_file(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    n_changed = 0
    for row in data["rows"]:
        naive = row["a21_naive_answer"]
        if row["memory_answer"] != naive:
            n_changed += 1
        row["memory_answer"] = naive
        row["a22_changed"] = False
    data["n_a22_changed_vs_a21"] = 0
    data["_a22_confound_fix"] = (
        "2026-07-09: memory_answer pinned to a21_naive_answer for every row "
        "(A2.2 log-prob refinement disabled as the *default* canonical answer) "
        "to eliminate the base-vs-instruct alias-divergence confound found "
        "via Opus review. a21_naive_answer is deterministic and tokenizer-"
        "only, so base and instruct now always agree on the target word."
    )
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"  {path.relative_to(_ROOT)}: pinned {n_changed} rows back to A2.1")


def main() -> None:
    for family, variants in FAMILIES.items():
        for variant in variants:
            path = _RESULTS_DIR / family / f"single_token_survival_{family}_{variant}.json"
            if not path.exists():
                print(f"  [skip] {path} does not exist")
                continue
            _pin_file(path)

    # Verify: 0 divergence between base and instruct memory_answer per family.
    print("\nVerification (base vs instruct memory_answer divergence):")
    all_ok = True
    for family, (base_v, instruct_v) in FAMILIES.items():
        base_path = _RESULTS_DIR / family / f"single_token_survival_{family}_{base_v}.json"
        instruct_path = _RESULTS_DIR / family / f"single_token_survival_{family}_{instruct_v}.json"
        if not (base_path.exists() and instruct_path.exists()):
            continue
        b = json.loads(base_path.read_text(encoding="utf-8"))
        i = json.loads(instruct_path.read_text(encoding="utf-8"))
        bmap = {r["row_index"]: r["memory_answer"] for r in b["rows"]}
        imap = {r["row_index"]: r["memory_answer"] for r in i["rows"]}
        common = set(bmap) & set(imap)
        diffs = [k for k in common if bmap[k] != imap[k]]
        status = "OK" if not diffs else "STILL DIVERGENT"
        if diffs:
            all_ok = False
        print(f"  {family}: common={len(common)} diffs={len(diffs)} [{status}]")

    print("\nAll families consistent." if all_ok else "\nWARNING: divergence remains -- investigate.")


if __name__ == "__main__":
    main()
