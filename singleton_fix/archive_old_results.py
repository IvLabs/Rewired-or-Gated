"""One-off archival of the pre-fix (clean_text-contrast, full-multitoken-set)
result files, per spec §A3.3. Run ONCE before any new single-token run so old
and new numbers can never be confused. Never deletes -- only moves (git mv
when inside a git repo, plain os.rename otherwise) into
results/archive_prefix_clean_text_buggy/, and refuses to overwrite an
existing archived file with the same name.

Usage:
    python singleton_fix/archive_old_results.py            # do the move
    python singleton_fix/archive_old_results.py --dry-run   # list only
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import List

_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = _ROOT / "results"
ARCHIVE_DIR = RESULTS_DIR / "archive_prefix_clean_text_buggy"

# Exact prefixes named in spec §A3.3, PLUS gemma3_4b_* (decided: same buggy
# vintage as llama32_3b_*/qwen25_3b_*, archive alongside them even though the
# spec text mistakenly said "Gemma has none").
ARCHIVE_FILE_GLOBS: List[str] = [
    "lds2_llama32_3b_*",
    "lds2_qwen25_3b_*",
    "lds2_gemma3_4b_*",
    "pp_topheads_llama32_3b_*",
    "pp_topheads_qwen25_3b_*",
    "pp_topheads_gemma3_4b_*",
    "llama32_3b_summary.json",
    "qwen25_3b_summary.json",
    "gemma3_4b_summary.json",
]


def _find_files() -> List[Path]:
    found: List[Path] = []
    for pattern in ARCHIVE_FILE_GLOBS:
        found.extend(sorted(RESULTS_DIR.glob(pattern)))
    return found


def _git_mv(src: Path, dst: Path) -> None:
    try:
        subprocess.check_call(
            ["git", "mv", str(src), str(dst)], cwd=_ROOT,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        shutil.move(str(src), str(dst))


def main(dry_run: bool = False) -> None:
    files = _find_files()
    if not files:
        print("[archive] nothing to archive (no matching files found).")
        return

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    for src in files:
        dst = ARCHIVE_DIR / src.name
        if dst.exists():
            raise FileExistsError(
                f"[archive] refusing to overwrite already-archived file: {dst}"
            )
        if dry_run:
            print(f"[archive] would move {src} -> {dst}")
        else:
            _git_mv(src, dst)
            print(f"[archive] moved {src} -> {dst}")

    print(f"[archive] {'would archive' if dry_run else 'archived'} {len(files)} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
