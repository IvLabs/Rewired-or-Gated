#!/usr/bin/env bash
# run_superposition_all.sh
# Runs superposition analysis + visualisation for all three families,
# then organises outputs into results_superposition/{qwen,llama,gemma}/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="$SCRIPT_DIR/results_superposition"
SUPER_DIR="$SCRIPT_DIR/superposition"

export HF_TOKEN="${HF_TOKEN:-}"

cd "$SCRIPT_DIR"

# Strip tqdm \r noise + HTTP INFO lines from a tee'd log file
clean_log() {
    local src="$1" dst="$2"
    python3 - "$src" "$dst" <<'PYEOF'
import sys, re
src, dst = sys.argv[1], sys.argv[2]
with open(src, "rb") as f:
    raw = f.read()
lines = raw.decode("utf-8", errors="replace").split("\n")
clean = []
for line in lines:
    seg = line.split("\r")[-1]          # keep last \r segment (final tqdm state)
    if re.search(r"INFO\s+HTTP Request:", seg): continue
    if "unauthenticated requests" in seg:       continue
    if seg.strip():
        clean.append(seg)
with open(dst, "w") as f:
    f.write("\n".join(clean) + "\n")
print(f"  [log] {len(lines)} lines → {len(clean)} lines  ({len(raw)//1024} KB → {len(''.join(clean))//1024} KB)")
PYEOF
}

run_family() {
    local FAM="$1"
    echo ""
    echo "================================================================="
    echo "  RUNNING: $FAM"
    echo "================================================================="

    # Analysis
    python3 "$SUPER_DIR/superposition_analysis.py" --family "$FAM" \
        2>&1 | tee "$RESULTS/logs/${FAM}_run.txt"

    # Visualise
    python3 "$SUPER_DIR/visualize_superposition.py" --family "$FAM" \
        2>&1 | tee -a "$RESULTS/logs/${FAM}_run.txt"

    # Organise into per-family subfolder
    echo "[organise] Moving $FAM results to $RESULTS/$FAM/"
    mkdir -p "$RESULTS/$FAM/plots" "$RESULTS/$FAM/logs"

    # JSONs
    for f in "$RESULTS/superposition_${FAM}"_*.json \
              "$RESULTS/intervention_${FAM}"_*.json; do
        [ -f "$f" ] && mv "$f" "$RESULTS/$FAM/"
    done

    # Plots
    for f in "$RESULTS/plots/scatter_${FAM}.png" \
              "$RESULTS/plots/ratio_bars_${FAM}_base.png" \
              "$RESULTS/plots/ratio_bars_${FAM}_instruct.png" \
              "$RESULTS/plots/delta_ratio_${FAM}.png" \
              "$RESULTS/plots/intervention_crr_${FAM}.png"; do
        [ -f "$f" ] && cp "$f" "$RESULTS/$FAM/plots/"
    done

    # Log — clean it first (strip tqdm/HTTP noise), then store in family folder
    RAW_LOG="$RESULTS/logs/${FAM}_run.txt"
    CLEAN_LOG="$RESULTS/$FAM/logs/${FAM}_run.txt"
    if [ -f "$RAW_LOG" ]; then
        echo "[log] Cleaning $RAW_LOG → $CLEAN_LOG"
        clean_log "$RAW_LOG" "$CLEAN_LOG"
    fi

    echo "[organise] $FAM done."
}

# ── Run all three families ──────────────────────────────────────────────
for FAM in qwen llama gemma; do
    run_family "$FAM"
done

# ── Cross-family comparison plots ──────────────────────────────────────
echo ""
echo "================================================================="
echo "  RUNNING: cross-family comparison (--all)"
echo "================================================================="
python3 "$SUPER_DIR/visualize_superposition.py" --all 2>&1 | tee "$RESULTS/logs/all_families.txt"

# Copy cross-family plots to top-level plots/
echo "[organise] Cross-family plots already in $RESULTS/plots/"

# ── Final summary ──────────────────────────────────────────────────────
echo ""
echo "================================================================="
echo "  FINAL SUMMARY"
echo "================================================================="
python3 - <<'PYEOF'
import json
from pathlib import Path

RESULTS = Path("results_superposition")
families = ['qwen', 'llama', 'gemma']
print('=== SUPERPOSITION SUMMARY ===\n')

for fam in families:
    comp_path = RESULTS / fam / f"superposition_{fam}_comparison_substitution.json"
    int_path  = RESULTS / fam / f"intervention_{fam}_substitution.json"

    if not comp_path.exists():
        print(f'{fam.upper()}: no results yet')
        continue

    comp  = json.load(open(comp_path))
    items = [(k,v) for k,v in comp.items() if k != '_meta']
    changed = [v for k,v in items if v['role_changed']]
    shifts  = [v['ratio_shift'] for k,v in items]

    print(f'{fam.upper()}:')
    print(f'  Heads compared: {len(items)}')
    print(f'  Role changed:   {len(changed)} ({100*len(changed)/max(len(items),1):.0f}%)')
    print(f'  Mean ratio shift: {sum(shifts)/max(len(shifts),1):+.4f}')

    if int_path.exists():
        iv = json.load(open(int_path))
        print(f'  Baseline CRR: {iv["baseline_crr"]:.4f}')
        for cname, cres in iv["conditions"].items():
            marker = "OK" if cres["expected_positive"] else "WRONG"
            print(f'  [{marker}] {cname}: ΔCRR={cres["delta_crr"]:+.4f}')
    print()
PYEOF

echo ""
echo "All done. Results in: $RESULTS/{qwen,llama,gemma}/"
ls -lh "$RESULTS/"
