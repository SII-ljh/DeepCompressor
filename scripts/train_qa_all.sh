#!/bin/bash
# Run QA training experiments sequentially.
# Each Q value runs independently — one failure won't stop the rest.
#
# Usage:
#   bash scripts/train_qa_all.sh              # run all Q values
#   bash scripts/train_qa_all.sh 64 256 1024  # run only specified Q values

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ $# -gt 0 ]; then
    Q_VALUES=("$@")
else
    Q_VALUES=(64 128 256 512 1024 2048)
fi

declare -A RESULTS
FAILED=0
SUCCEEDED=0

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║        Deep Compressor — QA Training All Q Values (8 GPUs)         ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║  Q values: ${Q_VALUES[*]}"
echo "║  Start:    $(date)"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

for Q in "${Q_VALUES[@]}"; do
    SCRIPT="${SCRIPT_DIR}/train_qa_q${Q}_8gpu.sh"

    echo "────────────────────────────────────────────────────────────────────"
    echo "[Q=$Q] Starting at $(date)"
    echo "────────────────────────────────────────────────────────────────────"

    if [ ! -f "$SCRIPT" ]; then
        echo "[Q=$Q] ✗ Script not found: $SCRIPT"
        RESULTS[$Q]="SKIPPED (script not found)"
        ((FAILED++))
        continue
    fi

    bash "$SCRIPT"
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        RESULTS[$Q]="OK"
        ((SUCCEEDED++))
    else
        RESULTS[$Q]="FAILED (exit $EXIT_CODE)"
        ((FAILED++))
    fi

    echo "[Q=$Q] Finished with: ${RESULTS[$Q]}"
    echo ""
done

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                          SUMMARY                                   ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
printf "║  %-8s %-58s ║\n" "Q" "Status"
echo "╠══════════════════════════════════════════════════════════════════════╣"
for Q in "${Q_VALUES[@]}"; do
    printf "║  %-8s %-58s ║\n" "$Q" "${RESULTS[$Q]}"
done
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║  Succeeded: $SUCCEEDED / ${#Q_VALUES[@]}    Failed: $FAILED / ${#Q_VALUES[@]}"
echo "║  End: $(date)"
echo "╚══════════════════════════════════════════════════════════════════════╝"

exit $FAILED
