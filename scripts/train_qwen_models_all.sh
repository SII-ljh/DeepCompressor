#!/bin/bash
# Run QA training experiments across different Qwen models, all with Q=512.
# Each model runs independently — one failure won't stop the rest.
#
# Usage:
#   bash scripts/train_qwen_models_all.sh                    # run all models
#   bash scripts/train_qwen_models_all.sh 0.6b 4b            # run only specified models
#
# Supported model keys: 0.6b, 1.7b, 4b, 8b

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ $# -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=(0.6b 1.7b 4b 8b)
fi

# Map model key -> display name
declare -A MODEL_NAMES
MODEL_NAMES[0.6b]="Qwen3-0.6B"
MODEL_NAMES[1.7b]="Qwen3-1.7B"
MODEL_NAMES[4b]="Qwen3-4B"
MODEL_NAMES[8b]="Qwen3-8B"

# Map model key -> script name suffix
declare -A SCRIPT_SUFFIX
SCRIPT_SUFFIX[0.6b]="qwen0.6b"
SCRIPT_SUFFIX[1.7b]="qwen1.7b"
SCRIPT_SUFFIX[4b]="qwen4b"
SCRIPT_SUFFIX[8b]="qwen8b"

declare -A RESULTS
declare -A DURATIONS
FAILED=0
SUCCEEDED=0

echo ""
echo "======================================================================"
echo "  Deep Compressor — Multi-Model QA Training (Q=512, 8 GPUs)"
echo "======================================================================"
echo "  Models:  ${MODELS[*]}"
echo "  Q value: 512 (fixed)"
echo "  Start:   $(date)"
echo "======================================================================"
echo ""

for MODEL in "${MODELS[@]}"; do
    SUFFIX="${SCRIPT_SUFFIX[$MODEL]}"
    NAME="${MODEL_NAMES[$MODEL]}"

    if [ -z "$SUFFIX" ]; then
        echo "[${MODEL}] Unknown model key. Supported: 0.6b, 1.7b, 4b, 8b"
        RESULTS[$MODEL]="SKIPPED (unknown model)"
        ((FAILED++))
        continue
    fi

    SCRIPT="${SCRIPT_DIR}/train_qa_${SUFFIX}_q512_8gpu.sh"

    echo "----------------------------------------------------------------------"
    echo "[${NAME}] Starting at $(date)"
    echo "----------------------------------------------------------------------"

    if [ ! -f "$SCRIPT" ]; then
        echo "[${NAME}] Script not found: $SCRIPT"
        RESULTS[$MODEL]="SKIPPED (script not found)"
        ((FAILED++))
        continue
    fi

    START_SEC=$SECONDS
    bash "$SCRIPT"
    EXIT_CODE=$?
    ELAPSED=$(( SECONDS - START_SEC ))
    HOURS=$(( ELAPSED / 3600 ))
    MINS=$(( (ELAPSED % 3600) / 60 ))

    DURATIONS[$MODEL]="${HOURS}h${MINS}m"

    if [ $EXIT_CODE -eq 0 ]; then
        RESULTS[$MODEL]="OK"
        ((SUCCEEDED++))
    else
        RESULTS[$MODEL]="FAILED (exit $EXIT_CODE)"
        ((FAILED++))
    fi

    echo "[${NAME}] Finished: ${RESULTS[$MODEL]}  (${DURATIONS[$MODEL]})"
    echo ""
done

echo ""
echo "======================================================================"
echo "                           SUMMARY"
echo "======================================================================"
printf "  %-14s %-12s %-30s\n" "Model" "Duration" "Status"
echo "----------------------------------------------------------------------"
for MODEL in "${MODELS[@]}"; do
    NAME="${MODEL_NAMES[$MODEL]:-$MODEL}"
    printf "  %-14s %-12s %-30s\n" "$NAME" "${DURATIONS[$MODEL]:---}" "${RESULTS[$MODEL]}"
done
echo "----------------------------------------------------------------------"
echo "  Succeeded: $SUCCEEDED / ${#MODELS[@]}    Failed: $FAILED / ${#MODELS[@]}"
echo "  End: $(date)"
echo "======================================================================"

exit $FAILED
