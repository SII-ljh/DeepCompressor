#!/bin/bash
# Unified training script for varying Q values.
# Uses auto batch size detection — no manual batch/accum tuning needed.
#
# Usage:
#   bash scripts/train_varying_q.sh                    # run all Q values (64 128 256 512 1024 2048)
#   bash scripts/train_varying_q.sh 64 256 1024        # run only specified Q values
#   bash scripts/train_varying_q.sh --dry-run 64       # print command without executing
#   bash scripts/train_varying_q.sh --dry-run           # dry-run all Q values
#
# Environment variables:
#   NGPUS=4                     # number of GPUs (default: 8)
#   TARGET_EBS=128              # target effective batch size (default: 256)
#   EPOCHS=3                    # epoch-based training (default: 0 = use max_steps from config)
#   MAX_EVAL_SAMPLES=1000       # limit eval samples (default: 5000)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Parse flags ──────────────────────────────────────────────────────────────
DRY_RUN=false
Q_VALUES=()

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        *)         Q_VALUES+=("$arg") ;;
    esac
done

if [ ${#Q_VALUES[@]} -eq 0 ]; then
    Q_VALUES=(64 128 256 512 1024 2048)
fi

# ── Common settings ──────────────────────────────────────────────────────────
NUM_GPUS="${NGPUS:-8}"
TARGET_EBS="${TARGET_EBS:-256}"
EPOCHS="${EPOCHS:-0}"
DATA_PATH="${DATA_PATH:-data/qa_large_train.json}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-data/qa_large_dev.json}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-5000}"

# Supported Q values (for validation)
SUPPORTED_Q="64 128 256 512 1024 2048"

# ── Data check ───────────────────────────────────────────────────────────────
if [ "$DRY_RUN" = false ]; then
    for f in "$DATA_PATH" "$EVAL_DATA_PATH"; do
        if [ ! -f "$f" ]; then
            echo "Error: Data not found at $f"
            echo "Please run: python scripts/prepare_large_qa_data.py"
            exit 1
        fi
    done
fi

# ── Run ──────────────────────────────────────────────────────────────────────
declare -A RESULTS
declare -A DURATIONS
FAILED=0
SUCCEEDED=0

echo ""
echo "======================================================================"
echo "  Deep Compressor — Varying Q Training (${NUM_GPUS} GPUs, auto batch)"
echo "======================================================================"
echo "  Q values:   ${Q_VALUES[*]}"
echo "  Target EBS: $TARGET_EBS"
echo "  Epochs:     ${EPOCHS:-max_steps from config}"
echo "  Dry run:    $DRY_RUN"
echo "  Start:      $(date)"
echo "======================================================================"
echo ""

for Q in "${Q_VALUES[@]}"; do
    # Validate Q value
    if ! echo "$SUPPORTED_Q" | grep -qw "$Q"; then
        echo "[Q=$Q] Unknown Q value (supported: $SUPPORTED_Q) — SKIPPED"
        RESULTS[$Q]="SKIPPED (unknown Q)"
        ((FAILED++))
        continue
    fi

    CONFIG="configs/qa_q${Q}_8gpu.yaml"
    OUTPUT_DIR="outputs/qa_q${Q}_8gpu"
    RATIO=$((4096 / Q))

    echo "----------------------------------------------------------------------"
    echo "[Q=$Q] Compression ${RATIO}:1 | auto batch | target_ebs=${TARGET_EBS}"
    echo "----------------------------------------------------------------------"

    if [ "$DRY_RUN" = false ] && [ ! -f "$CONFIG" ]; then
        echo "[Q=$Q] Config not found: $CONFIG"
        RESULTS[$Q]="SKIPPED (config not found)"
        ((FAILED++))
        continue
    fi

    CMD="accelerate launch \
    --multi_gpu \
    --num_processes $NUM_GPUS \
    --mixed_precision bf16 \
    -m deep_compressor.train \
    --config $CONFIG \
    --data_path $DATA_PATH \
    --eval_data_path $EVAL_DATA_PATH \
    --max_eval_samples $MAX_EVAL_SAMPLES \
    --auto_batch_size \
    --target_effective_batch_size $TARGET_EBS \
    --wandb --wandb_project deep-compressor --wandb_offline"

    # Add epochs if specified
    if [ "$EPOCHS" -gt 0 ] 2>/dev/null; then
        CMD="$CMD --epochs $EPOCHS"
    fi

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would execute:"
        echo "  export WANDB_MODE=offline"
        echo "  $CMD"
        echo "  Output: $OUTPUT_DIR"
        echo ""
        RESULTS[$Q]="DRY RUN"
        ((SUCCEEDED++))
        continue
    fi

    mkdir -p "$OUTPUT_DIR"
    export WANDB_MODE=offline
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    START_SEC=$SECONDS
    set +e
    eval "$CMD" 2>&1 | tee "${OUTPUT_DIR}_training.log"
    EXIT_CODE=${PIPESTATUS[0]}
    set -e
    ELAPSED=$(( SECONDS - START_SEC ))
    HOURS=$(( ELAPSED / 3600 ))
    MINS=$(( (ELAPSED % 3600) / 60 ))
    DURATIONS[$Q]="${HOURS}h${MINS}m"

    if [ $EXIT_CODE -eq 0 ]; then
        RESULTS[$Q]="OK"
        ((SUCCEEDED++))
    else
        RESULTS[$Q]="FAILED (exit $EXIT_CODE)"
        ((FAILED++))
    fi

    echo "[Q=$Q] Finished: ${RESULTS[$Q]}  (${DURATIONS[$Q]})"
    echo ""
done

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "                           SUMMARY"
echo "======================================================================"
printf "  %-8s %-12s %-30s\n" "Q" "Duration" "Status"
echo "----------------------------------------------------------------------"
for Q in "${Q_VALUES[@]}"; do
    printf "  %-8s %-12s %-30s\n" "$Q" "${DURATIONS[$Q]:---}" "${RESULTS[$Q]}"
done
echo "----------------------------------------------------------------------"
echo "  Succeeded: $SUCCEEDED / ${#Q_VALUES[@]}    Failed: $FAILED / ${#Q_VALUES[@]}"
echo "  End: $(date)"
echo "======================================================================"

exit $FAILED
