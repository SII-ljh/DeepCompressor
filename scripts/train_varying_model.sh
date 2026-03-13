#!/bin/bash
# Unified training script for varying Qwen model sizes. All models use Q=512.
# Uses auto batch size detection — no manual batch/accum tuning needed.
#
# Usage:
#   bash scripts/train_varying_model.sh                # run all models (0.6b 1.7b 4b 8b)
#   bash scripts/train_varying_model.sh 0.6b 4b        # run only specified models
#   bash scripts/train_varying_model.sh --dry-run 0.6b  # print command without executing
#   bash scripts/train_varying_model.sh --dry-run        # dry-run all models
#
# Environment variables:
#   NGPUS=4                     # number of GPUs (default: 8)
#   TARGET_EBS=128              # target effective batch size (default: 256)
#   EPOCHS=5                    # epoch-based training (default: 3)
#   MAX_EVAL_SAMPLES=1000       # limit eval samples (default: 5000)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Parse flags ──────────────────────────────────────────────────────────────
DRY_RUN=false
MODELS=()

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        *)         MODELS+=("$arg") ;;
    esac
done

if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS=(0.6b 1.7b 4b 8b)
fi

# ── Model display names ─────────────────────────────────────────────────────
get_display_name() {
    local model=$1
    case $model in
        0.6b) NAME="Qwen3-0.6B" ;;
        1.7b) NAME="Qwen3-1.7B" ;;
        4b)   NAME="Qwen3-4B"   ;;
        8b)   NAME="Qwen3-8B"   ;;
        *)    echo "Unknown model: $model (supported: 0.6b 1.7b 4b 8b)"; return 1 ;;
    esac
}

# ── Common settings ──────────────────────────────────────────────────────────
NUM_GPUS="${NGPUS:-8}"
TARGET_EBS="${TARGET_EBS:-256}"
EPOCHS="${EPOCHS:-3}"
Q_VALUE=512
DATA_PATH="${DATA_PATH:-data/qa_large_train.json}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-data/qa_large_dev.json}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-1000}"

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
echo "  Deep Compressor — Multi-Model QA Training (Q=$Q_VALUE, ${NUM_GPUS} GPUs, auto batch)"
echo "======================================================================"
echo "  Models:     ${MODELS[*]}"
echo "  Q value:    $Q_VALUE (fixed)"
echo "  Target EBS: $TARGET_EBS"
echo "  Epochs:     ${EPOCHS:-max_steps from config}"
echo "  Dry run:    $DRY_RUN"
echo "  Start:      $(date)"
echo "======================================================================"
echo ""

for MODEL in "${MODELS[@]}"; do
    if ! get_display_name "$MODEL"; then
        RESULTS[$MODEL]="SKIPPED (unknown model)"
        ((FAILED++))
        continue
    fi

    CONFIG="configs/qa_qwen${MODEL}_q${Q_VALUE}_8gpu.yaml"
    OUTPUT_DIR="outputs/qa_qwen${MODEL}_q${Q_VALUE}_8gpu"

    echo "----------------------------------------------------------------------"
    echo "[$NAME] auto batch | target_ebs=${TARGET_EBS} | Q=$Q_VALUE"
    echo "----------------------------------------------------------------------"

    if [ "$DRY_RUN" = false ] && [ ! -f "$CONFIG" ]; then
        echo "[$NAME] Config not found: $CONFIG"
        RESULTS[$MODEL]="SKIPPED (config not found)"
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
        RESULTS[$MODEL]="DRY RUN"
        ((SUCCEEDED++))
        continue
    fi

    mkdir -p "$OUTPUT_DIR"
    export WANDB_MODE=offline
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    START_SEC=$SECONDS
    set +e
    eval "$CMD" 2>&1 | tee "${OUTPUT_DIR}/training.log"
    EXIT_CODE=${PIPESTATUS[0]}
    set -e
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

    echo "[$NAME] Finished: ${RESULTS[$MODEL]}  (${DURATIONS[$MODEL]})"
    echo ""
done

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "                           SUMMARY"
echo "======================================================================"
printf "  %-14s %-12s %-30s\n" "Model" "Duration" "Status"
echo "----------------------------------------------------------------------"
for MODEL in "${MODELS[@]}"; do
    get_display_name "$MODEL" 2>/dev/null
    printf "  %-14s %-12s %-30s\n" "${NAME:-$MODEL}" "${DURATIONS[$MODEL]:---}" "${RESULTS[$MODEL]}"
done
echo "----------------------------------------------------------------------"
echo "  Succeeded: $SUCCEEDED / ${#MODELS[@]}    Failed: $FAILED / ${#MODELS[@]}"
echo "  End: $(date)"
echo "======================================================================"

exit $FAILED
