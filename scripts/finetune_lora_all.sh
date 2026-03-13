#!/bin/bash
# LoRA fine-tune Qwen models of various sizes on QA data (upper-bound baseline).
# Uses auto batch size detection — no manual batch/accum tuning needed.
# Runs each model size sequentially — one failure won't stop the rest.
#
# Usage:
#   bash scripts/finetune_lora_all.sh                    # all models
#   bash scripts/finetune_lora_all.sh 0.6B 1.7B          # specific sizes
#   NGPUS=4 bash scripts/finetune_lora_all.sh             # custom GPU count
#
# Environment variables:
#   NGPUS=4                     # number of GPUs (default: 8)
#   TARGET_EBS=128              # target effective batch size (default: 256)
#   NUM_EPOCHS=2                # number of epochs (default: 1)
#   MAX_EVAL_SAMPLES=1000       # limit eval samples (default: 5000)

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configurable
NGPUS="${NGPUS:-8}"
TARGET_EBS="${TARGET_EBS:-256}"
DATA_PATH="${DATA_PATH:-data/qa_large_train.json}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-data/qa_large_dev.json}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-5000}"

# Model registry
declare -A MODEL_PATHS
MODEL_PATHS["0.6B"]="models/Qwen3-0.6B"
MODEL_PATHS["1.7B"]="models/Qwen3-1.7B"
MODEL_PATHS["4B"]="models/Qwen3-4B"
MODEL_PATHS["8B"]="models/Qwen3-8B"

# Eval batch sizes (generation needs more memory than training due to KV cache)
declare -A EVAL_BATCH_SIZES
EVAL_BATCH_SIZES["0.6B"]=8
EVAL_BATCH_SIZES["1.7B"]=5
EVAL_BATCH_SIZES["4B"]=2
EVAL_BATCH_SIZES["8B"]=1

if [ $# -gt 0 ]; then
    SIZES=("$@")
else
    SIZES=("0.6B" "1.7B" "4B" "8B")
fi

declare -A RESULTS
declare -A DURATIONS
FAILED=0
SUCCEEDED=0

echo "======================================================================"
echo "  LoRA Fine-tuning Qwen Models — Upper Bound Baseline (auto batch)"
echo "======================================================================"
echo "  Models:     ${SIZES[*]}"
echo "  GPUs:       $NGPUS"
echo "  Target EBS: $TARGET_EBS"
echo "  Epochs:     $NUM_EPOCHS"
echo "  Train:      $DATA_PATH"
echo "  Eval:       $EVAL_DATA_PATH"
echo "  Start:      $(date)"
echo "======================================================================"
echo ""

# Check data exists
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Training data not found at $DATA_PATH"
    echo "Please run: python scripts/prepare_large_qa_data.py"
    exit 1
fi
if [ ! -f "$EVAL_DATA_PATH" ]; then
    echo "Error: Eval data not found at $EVAL_DATA_PATH"
    echo "Please run: python scripts/prepare_large_qa_data.py"
    exit 1
fi

for SIZE in "${SIZES[@]}"; do
    MODEL_PATH="${MODEL_PATHS[$SIZE]}"
    EBS="${EVAL_BATCH_SIZES[$SIZE]}"
    OUTPUT_DIR="outputs/lora_qwen3-${SIZE,,}"

    echo "--------------------------------------------------------------------"
    echo "[${SIZE}] Starting at $(date)"
    echo "  Model:         $MODEL_PATH"
    echo "  Auto batch:    yes (target_ebs=$TARGET_EBS)"
    echo "  Eval batch:    $EBS"
    echo "  Output:        $OUTPUT_DIR"
    echo "--------------------------------------------------------------------"

    if [ ! -d "$MODEL_PATH" ]; then
        echo "[${SIZE}] Model not found at $MODEL_PATH — SKIPPED"
        RESULTS[$SIZE]="SKIPPED (model not found)"
        ((FAILED++))
        continue
    fi

    export WANDB_MODE=offline
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    START_SEC=$SECONDS
    set +e

    accelerate launch \
        --multi_gpu \
        --num_processes "$NGPUS" \
        --mixed_precision bf16 \
        "$SCRIPT_DIR/finetune_qwen_lora.py" \
        --model_name_or_path "$MODEL_PATH" \
        --train_data "$DATA_PATH" \
        --eval_data "$EVAL_DATA_PATH" \
        --num_epochs "$NUM_EPOCHS" \
        --auto_batch_size \
        --target_effective_batch_size "$TARGET_EBS" \
        --eval_batch_size "$EBS" \
        --max_eval_samples "$MAX_EVAL_SAMPLES" \
        --gradient_checkpointing \
        --eval_every_steps 250 \
        --output_dir "$OUTPUT_DIR" \
        2>&1 | tee "${OUTPUT_DIR}_training.log"

    EXIT_CODE=${PIPESTATUS[0]}
    set -e
    ELAPSED=$(( SECONDS - START_SEC ))
    HOURS=$(( ELAPSED / 3600 ))
    MINS=$(( (ELAPSED % 3600) / 60 ))
    DURATIONS[$SIZE]="${HOURS}h${MINS}m"

    if [ $EXIT_CODE -eq 0 ]; then
        RESULTS[$SIZE]="OK"
        ((SUCCEEDED++))
    else
        RESULTS[$SIZE]="FAILED (exit $EXIT_CODE)"
        ((FAILED++))
    fi

    echo "[${SIZE}] Finished: ${RESULTS[$SIZE]}  (${DURATIONS[$SIZE]})"
    echo ""
done

echo ""
echo "======================================================================"
echo "                            SUMMARY"
echo "======================================================================"
printf "  %-10s %-12s %-48s\n" "Model" "Duration" "Status"
echo "  ----------------------------------------------------------------"
for SIZE in "${SIZES[@]}"; do
    printf "  %-10s %-12s %-48s\n" "$SIZE" "${DURATIONS[$SIZE]:---}" "${RESULTS[$SIZE]}"
done
echo "======================================================================"
echo "  Succeeded: $SUCCEEDED / ${#SIZES[@]}    Failed: $FAILED / ${#SIZES[@]}"
echo "  End: $(date)"
echo "======================================================================"

exit $FAILED
