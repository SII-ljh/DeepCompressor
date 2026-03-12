#!/bin/bash
# LoRA fine-tune Qwen models of various sizes on QA data (upper-bound baseline).
# Runs each model size sequentially — one failure won't stop the rest.
#
# Usage:
#   bash scripts/finetune_lora_all.sh                    # all models
#   bash scripts/finetune_lora_all.sh 0.6B 1.7B          # specific sizes
#   NGPUS=4 bash scripts/finetune_lora_all.sh             # custom GPU count

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configurable
NGPUS="${NGPUS:-8}"
DATA_PATH="${DATA_PATH:-data/qa_large_train.json}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-data/qa_large_dev.json}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-5000}"

# Model sizes and their per-GPU batch sizes (tuned for 8x H200 141GB)
# Note: 4096-token sequences make attention memory the bottleneck, not model size.
# Generate (eval) needs more memory than training due to KV cache, so eval_bs < train_bs.
declare -A MODEL_PATHS
MODEL_PATHS["0.6B"]="models/Qwen3-0.6B"
MODEL_PATHS["1.7B"]="models/Qwen3-1.7B"
MODEL_PATHS["4B"]="models/Qwen3-4B"
MODEL_PATHS["8B"]="models/Qwen3-8B"

declare -A BATCH_SIZES
BATCH_SIZES["0.6B"]=16
BATCH_SIZES["1.7B"]=10
BATCH_SIZES["4B"]=5
BATCH_SIZES["8B"]=2

declare -A EVAL_BATCH_SIZES
EVAL_BATCH_SIZES["0.6B"]=8
EVAL_BATCH_SIZES["1.7B"]=5
EVAL_BATCH_SIZES["4B"]=2
EVAL_BATCH_SIZES["8B"]=1

declare -A GRAD_ACCUM
GRAD_ACCUM["0.6B"]=2
GRAD_ACCUM["1.7B"]=3
GRAD_ACCUM["4B"]=6
GRAD_ACCUM["8B"]=16

if [ $# -gt 0 ]; then
    SIZES=("$@")
else
    SIZES=("0.6B" "1.7B" "4B" "8B")
fi

declare -A RESULTS
FAILED=0
SUCCEEDED=0

echo "======================================================================"
echo "  LoRA Fine-tuning Qwen Models — Upper Bound Baseline"
echo "======================================================================"
echo "  Models:    ${SIZES[*]}"
echo "  GPUs:      $NGPUS"
echo "  Epochs:    $NUM_EPOCHS"
echo "  Train:     $DATA_PATH"
echo "  Eval:      $EVAL_DATA_PATH"
echo "  Start:     $(date)"
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
    BS="${BATCH_SIZES[$SIZE]}"
    EBS="${EVAL_BATCH_SIZES[$SIZE]}"
    GA="${GRAD_ACCUM[$SIZE]}"
    OUTPUT_DIR="outputs/lora_qwen3-${SIZE,,}"

    echo "--------------------------------------------------------------------"
    echo "[${SIZE}] Starting at $(date)"
    echo "  Model:         $MODEL_PATH"
    echo "  Batch/GPU:     $BS (eval: $EBS)"
    echo "  Grad accum:    $GA"
    EFF=$((NGPUS * BS * GA))
    echo "  Effective:     $EFF"
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

    accelerate launch \
        --multi_gpu \
        --num_processes "$NGPUS" \
        --mixed_precision bf16 \
        "$SCRIPT_DIR/finetune_qwen_lora.py" \
        --model_name_or_path "$MODEL_PATH" \
        --train_data "$DATA_PATH" \
        --eval_data "$EVAL_DATA_PATH" \
        --num_epochs "$NUM_EPOCHS" \
        --batch_size "$BS" \
        --eval_batch_size "$EBS" \
        --gradient_accumulation "$GA" \
        --max_eval_samples "$MAX_EVAL_SAMPLES" \
        --gradient_checkpointing \
        --eval_every_steps 500 \
        --output_dir "$OUTPUT_DIR" \
        2>&1 | tee "${OUTPUT_DIR}_training.log"

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        RESULTS[$SIZE]="OK"
        ((SUCCEEDED++))
    else
        RESULTS[$SIZE]="FAILED (exit $EXIT_CODE)"
        ((FAILED++))
    fi

    echo "[${SIZE}] Finished with: ${RESULTS[$SIZE]}"
    echo ""
done

echo ""
echo "======================================================================"
echo "                            SUMMARY"
echo "======================================================================"
printf "  %-10s %-58s\n" "Model" "Status"
echo "  ----------------------------------------------------------------"
for SIZE in "${SIZES[@]}"; do
    printf "  %-10s %-58s\n" "$SIZE" "${RESULTS[$SIZE]}"
done
echo "======================================================================"
echo "  Succeeded: $SUCCEEDED / ${#SIZES[@]}    Failed: $FAILED / ${#SIZES[@]}"
echo "  End: $(date)"
echo "======================================================================"

exit $FAILED
