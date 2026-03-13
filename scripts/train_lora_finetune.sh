#!/bin/bash
# Unified LoRA fine-tuning shell wrapper for finetune_qwen_lora.py.
# Uses auto batch size detection — no manual batch/accum tuning needed.
# Runs LoRA fine-tuning across different Qwen model sizes as upper-bound baselines.
#
# Usage:
#   bash scripts/train_lora_finetune.sh                # fine-tune all models (0.6b 1.7b 4b 8b)
#   bash scripts/train_lora_finetune.sh 0.6b 4b        # fine-tune only specified models
#   bash scripts/train_lora_finetune.sh --dry-run 0.6b  # print command without executing
#   bash scripts/train_lora_finetune.sh --dry-run        # dry-run all models
#
# Environment variables:
#   NGPUS=4                     # number of GPUs (default: 8)
#   TARGET_EBS=128              # target effective batch size (default: 256)
#   NUM_EPOCHS=2                # number of epochs (default: 1)
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

# ── Model registry (only model-specific settings that can't be auto-detected) ──
get_model_info() {
    local model=$1
    case $model in
        0.6b) NAME="Qwen3-0.6B"; MODEL_PATH="models/Qwen3-0.6B"; LR=2e-4; LORA_R=16; CKPT="" ;;
        1.7b) NAME="Qwen3-1.7B"; MODEL_PATH="models/Qwen3-1.7B"; LR=2e-4; LORA_R=16; CKPT="" ;;
        4b)   NAME="Qwen3-4B";   MODEL_PATH="models/Qwen3-4B";   LR=1e-4; LORA_R=16; CKPT="--gradient_checkpointing" ;;
        8b)   NAME="Qwen3-8B";   MODEL_PATH="models/Qwen3-8B";   LR=1e-4; LORA_R=16; CKPT="--gradient_checkpointing" ;;
        *)    echo "Unknown model: $model (supported: 0.6b 1.7b 4b 8b)"; return 1 ;;
    esac
}

# ── Common settings ──────────────────────────────────────────────────────────
NUM_GPUS="${NGPUS:-8}"
TARGET_EBS="${TARGET_EBS:-256}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
TRAIN_DATA="${DATA_PATH:-data/qa_large_train.json}"
EVAL_DATA="${EVAL_DATA_PATH:-data/qa_large_dev.json}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-5000}"
FINETUNE_SCRIPT="$SCRIPT_DIR/finetune_qwen_lora.py"

# ── Data check ───────────────────────────────────────────────────────────────
if [ "$DRY_RUN" = false ]; then
    if [ ! -f "$FINETUNE_SCRIPT" ]; then
        echo "Error: finetune_qwen_lora.py not found at $FINETUNE_SCRIPT"
        exit 1
    fi
    for f in "$TRAIN_DATA" "$EVAL_DATA"; do
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
echo "  LoRA Fine-tuning — Upper-Bound Baselines (${NUM_GPUS} GPUs, auto batch)"
echo "======================================================================"
echo "  Models:     ${MODELS[*]}"
echo "  Target EBS: $TARGET_EBS"
echo "  Epochs:     $NUM_EPOCHS"
echo "  Dry run:    $DRY_RUN"
echo "  Start:      $(date)"
echo "======================================================================"
echo ""

for MODEL in "${MODELS[@]}"; do
    if ! get_model_info "$MODEL"; then
        RESULTS[$MODEL]="SKIPPED (unknown model)"
        ((FAILED++))
        continue
    fi

    OUTPUT_DIR="outputs/lora_qwen3-${MODEL}"

    echo "----------------------------------------------------------------------"
    echo "[$NAME] auto batch | target_ebs=$TARGET_EBS | lr=$LR | lora_r=$LORA_R ${CKPT:+| grad_ckpt}"
    echo "----------------------------------------------------------------------"

    CMD="accelerate launch \
    --multi_gpu \
    --num_processes $NUM_GPUS \
    --mixed_precision bf16 \
    $FINETUNE_SCRIPT \
    --model_name_or_path $MODEL_PATH \
    --train_data $TRAIN_DATA \
    --eval_data $EVAL_DATA \
    --max_eval_samples $MAX_EVAL_SAMPLES \
    --auto_batch_size \
    --target_effective_batch_size $TARGET_EBS \
    --learning_rate $LR \
    --lora_r $LORA_R \
    --num_epochs $NUM_EPOCHS \
    --output_dir $OUTPUT_DIR \
    $CKPT"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would execute:"
        echo "  $CMD"
        echo "  Output: $OUTPUT_DIR"
        echo ""
        RESULTS[$MODEL]="DRY RUN"
        ((SUCCEEDED++))
        continue
    fi

    mkdir -p "$OUTPUT_DIR"
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
    get_model_info "$MODEL" 2>/dev/null
    printf "  %-14s %-12s %-30s\n" "${NAME:-$MODEL}" "${DURATIONS[$MODEL]:---}" "${RESULTS[$MODEL]}"
done
echo "----------------------------------------------------------------------"
echo "  Succeeded: $SUCCEEDED / ${#MODELS[@]}    Failed: $FAILED / ${#MODELS[@]}"
echo "  End: $(date)"
echo "======================================================================"

exit $FAILED
