#!/bin/bash
# Unified LoRA fine-tuning shell wrapper for finetune_qwen_lora.py.
# Runs LoRA fine-tuning across different Qwen model sizes as upper-bound baselines.
#
# Usage:
#   bash scripts/train_lora_finetune.sh                # fine-tune all models (0.6b 1.7b 4b 8b)
#   bash scripts/train_lora_finetune.sh 0.6b 4b        # fine-tune only specified models
#   bash scripts/train_lora_finetune.sh --dry-run 0.6b  # print command without executing
#   bash scripts/train_lora_finetune.sh --dry-run        # dry-run all models

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

# ── Hyperparameter lookup table ──────────────────────────────────────────────
#   model -> display_name  model_path  batch  grad_accum  lr  lora_r  epochs  grad_ckpt
get_hparams() {
    local model=$1
    case $model in
        0.6b) NAME="Qwen3-0.6B"; MODEL_PATH="models/Qwen3-0.6B"; BATCH=20; ACCUM=2;  LR=2e-4; LORA_R=16; EPOCHS=2; CKPT="" ;;
        1.7b) NAME="Qwen3-1.7B"; MODEL_PATH="models/Qwen3-1.7B"; BATCH=12; ACCUM=4;  LR=2e-4; LORA_R=16; EPOCHS=2; CKPT="" ;;
        4b)   NAME="Qwen3-4B";   MODEL_PATH="models/Qwen3-4B";   BATCH=8;  ACCUM=4;  LR=1e-4; LORA_R=16; EPOCHS=2; CKPT="--gradient_checkpointing" ;;
        8b)   NAME="Qwen3-8B";   MODEL_PATH="models/Qwen3-8B";   BATCH=4;  ACCUM=8;  LR=1e-4; LORA_R=16; EPOCHS=2; CKPT="--gradient_checkpointing" ;;
        *)    echo "Unknown model: $model (supported: 0.6b 1.7b 4b 8b)"; return 1 ;;
    esac
}

# ── Common settings ──────────────────────────────────────────────────────────
NUM_GPUS=8
TRAIN_DATA="data/qa_large_train.json"
EVAL_DATA="data/qa_large_dev.json"
MAX_EVAL_SAMPLES=5000
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
echo "  LoRA Fine-tuning — Upper-Bound Baselines (8 GPUs)"
echo "======================================================================"
echo "  Models:  ${MODELS[*]}"
echo "  Dry run: $DRY_RUN"
echo "  Start:   $(date)"
echo "======================================================================"
echo ""

for MODEL in "${MODELS[@]}"; do
    if ! get_hparams "$MODEL"; then
        RESULTS[$MODEL]="SKIPPED (unknown model)"
        ((FAILED++))
        continue
    fi

    OUTPUT_DIR="outputs/lora_qwen3-${MODEL}"
    EFF_BATCH=$((NUM_GPUS * BATCH * ACCUM))

    echo "----------------------------------------------------------------------"
    echo "[$NAME] batch=$BATCH accum=$ACCUM eff=$EFF_BATCH lr=$LR lora_r=$LORA_R epochs=$EPOCHS ${CKPT:+grad_ckpt}"
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
    --batch_size $BATCH \
    --gradient_accumulation $ACCUM \
    --learning_rate $LR \
    --lora_r $LORA_R \
    --num_epochs $EPOCHS \
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
    get_hparams "$MODEL" 2>/dev/null
    printf "  %-14s %-12s %-30s\n" "${NAME:-$MODEL}" "${DURATIONS[$MODEL]:---}" "${RESULTS[$MODEL]}"
done
echo "----------------------------------------------------------------------"
echo "  Succeeded: $SUCCEEDED / ${#MODELS[@]}    Failed: $FAILED / ${#MODELS[@]}"
echo "  End: $(date)"
echo "======================================================================"

exit $FAILED
