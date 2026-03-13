#!/bin/bash
# Unified training script for varying Q values (replaces 6 individual scripts + orchestrator).
#
# Usage:
#   bash scripts/train_varying_q.sh                    # run all Q values (64 128 256 512 1024 2048)
#   bash scripts/train_varying_q.sh 64 256 1024        # run only specified Q values
#   bash scripts/train_varying_q.sh --dry-run 64       # print command without executing
#   bash scripts/train_varying_q.sh --dry-run           # dry-run all Q values

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

# ── Hyperparameter lookup table ──────────────────────────────────────────────
#   Q -> batch_size  grad_accum  eval_every  save_every  grad_ckpt
get_hparams() {
    local q=$1
    case $q in
        64)   BATCH=40; ACCUM=2; EVAL=189;  SAVE=756;  CKPT=false ;;
        128)  BATCH=30; ACCUM=2; EVAL=252;  SAVE=1008; CKPT=false ;;
        256)  BATCH=20; ACCUM=2; EVAL=378;  SAVE=1512; CKPT=false ;;
        512)  BATCH=16; ACCUM=2; EVAL=473;  SAVE=1890; CKPT=false ;;
        1024) BATCH=10; ACCUM=2; EVAL=756;  SAVE=3024; CKPT=false ;;
        2048) BATCH=5;  ACCUM=4; EVAL=756;  SAVE=3024; CKPT=true  ;;
        *)    echo "Unknown Q value: $q (supported: 64 128 256 512 1024 2048)"; return 1 ;;
    esac
}

# ── Common settings ──────────────────────────────────────────────────────────
NUM_GPUS=8
DATA_PATH="data/qa_large_train.json"
EVAL_DATA_PATH="data/qa_large_dev.json"
MAX_EVAL_SAMPLES=5000
LR=1e-4
MAX_STEPS=10000
WARMUP_STEPS=500

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
echo "  Deep Compressor — Varying Q Training (8 GPUs)"
echo "======================================================================"
echo "  Q values: ${Q_VALUES[*]}"
echo "  Dry run:  $DRY_RUN"
echo "  Start:    $(date)"
echo "======================================================================"
echo ""

for Q in "${Q_VALUES[@]}"; do
    if ! get_hparams "$Q"; then
        RESULTS[$Q]="SKIPPED (unknown Q)"
        ((FAILED++))
        continue
    fi

    CONFIG="configs/qa_q${Q}_8gpu.yaml"
    OUTPUT_DIR="outputs/qa_q${Q}_8gpu"
    EFF_BATCH=$((NUM_GPUS * BATCH * ACCUM))
    RATIO=$((4096 / Q))

    echo "----------------------------------------------------------------------"
    echo "[Q=$Q] Compression ${RATIO}:1 | batch=${BATCH} accum=${ACCUM} eff=${EFF_BATCH} | grad_ckpt=${CKPT}"
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
    --wandb --wandb_project deep-compressor --wandb_offline"

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
