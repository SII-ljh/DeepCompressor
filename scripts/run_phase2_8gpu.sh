#!/bin/bash
# Phase 2 ablation: 13 experiments across 8x H200 GPUs
#
# Usage:
#   bash scripts/run_phase2_8gpu.sh              # full run
#   bash scripts/run_phase2_8gpu.sh --fast       # smoke test (2K+1K steps)
#
# Prerequisites:
#   python scripts/prepare_data.py --make-ablation

set -euo pipefail

FAST=""
if [[ "${1:-}" == "--fast" ]]; then
    FAST="--fast"
    echo "[INFO] Fast mode: 2K NTP + 1K QA per experiment"
fi

# ── Data paths (auto-detect ablation subset vs full data) ──
if [[ -f data/ablation/ntp_ablation.jsonl ]]; then
    NTP_DATA=data/ablation/ntp_ablation.jsonl
    QA_TRAIN=data/ablation/qa_ablation_train.json
    QA_DEV=data/ablation/qa_ablation_dev.json
elif [[ -f data/ntp_train.jsonl ]]; then
    NTP_DATA=data/ntp_train.jsonl
    QA_TRAIN=data/qa_train.json
    QA_DEV=data/qa_dev.json
else
    echo "[ERROR] No data found. Run: python scripts/prepare_data.py --make-ablation"
    exit 1
fi

echo "[INFO] NTP data:  $NTP_DATA"
echo "[INFO] QA train:  $QA_TRAIN"
echo "[INFO] QA dev:    $QA_DEV"

# ── Check GPU count ──
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [[ "$NUM_GPUS" -lt 8 ]]; then
    echo "[WARN] Found $NUM_GPUS GPU(s), expected 8. Some jobs will queue."
fi

# ── Directories ──
OUTDIR=outputs/ablation_phase2
LOGDIR=logs/phase2
mkdir -p "$LOGDIR"

COMMON="--data_path $NTP_DATA --qa_data_path $QA_TRAIN --eval_data_path $QA_DEV --mixed_precision bf16 --batch_size 16 $FAST"

# ── GPU assignment (balanced by compute cost) ──
#   GPU 0-4: 2 experiments each (standard, ~30K steps)
#   GPU 5:   deep (slower per step, more layers)
#   GPU 6:   full_distillation (loads teacher model)
#   GPU 7:   long_ntp (20K NTP steps)
declare -A GPU_CONFIGS
GPU_CONFIGS[0]="baseline,no_stage_a"
GPU_CONFIGS[1]="proj_identity,no_stage_b"
GPU_CONFIGS[2]="queries_16,proj_linear"
GPU_CONFIGS[3]="queries_32,no_stage_ac"
GPU_CONFIGS[4]="no_distillation,lr_1e-3"
GPU_CONFIGS[5]="deep"
GPU_CONFIGS[6]="full_distillation"
GPU_CONFIGS[7]="long_ntp"

# ── Launch 8 processes ──
PIDS=()
for GPU in $(seq 0 7); do
    CONFIGS="${GPU_CONFIGS[$GPU]}"
    LOG="$LOGDIR/gpu${GPU}.log"
    echo "[INFO] GPU $GPU: $CONFIGS -> $LOG"

    CUDA_VISIBLE_DEVICES=$GPU python scripts/ablation_phase2.py \
        --configs "$CONFIGS" \
        --output_dir "${OUTDIR}_gpu${GPU}" \
        $COMMON \
        > "$LOG" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "[INFO] Launched ${#PIDS[@]} processes: ${PIDS[*]}"
echo "[INFO] Monitor:  tail -f $LOGDIR/gpu0.log"
echo "[INFO] All logs:  tail -f $LOGDIR/gpu*.log"
echo ""

# ── Wait for all processes ──
FAILED=0
for i in $(seq 0 7); do
    PID=${PIDS[$i]}
    CONFIGS="${GPU_CONFIGS[$i]}"
    if wait "$PID"; then
        echo "[DONE] GPU $i ($CONFIGS): OK"
    else
        echo "[FAIL] GPU $i ($CONFIGS): exit code $?"
        FAILED=$((FAILED + 1))
    fi
done

# ── Merge results ──
echo ""
echo "[INFO] Merging results..."
python -c "
import json, glob, sys
merged = {}
for f in sorted(glob.glob('${OUTDIR}_gpu*/phase2_results.json')):
    try:
        data = json.load(open(f))
        merged.update(data)
        print(f'  Loaded {len(data)} results from {f}')
    except Exception as e:
        print(f'  WARN: {f}: {e}', file=sys.stderr)

out = '${OUTDIR}/phase2_results.json'
import os; os.makedirs('${OUTDIR}', exist_ok=True)
json.dump(merged, open(out, 'w'), indent=2)
print(f'Merged {len(merged)}/13 experiments -> {out}')

# Summary table
ok = sum(1 for r in merged.values() if r.get('status') == 'OK')
print(f'\nStatus: {ok} OK, {len(merged)-ok} other, {13-len(merged)} missing')
for name, r in sorted(merged.items()):
    s = r.get('status', '?')
    ntp_ppl = r.get('ntp_metrics', {}).get('perplexity')
    qa_f1 = r.get('qa_metrics', {}).get('f1')
    ppl_s = f'{ntp_ppl:.1f}' if ntp_ppl else '—'
    f1_s = f'{qa_f1:.4f}' if qa_f1 else '—'
    print(f'  {name:<22} PPL={ppl_s:<10} F1={f1_s:<10} [{s}]')
"

if [[ "$FAILED" -gt 0 ]]; then
    echo ""
    echo "[WARN] $FAILED process(es) failed. Check logs in $LOGDIR/"
    exit 1
fi

echo ""
echo "[INFO] All 13 experiments complete."
