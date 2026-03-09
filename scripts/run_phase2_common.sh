#!/bin/bash
# Shared functions for Phase 2 ablation launchers.
# Sourced by run_phase2_fast.sh and run_phase2_full.sh.

# 切换到项目根目录并设置 PYTHONPATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

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

# ── GPU assignment (15 experiments, balanced by compute cost) ──
declare -A GPU_CONFIGS
GPU_CONFIGS[0]="baseline,no_stage_a"
GPU_CONFIGS[1]="proj_identity,no_stage_b"
GPU_CONFIGS[2]="queries_16,no_query_cond"
GPU_CONFIGS[3]="no_stage_c,proj_linear"
GPU_CONFIGS[4]="queries_32,no_stage_ac"
GPU_CONFIGS[5]="deep,no_distillation"
GPU_CONFIGS[6]="full_distillation,lr_1e-3"
GPU_CONFIGS[7]="long_ntp"

# ── Shared state ──
PIDS=()

launch_all() {
    local outdir="$1"
    local logdir="$2"
    local common="$3"

    PIDS=()
    for GPU in $(seq 0 7); do
        local configs="${GPU_CONFIGS[$GPU]}"
        local log="$logdir/gpu${GPU}.log"
        echo "[INFO] GPU $GPU: $configs -> $log"

        CUDA_VISIBLE_DEVICES=$GPU python scripts/ablation_phase2.py \
            --configs "$configs" \
            --output_dir "${outdir}_gpu${GPU}" \
            $common \
            > "$log" 2>&1 &

        PIDS+=($!)
    done

    echo ""
    echo "[INFO] Launched ${#PIDS[@]} processes: ${PIDS[*]}"
    echo "[INFO] Monitor:  tail -f $logdir/gpu0.log"
    echo "[INFO] All logs:  tail -f $logdir/gpu*.log"
    echo ""
}

wait_all() {
    local failed=0
    for i in $(seq 0 7); do
        local pid=${PIDS[$i]}
        local configs="${GPU_CONFIGS[$i]}"
        if wait "$pid"; then
            echo "[DONE] GPU $i ($configs): OK"
        else
            echo "[FAIL] GPU $i ($configs): exit code $?"
            failed=$((failed + 1))
        fi
    done

    if [[ "$failed" -gt 0 ]]; then
        echo ""
        echo "[WARN] $failed process(es) failed. Check logs."
    fi
}

merge_results() {
    local outdir="$1"
    echo ""
    echo "[INFO] Merging results..."
    python -c "
import json, glob, sys, os
merged = {}
for f in sorted(glob.glob('${outdir}_gpu*/phase2_results.json')):
    try:
        data = json.load(open(f))
        merged.update(data)
        print(f'  Loaded {len(data)} results from {f}')
    except Exception as e:
        print(f'  WARN: {f}: {e}', file=sys.stderr)

os.makedirs('${outdir}', exist_ok=True)
out = '${outdir}/phase2_results.json'
json.dump(merged, open(out, 'w'), indent=2)
print(f'Merged {len(merged)}/15 experiments -> {out}')

ok = sum(1 for r in merged.values() if r.get('status') == 'OK')
print(f'\nStatus: {ok} OK, {len(merged)-ok} other, {15-len(merged)} missing')
for name, r in sorted(merged.items()):
    s = r.get('status', '?')
    ntp_ppl = r.get('ntp_metrics', {}).get('perplexity')
    qa_f1 = r.get('qa_metrics', {}).get('f1')
    ppl_s = f'{ntp_ppl:.1f}' if ntp_ppl else '—'
    f1_s = f'{qa_f1:.4f}' if qa_f1 else '—'
    print(f'  {name:<22} PPL={ppl_s:<10} F1={f1_s:<10} [{s}]')
"
}
