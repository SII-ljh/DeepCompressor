#!/bin/bash
# Phase 2 ablation — FULL run (10K NTP + 5K QA per experiment)
# 15 experiments across 8x GPU. Takes ~3h.
# Run scripts/run_phase2_fast.sh first to validate.
#
# Usage:
#   bash scripts/run_phase2_full.sh
#
# Prerequisites:
#   python scripts/prepare_data.py --make-ablation
#   bash scripts/run_phase2_fast.sh   (recommended)

set -euo pipefail
source "$(dirname "$0")/run_phase2_common.sh"

OUTDIR=outputs/ablation_phase2
LOGDIR=logs/phase2
mkdir -p "$LOGDIR"

echo "============================================"
echo "  Phase 2 FULL ablation (15 experiments)"
echo "  10K NTP + 5K QA per experiment"
echo "============================================"
echo ""

COMMON="--data_path $NTP_DATA --qa_data_path $QA_TRAIN --eval_data_path $QA_DEV --mixed_precision bf16 --batch_size 16 --max_eval_samples 500"

launch_all "$OUTDIR" "$LOGDIR" "$COMMON"
wait_all
merge_results "$OUTDIR"

echo ""
echo "[INFO] Full ablation complete. Results: ${OUTDIR}/phase2_results.json"
