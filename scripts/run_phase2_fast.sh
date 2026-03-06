#!/bin/bash
# Phase 2 ablation — FAST smoke test (2K NTP + 1K QA per experiment)
# Validates that all 15 experiments can run without errors on 8x GPU.
# Takes ~15 min. Run this BEFORE run_phase2_full.sh.
#
# Usage:
#   bash scripts/run_phase2_fast.sh
#
# Prerequisites:
#   python scripts/prepare_data.py --make-ablation

set -euo pipefail
source "$(dirname "$0")/run_phase2_common.sh"

OUTDIR=outputs/ablation_phase2_fast
LOGDIR=logs/phase2_fast
mkdir -p "$LOGDIR"

echo "============================================"
echo "  Phase 2 FAST smoke test (15 experiments)"
echo "  2K NTP + 1K QA per experiment"
echo "============================================"
echo ""

COMMON="--data_path $NTP_DATA --qa_data_path $QA_TRAIN --eval_data_path $QA_DEV --mixed_precision bf16 --batch_size 16 --fast"

launch_all "$OUTDIR" "$LOGDIR" "$COMMON"
wait_all
merge_results "$OUTDIR"

echo ""
echo "[INFO] Fast smoke test done. If all OK, run:"
echo "  bash scripts/run_phase2_full.sh"
