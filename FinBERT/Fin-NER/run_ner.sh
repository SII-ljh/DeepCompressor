#!/bin/bash
# Fine-tune FinBERT2 for financial NER
#
# Usage:
#   bash run_ner.sh                                    # default: FinBERT2-base
#   bash run_ner.sh valuesimplex-ai-lab/FinBERT2-large # use large model
#
# For GPU training:
#   CUDA_VISIBLE_DEVICES=0 bash run_ner.sh

MODEL_NAME="${1:-valuesimplex-ai-lab/FinBERT2-base}"

echo "=========================================="
echo "  Financial NER Fine-tuning"
echo "  Model: ${MODEL_NAME}"
echo "=========================================="

python finetune_ner.py \
    --model_name "${MODEL_NAME}" \
    --train_data "sample_data/train.json" \
    --test_data "sample_data/test.json" \
    --num_epochs 20 \
    --batch_size 4 \
    --learning_rate 3e-5
