#!/usr/bin/env python3
"""Measure teacher (frozen Qwen) NTP perplexity baseline.

Loads Qwen3-0.6B and computes next-token prediction loss on the same
NTP data format used by training.  This gives the theoretical PPL floor
that the compressor can never beat.

Usage:
    python scripts/teacher_ppl.py \
        --data_path data/ntp_tiny.jsonl \
        --max_samples 200 --max_doc_tokens 512 --segment_len 64
"""

import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Qwen3-0.6B")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--max_doc_tokens", type=int, default=512)
    parser.add_argument("--segment_len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dev = device()
    print(f"Device: {dev}")

    # Load model + tokenizer
    print("Loading Qwen model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32)
    model.to(dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Read NTP data (sample subset)
    print(f"Reading {args.data_path} ...")
    lines = []
    with open(args.data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
            if len(lines) >= args.max_samples * 5:
                break

    rng = random.Random(args.seed)
    if len(lines) > args.max_samples:
        lines = rng.sample(lines, args.max_samples)
    print(f"Sampled {len(lines)} documents")

    # Compute teacher PPL
    total_loss = 0.0
    total_tokens = 0
    t0 = time.time()

    for i, raw_line in enumerate(lines):
        text = json.loads(raw_line)["text"]
        tokens = tokenizer(
            text, truncation=True,
            max_length=args.max_doc_tokens + args.segment_len,
            return_tensors="pt", padding=False)
        input_ids = tokens["input_ids"].squeeze(0)
        total_len = input_ids.shape[0]

        # Same split logic as NTPDataset
        if total_len <= args.segment_len + 1:
            split_point = total_len // 2
        else:
            max_split = total_len - args.segment_len
            split_point = rng.randint(
                1, max(1, min(max_split, args.max_doc_tokens)))

        doc_ids = input_ids[:split_point]
        seg_ids = input_ids[split_point:]

        if len(seg_ids) < 2:
            continue

        # Teacher: feed full [doc + segment], compute loss on segment portion
        full_ids = torch.cat([doc_ids, seg_ids]).unsqueeze(0).to(dev)
        with torch.no_grad():
            out = model(input_ids=full_ids, use_cache=False)
            logits = out.logits  # (1, seq_len, vocab)

        # Loss on segment tokens only (same as training: predict seg[1:] from seg[:-1])
        # logits[0, i] predicts input_ids[i+1]
        # We want to predict seg_ids[1], ..., seg_ids[-1]
        # These are predicted by logits at positions split_point, ..., split_point+len(seg_ids)-2
        seg_logits = logits[0, split_point: split_point + len(seg_ids) - 1]
        seg_targets = seg_ids[1:].to(dev)

        loss = torch.nn.functional.cross_entropy(
            seg_logits, seg_targets, reduction="sum")

        n_tokens = seg_targets.shape[0]
        total_loss += loss.item()
        total_tokens += n_tokens

        if (i + 1) % 50 == 0:
            running_ppl = math.exp(total_loss / total_tokens)
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(lines)}] running PPL={running_ppl:.2f}  "
                  f"({elapsed:.1f}s)")

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss)
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"  Teacher (Qwen3-0.6B) NTP Baseline")
    print(f"{'='*60}")
    print(f"  Samples:        {len(lines)}")
    print(f"  Total tokens:   {total_tokens}")
    print(f"  max_doc_tokens: {args.max_doc_tokens}")
    print(f"  segment_len:    {args.segment_len}")
    print(f"  Avg loss:       {avg_loss:.4f}")
    print(f"  Perplexity:     {ppl:.2f}")
    print(f"  Time:           {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"\n  Your compressor should aim for PPL < {ppl * 3:.0f} "
          f"(3x teacher)")
    print(f"  Ideal target: PPL < {ppl * 2:.0f} (2x teacher)")


if __name__ == "__main__":
    main()
