#!/usr/bin/env python3
"""LoRA fine-tune a Qwen model on QA data as an upper-bound baseline.

Trains for 1-2 epochs with LoRA adapters on the frozen base model, then
evaluates with both teacher-forcing (loss/perplexity) and greedy generation
(EM/F1/ROUGE-L).  Results serve as a *fair* upper bound for comparison
against Deep Compressor — both see the same training data.

Usage:
    # Single GPU
    python scripts/finetune_qwen_lora.py \
        --model_name_or_path models/Qwen3-0.6B \
        --train_data data/qa_large_train.json \
        --eval_data data/qa_large_dev.json \
        --num_epochs 2 --batch_size 4 --gradient_accumulation 4 \
        --output_dir outputs/lora_qwen3-0.6b

    # Multi-GPU via accelerate
    accelerate launch --multi_gpu --num_processes 8 \
        scripts/finetune_qwen_lora.py \
        --model_name_or_path models/Qwen3-0.6B \
        --train_data data/qa_large_train.json \
        --eval_data data/qa_large_dev.json \
        --batch_size 20 --gradient_accumulation 2 \
        --output_dir outputs/lora_qwen3-0.6b
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root so we can import deep_compressor.eval
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deep_compressor.eval import compute_exact_match, compute_f1, compute_rouge_l


# ── Dataset ──────────────────────────────────────────────────────────────────


def _apply_chat_template(tokenizer, context: str, question: str,
                         enable_thinking: bool = False) -> str:
    """Format context+question using the model's chat template."""
    messages = [
        {"role": "user",
         "content": f"Context: {context}\n\nQuestion: {question}"},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True,
            tokenize=False, enable_thinking=enable_thinking)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)


def _strip_thinking(text: str) -> str:
    """Remove ``<think>...</think>`` blocks from generated text."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class LoRAQADataset(Dataset):
    """QA dataset that produces chat-template-formatted training sequences."""

    def __init__(self, data_path: str, tokenizer, max_context_tokens: int = 4096,
                 max_answer_tokens: int = 512):
        with open(data_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)
        self.tokenizer = tokenizer
        self.max_context_tokens = max_context_tokens
        self.max_answer_tokens = max_answer_tokens
        self.eos_id = tokenizer.eos_token_id

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        context = s.get("context", s.get("context_text", ""))
        question = s.get("question", s.get("question_text", ""))
        answer = s.get("answer", s.get("answer_text", ""))

        prompt_text = _apply_chat_template(
            self.tokenizer, context, question, enable_thinking=False)
        prompt_ids = self.tokenizer(
            prompt_text, add_special_tokens=False,
            truncation=True, max_length=self.max_context_tokens,
        )["input_ids"]
        answer_ids = self.tokenizer(
            answer, add_special_tokens=False,
            truncation=True, max_length=self.max_answer_tokens,
        )["input_ids"]

        full_ids = prompt_ids + answer_ids + [self.eos_id]
        labels = [-100] * len(prompt_ids) + answer_ids + [self.eos_id]

        return {
            "input_ids": full_ids,
            "labels": labels,
            "context_text": context,
            "question_text": question,
            "answer_text": answer,
        }


def collate_fn(batch: List[Dict], pad_id: int) -> Dict:
    """Right-pad and collate a batch of variable-length sequences."""
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

    for i, b in enumerate(batch):
        slen = len(b["input_ids"])
        input_ids[i, :slen] = torch.tensor(b["input_ids"], dtype=torch.long)
        labels[i, :slen] = torch.tensor(b["labels"], dtype=torch.long)
        attention_mask[i, :slen] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "context_text": [b["context_text"] for b in batch],
        "question_text": [b["question_text"] for b in batch],
        "answer_text": [b["answer_text"] for b in batch],
    }


# ── Evaluation ───────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(model, eval_loader: DataLoader, tokenizer, accelerator: Accelerator,
             max_new_tokens: int = 64, max_context_tokens: int = 4096,
             show_samples: int = 5) -> Dict[str, float]:
    """Two-pass evaluation: right-pad loss + left-pad generation."""
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    unwrapped = accelerator.unwrap_model(model)

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    total_loss = 0.0
    total_batches = 0
    all_em, all_f1, all_rl = [], [], []
    sample_preds: List[Tuple[str, str, str]] = []  # (question, pred, gold)

    for batch in eval_loader:
        contexts = batch["context_text"]
        questions = batch["question_text"]
        golds = batch["answer_text"]
        B = len(contexts)
        device = accelerator.device

        # ── Per-sample: build chat-formatted token sequences ──
        all_full_ids: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []
        all_prompt_ids: List[torch.Tensor] = []

        for i in range(B):
            prompt_text = _apply_chat_template(
                tokenizer, contexts[i], questions[i], enable_thinking=False)
            p_ids = tokenizer(
                prompt_text, add_special_tokens=False,
                truncation=True, max_length=max_context_tokens,
            )["input_ids"]
            a_ids = tokenizer(
                golds[i], add_special_tokens=False,
                truncation=True, max_length=512,
            )["input_ids"]

            full_ids = p_ids + a_ids + [eos_id]
            lab = [-100] * len(p_ids) + a_ids + [eos_id]

            all_full_ids.append(torch.tensor(full_ids, dtype=torch.long))
            all_labels.append(torch.tensor(lab, dtype=torch.long))
            all_prompt_ids.append(torch.tensor(p_ids, dtype=torch.long))

        # ── Pass 1: Right-pad for teacher-forcing loss ──
        max_len = max(t.shape[0] for t in all_full_ids)
        rp_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
        rp_labels = torch.full((B, max_len), -100, dtype=torch.long, device=device)
        rp_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
        for i in range(B):
            slen = all_full_ids[i].shape[0]
            rp_ids[i, :slen] = all_full_ids[i]
            rp_labels[i, :slen] = all_labels[i]
            rp_mask[i, :slen] = 1

        outputs = unwrapped(
            input_ids=rp_ids, attention_mask=rp_mask,
            labels=rp_labels, use_cache=False,
        )
        total_loss += outputs.loss.detach().item()
        total_batches += 1
        del outputs, rp_ids, rp_labels, rp_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Pass 2: Left-pad for generation ──
        max_plen = max(t.shape[0] for t in all_prompt_ids)
        lp_ids = torch.full((B, max_plen), pad_id, dtype=torch.long, device=device)
        lp_mask = torch.zeros((B, max_plen), dtype=torch.long, device=device)
        for i in range(B):
            slen = all_prompt_ids[i].shape[0]
            lp_ids[i, max_plen - slen:] = all_prompt_ids[i]
            lp_mask[i, max_plen - slen:] = 1

        gen_out = unwrapped.generate(
            input_ids=lp_ids, attention_mask=lp_mask,
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=pad_id, eos_token_id=eos_id,
            repetition_penalty=1.2,
        )
        gen_only = gen_out[:, max_plen:]
        preds_raw = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
        preds = [_strip_thinking(p) for p in preds_raw]
        del lp_ids, lp_mask, gen_out, gen_only

        for i, (pred, gold) in enumerate(zip(preds, golds)):
            em = compute_exact_match(pred, gold)
            f1_val = compute_f1(pred, gold)
            rl = compute_rouge_l(pred, gold)
            all_em.append(em)
            all_f1.append(f1_val)
            all_rl.append(rl)
            if len(sample_preds) < show_samples:
                sample_preds.append((questions[i], pred, gold))

    model.train()

    avg_loss = total_loss / max(total_batches, 1)
    ppl = math.exp(min(avg_loss, 20.0))
    avg_em = sum(all_em) / max(len(all_em), 1)
    avg_f1 = sum(all_f1) / max(len(all_f1), 1)
    avg_rl = sum(all_rl) / max(len(all_rl), 1)

    metrics = {
        "loss": avg_loss,
        "perplexity": ppl,
        "exact_match": avg_em,
        "f1": avg_f1,
        "rouge_l": avg_rl,
        "n_samples": len(all_em),
    }

    if accelerator.is_main_process and sample_preds:
        print("\n  Sample predictions:")
        for q, pred, gold in sample_preds:
            print(f"    Q: {q[:80]}")
            print(f"    Pred: {pred[:120]}")
            print(f"    Gold: {gold[:120]}")
            print()

    return metrics


# ── Training ─────────────────────────────────────────────────────────────────


def train(args):
    # ── Accelerator setup ──
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation,
        mixed_precision="bf16" if torch.cuda.is_available() else "no",
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_main_process:
        print("=" * 72)
        print("  LoRA Fine-tuning Qwen for QA (Upper Bound Baseline)")
        print("=" * 72)
        print(f"  Model:           {args.model_name_or_path}")
        print(f"  Train data:      {args.train_data}")
        print(f"  Eval data:       {args.eval_data}")
        print(f"  Epochs:          {args.num_epochs}")
        print(f"  Batch size/GPU:  {args.batch_size}")
        print(f"  Grad accum:      {args.gradient_accumulation}")
        print(f"  Num processes:   {accelerator.num_processes}")
        eff_bs = args.batch_size * args.gradient_accumulation * accelerator.num_processes
        print(f"  Effective batch: {eff_bs}")
        print(f"  Learning rate:   {args.learning_rate}")
        print(f"  LoRA r:          {args.lora_r}")
        print(f"  Output dir:      {args.output_dir}")
        print("=" * 72)
        print()

    # ── cuDNN SDPA workaround ──
    if torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(False)

    # ── Load model + tokenizer ──
    if accelerator.is_main_process:
        print("Loading model and tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=dtype,
        trust_remote_code=True,
    )

    # ── Gradient checkpointing ──
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

    # ── Apply LoRA ──
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    if accelerator.is_main_process:
        model.print_trainable_parameters()

    # ── Datasets ──
    if accelerator.is_main_process:
        print(f"\nLoading training data from {args.train_data}...")
    train_dataset = LoRAQADataset(
        args.train_data, tokenizer,
        max_context_tokens=args.max_context_tokens,
        max_answer_tokens=args.max_answer_tokens,
    )

    eval_dataset = None
    if args.eval_data:
        if accelerator.is_main_process:
            print(f"Loading eval data from {args.eval_data}...")
        eval_dataset = LoRAQADataset(
            args.eval_data, tokenizer,
            max_context_tokens=args.max_context_tokens,
            max_answer_tokens=args.max_answer_tokens,
        )
        if args.max_eval_samples and len(eval_dataset.samples) > args.max_eval_samples:
            eval_dataset.samples = eval_dataset.samples[:args.max_eval_samples]

    if accelerator.is_main_process:
        print(f"  Train samples: {len(train_dataset)}")
        if eval_dataset:
            print(f"  Eval samples:  {len(eval_dataset)}")

    pad_id = tokenizer.pad_token_id
    num_workers = 0 if not torch.cuda.is_available() else args.num_workers

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        collate_fn=lambda b: collate_fn(b, pad_id),
    )
    eval_loader = None
    if eval_dataset:
        eval_loader = DataLoader(
            eval_dataset, batch_size=args.eval_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
            collate_fn=lambda b: collate_fn(b, pad_id),
        )

    # ── Optimizer + Scheduler ──
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.learning_rate,
                      weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation)
    total_steps = steps_per_epoch * args.num_epochs

    warmup_steps = args.warmup_steps
    if warmup_steps <= 0:
        warmup_steps = max(1, int(0.05 * total_steps))

    if warmup_steps >= total_steps:
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=1.0,
                             total_iters=total_steps)
    else:
        warmup = LinearLR(
            optimizer,
            start_factor=1e-8 / max(args.learning_rate, 1e-12),
            end_factor=1.0, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
        scheduler = SequentialLR(optimizer, [warmup, cosine],
                                 milestones=[warmup_steps])

    # ── Prepare with Accelerator ──
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler)
    if eval_loader is not None:
        eval_loader = accelerator.prepare(eval_loader)

    # ── Training loop ──
    os.makedirs(args.output_dir, exist_ok=True)
    best_f1 = -1.0
    global_step = 0
    start_time = time.time()

    if accelerator.is_main_process:
        print(f"\nTotal training steps: {total_steps} "
              f"({steps_per_epoch}/epoch x {args.num_epochs} epochs)")
        print(f"Warmup steps: {warmup_steps}")
        print(f"Eval every: {args.eval_every_steps} steps")
        print()

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        pbar = None
        if accelerator.is_main_process:
            pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{args.num_epochs}")

        for batch_idx, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    use_cache=False,
                )
                loss = outputs.loss
                del outputs  # free logits (batch*seq*vocab ≈ 10GB)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(),
                                                args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                epoch_loss += loss.detach().item()
                epoch_steps += 1

                # ── Logging ──
                if accelerator.is_main_process and global_step % args.log_every_steps == 0:
                    avg = epoch_loss / max(epoch_steps, 1)
                    lr_now = scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    print(f"  step {global_step}/{total_steps}  "
                          f"loss={avg:.4f}  lr={lr_now:.2e}  "
                          f"elapsed={elapsed:.0f}s")

                # ── Evaluation ──
                if (eval_loader is not None
                        and args.eval_every_steps > 0
                        and global_step % args.eval_every_steps == 0):
                    if accelerator.is_main_process:
                        print(f"\n  [Eval @ step {global_step}]")
                    metrics = evaluate(
                        model, eval_loader, tokenizer, accelerator,
                        max_new_tokens=args.max_new_tokens,
                        max_context_tokens=args.max_context_tokens,
                    )
                    if accelerator.is_main_process:
                        print(f"  Loss={metrics['loss']:.4f}  "
                              f"PPL={metrics['perplexity']:.2f}  "
                              f"EM={metrics['exact_match']:.4f}  "
                              f"F1={metrics['f1']:.4f}  "
                              f"RL={metrics['rouge_l']:.4f}  "
                              f"(n={metrics['n_samples']})")

                        if metrics["f1"] > best_f1:
                            best_f1 = metrics["f1"]
                            save_dir = os.path.join(args.output_dir, "best_adapter")
                            print(f"  New best F1={best_f1:.4f}, saving to {save_dir}")
                            unwrapped = accelerator.unwrap_model(model)
                            unwrapped.save_pretrained(save_dir)
                            tokenizer.save_pretrained(save_dir)

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(loss=loss.detach().item(), step=global_step)

        if pbar is not None:
            pbar.close()

        # ── End of epoch eval ──
        if eval_loader is not None:
            if accelerator.is_main_process:
                print(f"\n  [End of epoch {epoch + 1} eval]")
            metrics = evaluate(
                model, eval_loader, tokenizer, accelerator,
                max_new_tokens=args.max_new_tokens,
                max_context_tokens=args.max_context_tokens,
            )
            if accelerator.is_main_process:
                print(f"  Loss={metrics['loss']:.4f}  "
                      f"PPL={metrics['perplexity']:.2f}  "
                      f"EM={metrics['exact_match']:.4f}  "
                      f"F1={metrics['f1']:.4f}  "
                      f"RL={metrics['rouge_l']:.4f}  "
                      f"(n={metrics['n_samples']})")

                if metrics["f1"] > best_f1:
                    best_f1 = metrics["f1"]
                    save_dir = os.path.join(args.output_dir, "best_adapter")
                    print(f"  New best F1={best_f1:.4f}, saving to {save_dir}")
                    unwrapped = accelerator.unwrap_model(model)
                    unwrapped.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)

    # ── Save final adapter ──
    if accelerator.is_main_process:
        save_dir = os.path.join(args.output_dir, "final_adapter")
        print(f"\nSaving final adapter to {save_dir}")
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

        elapsed = time.time() - start_time
        print(f"\nTraining complete in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
        print(f"Best eval F1: {best_f1:.4f}")
        print(f"Output dir: {args.output_dir}")

    # ── Final summary ──
    if accelerator.is_main_process and eval_loader is not None:
        print("\n" + "=" * 72)
        print("  FINAL EVALUATION (best adapter)")
        print("=" * 72)

        # Free training model before loading best adapter
        del model, optimizer, scheduler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        best_dir = os.path.join(args.output_dir, "best_adapter")
        if os.path.exists(best_dir):
            from peft import PeftModel
            reload_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                dtype=reload_dtype,
                trust_remote_code=True,
            )
            best_model = PeftModel.from_pretrained(base_model, best_dir)
            best_model = best_model.to(accelerator.device)
            best_model.eval()

            metrics = evaluate(
                best_model, eval_loader, tokenizer, accelerator,
                max_new_tokens=args.max_new_tokens,
                max_context_tokens=args.max_context_tokens,
                show_samples=10,
            )
            print(f"\n  Final:  Loss={metrics['loss']:.4f}  "
                  f"PPL={metrics['perplexity']:.2f}  "
                  f"EM={metrics['exact_match']:.4f}  "
                  f"F1={metrics['f1']:.4f}  "
                  f"RL={metrics['rouge_l']:.4f}")

            # Save metrics to JSON
            metrics_path = os.path.join(args.output_dir, "eval_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"  Metrics saved to {metrics_path}")

        print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune Qwen on QA data (upper bound baseline)")

    # Model
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to Qwen model (e.g. models/Qwen3-0.6B)")

    # Data
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to training QA JSON file")
    parser.add_argument("--eval_data", type=str, default=None,
                        help="Path to eval QA JSON file")
    parser.add_argument("--max_eval_samples", type=int, default=None,
                        help="Cap eval set size for faster evaluation")
    parser.add_argument("--max_context_tokens", type=int, default=4096,
                        help="Max context tokens (default: 4096)")
    parser.add_argument("--max_answer_tokens", type=int, default=512,
                        help="Max answer tokens (default: 512)")

    # Training
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-GPU batch size")
    parser.add_argument("--eval_batch_size", type=int, default=8,
                        help="Per-GPU eval batch size")
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Warmup steps (0 = auto 5%% of total)")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to save memory")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers (forced 0 on MPS)")

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Evaluation
    parser.add_argument("--eval_every_steps", type=int, default=500,
                        help="Evaluate every N optimizer steps (0 = end of epoch only)")
    parser.add_argument("--max_new_tokens", type=int, default=64,
                        help="Max tokens to generate during eval")
    parser.add_argument("--log_every_steps", type=int, default=50,
                        help="Print training loss every N steps")

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/lora_qwen",
                        help="Output directory for adapters and metrics")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
