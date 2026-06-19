#!/usr/bin/env python3
"""
LoRA SFT for the 2048 single-turn valid-move task.

Uses HuggingFace TRL's SFTTrainer + PEFT LoRA. Outputs both the raw adapter and
a merged (LoRA folded into base) checkpoint, so the merged checkpoint can be
loaded by prime-rl / vLLM as a regular HF model for downstream RL.

Trained on either the XML or JSON output-format dataset (see generate_sft_data.py).

Usage:
    python train_lora_sft.py \\
        --data_dir sft_data_xml \\
        --output_dir lora_outputs_xml \\
        --model Qwen/Qwen3-0.6B \\
        --num_epochs 2 \\
        --batch_size 4 \\
        --grad_accum 8 \\
        --lr 2e-4 \\
        --rank 16 --alpha 32 \\
        --merge

Run with accelerate for multi-GPU:
    accelerate launch train_lora_sft.py ...
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def load_jsonl_dataset(path: str) -> Dataset:
    """Load a {prompt, completion} JSONL file as a HF Dataset.

    TRL's SFTTrainer accepts the prompt/completion conversational format
    directly when the chat_template is set on the tokenizer.
    """
    p = Path(path)
    if p.is_dir():
        train_path = p / "train.jsonl"
    else:
        train_path = p
    if not train_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {train_path}")
    return load_dataset("json", data_files=str(train_path), split="train")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Directory containing train.jsonl")
    ap.add_argument("--output_dir", required=True, help="Where to save adapter (and optionally merged model)")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--num_epochs", type=float, default=2.0)
    ap.add_argument("--max_steps", type=int, default=-1, help="Override num_epochs if > 0")
    ap.add_argument("--batch_size", type=int, default=4, help="Per-device train batch size")
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument(
        "--target_modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--merge",
        action="store_true",
        help="After training, merge LoRA into base and save merged HF model alongside the adapter.",
    )
    ap.add_argument(
        "--merge_only",
        action="store_true",
        help="Skip training; just merge an existing adapter at --output_dir/adapter into a merged checkpoint.",
    )
    return ap.parse_args()


def train(args: argparse.Namespace) -> None:
    print(f"Loading dataset from {args.data_dir}")
    ds = load_jsonl_dataset(args.data_dir)
    print(f"Loaded {len(ds)} examples")
    print(f"Example: prompt roles={[m['role'] for m in ds[0]['prompt']]}, "
          f"completion={ds[0]['completion'][0]['content']!r}")

    print(f"Loading tokenizer & model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if args.bf16 else torch.float32,
        attn_implementation="sdpa",
    )

    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        target_modules=args.target_modules,
        task_type="CAUSAL_LM",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = output_dir / "adapter"
    merged_dir = output_dir / "merged"
    trainer_dir = output_dir / "trainer"

    sft_config = SFTConfig(
        output_dir=str(trainer_dir),
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        max_length=args.max_length,
        completion_only_loss=True,
        packing=False,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    sanity_check_supervised_tokens(trainer, tokenizer)

    print("Starting training...")
    trainer.train()

    print(f"Saving adapter to {adapter_dir}")
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # Save run config for reproducibility
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    if args.merge:
        merge_adapter(args.model, adapter_dir, merged_dir, bf16=args.bf16)


def sanity_check_supervised_tokens(trainer, tokenizer, n: int = 2) -> None:
    """Decode the first n examples' supervised tokens to confirm only the answer is trained on.

    Fails loudly if the supervised slice contains any non-completion content beyond what we expect.
    """
    train_ds = trainer.train_dataset
    sample_indices = list(range(min(n, len(train_ds))))

    for i in sample_indices:
        ex = train_ds[i]
        input_ids = ex["input_ids"]
        completion_mask = ex.get("completion_mask")
        if completion_mask is None:
            raise RuntimeError(
                "Sanity check: example has no 'completion_mask'. completion_only_loss may not be active."
            )
        supervised = [tid for tid, m in zip(input_ids, completion_mask) if m == 1]
        unsupervised = [tid for tid, m in zip(input_ids, completion_mask) if m == 0]
        supervised_text = tokenizer.decode(supervised, skip_special_tokens=False)
        prompt_tail = tokenizer.decode(unsupervised[-50:], skip_special_tokens=False)

        print(f"\n--- Sanity check sample #{i} ---")
        print(f"prompt tail (last 50 tokens, NOT trained on): {prompt_tail!r}")
        print(f"supervised (TRAINED ON): {supervised_text!r}")

        if "<think>" in supervised_text or "</think>" in supervised_text:
            raise RuntimeError(
                "Sanity check failed: supervised tokens contain a <think> block. "
                "Did you forget chat_template_kwargs={'enable_thinking': False}?"
            )

    print("--- Sanity check passed ---\n")


def merge_adapter(base_model_name: str, adapter_dir: Path, merged_dir: Path, bf16: bool = True) -> None:
    print(f"Merging adapter from {adapter_dir} into base {base_model_name} -> {merged_dir}")
    merged_dir.mkdir(parents=True, exist_ok=True)

    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.bfloat16 if bf16 else torch.float32,
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model = model.merge_and_unload()
    model.save_pretrained(str(merged_dir), safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    tokenizer.save_pretrained(str(merged_dir))
    print(f"Merged model saved to {merged_dir}")


def main():
    args = parse_args()

    if args.merge_only:
        output_dir = Path(args.output_dir)
        adapter_dir = output_dir / "adapter"
        merged_dir = output_dir / "merged"
        if not adapter_dir.exists():
            raise FileNotFoundError(f"No adapter found at {adapter_dir} for --merge_only")
        merge_adapter(args.model, adapter_dir, merged_dir, bf16=args.bf16)
        return

    train(args)


if __name__ == "__main__":
    main()
