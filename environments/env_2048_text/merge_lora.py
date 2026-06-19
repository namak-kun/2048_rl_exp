#!/usr/bin/env python3
"""Merge a LoRA adapter into its base model for vLLM serving."""
import argparse
import os
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"Loading base {args.base}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(args.adapter, use_fast=True)
    print(f"Loading adapter {args.adapter}", flush=True)
    peft = PeftModel.from_pretrained(base, args.adapter)
    print("Merging", flush=True)
    merged = peft.merge_and_unload()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
