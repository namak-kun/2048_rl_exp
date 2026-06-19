#!/usr/bin/env python3
"""Smoke-test: feed the multi-turn env's pick_one prompt to a candidate seed model
and see what it outputs. Specifically check if it's stuck producing the enumerate
format `{"valid_moves":[...]}` or if it adapts to `{"move":"..."}`.

Usage:
    python smoke_test_seed.py --model path/to/ckpt
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import os
os.environ.setdefault("VLLM_USE_V1", "1")

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from env_2048_text import (
    Game2048,
    get_system_prompt,
    _extract_move_json,
)


def make_prompts(n, sizes=(4,), target=2048, seed=42):
    import random
    random.seed(seed)
    prompts = []
    for _ in range(n):
        size = random.choice(sizes)
        game = Game2048(size=size, target_tile=target)
        sys_p = get_system_prompt(grid_size=size, target_tile=target, output_format="json")
        user_p = (
            f"Let's play 2048! Here's the starting board ({size}x{size} grid, target: {target}):\n\n"
            f"{game.get_state_text()}\n\n"
            'What\'s your first move? Respond with {"move": "direction"}.'
        )
        prompts.append((sys_p, user_p))
    return prompts


def classify_output(text):
    has_think = bool(re.search(r"<think>", text))
    has_valid_moves_key = bool(re.search(r'"valid_moves"', text))
    has_move_key = bool(re.search(r'"move"', text, re.IGNORECASE))
    move = _extract_move_json(text)
    return {
        "has_think": has_think,
        "has_valid_moves_key": has_valid_moves_key,
        "has_move_key": has_move_key,
        "parseable_move": move,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"Loading {args.model}", flush=True)
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=8192,
    )
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    print(f"Generating {args.n} prompts (multi-turn 2048 first-move format)...", flush=True)
    prompts = make_prompts(args.n)
    rendered = []
    for sys_p, user_p in prompts:
        txt = tok.apply_chat_template(
            [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        rendered.append(txt)

    sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=args.max_tokens)
    outs = llm.generate(rendered, sp, use_tqdm=True)

    results = []
    stats = Counter()
    move_counts = Counter()
    for o in outs:
        text = o.outputs[0].text
        info = classify_output(text)
        info["finish_reason"] = o.outputs[0].finish_reason
        info["len_chars"] = len(text)
        stats["has_think"] += int(info["has_think"])
        stats["has_valid_moves_key"] += int(info["has_valid_moves_key"])
        stats["has_move_key"] += int(info["has_move_key"])
        stats["parseable_move"] += int(info["parseable_move"] is not None)
        stats["truncated"] += int(o.outputs[0].finish_reason == "length")
        if info["parseable_move"]:
            move_counts[info["parseable_move"]] += 1
        results.append({
            "info": info,
            "text_tail": text[-400:],
            "text_head": text[:200],
        })

    n = max(1, args.n)
    print()
    print("=" * 70)
    print(f"Summary (n={n}):")
    print(f"  has_think:           {100*stats['has_think']/n:.1f}%")
    print(f"  output has 'valid_moves' key: {100*stats['has_valid_moves_key']/n:.1f}%  (enum format remnant)")
    print(f"  output has 'move' key:        {100*stats['has_move_key']/n:.1f}%  (correct pick_one)")
    print(f"  parseable single move (any path): {100*stats['parseable_move']/n:.1f}%")
    print(f"  truncated at max_tokens:          {100*stats['truncated']/n:.1f}%")
    print(f"  move distribution: {dict(move_counts)}")
    print()
    print("First 3 examples:")
    for i, r in enumerate(results[:3]):
        print(f"--- Example {i} ---")
        print(f"  parsed: {r['info']['parseable_move']}  has_valid_moves: {r['info']['has_valid_moves_key']}  has_move: {r['info']['has_move_key']}  finish: {r['info']['finish_reason']}")
        print(f"  HEAD: {r['text_head'][:200]}")
        print(f"  TAIL: {r['text_tail'][-200:]}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"model": args.model, "n": n, "stats": dict(stats),
                       "move_counts": dict(move_counts), "results": results}, f, indent=2)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
