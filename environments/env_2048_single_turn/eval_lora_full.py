#!/usr/bin/env python3
"""
Comprehensive evaluation of a LoRA-merged checkpoint.

For each combination of (grid_size, enable_thinking), runs N rollouts on
balanced-difficulty boards and reports:
  - format adherence
  - valid-move accuracy (overall and per difficulty)
  - whether thinking blocks appear
  - average response length

Loads the model once so we don't pay reload cost per condition.

Usage:
    python eval_lora_full.py \
        --model lora_outputs_json/merged \
        --output_format json \
        --num_per_condition 100 \
        --max_new_tokens 4096 \
        --grid_sizes 4 5 6 \
        --out_json results/lora_json_eval.json
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from env_2048_single_turn import (  # noqa: E402
    Game2048,
    Grid,
    count_valid_moves,
    generate_playthrough_states,
    get_system_prompt,
    get_user_prompt,
    _extract_move_xml,
    _extract_move_json,
)


def make_balanced_boards(n: int, size: int, seed: int) -> list[Game2048]:
    """Return n boards: 75% 2-valid, 12.5% 3-valid, 12.5% 4-valid."""
    random.seed(seed)
    target = {2: int(n * 0.75), 3: int(n * 0.125), 4: int(n * 0.125)}
    target[2] += n - sum(target.values())
    buckets = {k: [] for k in target}
    iters = 0
    max_iters = n * 100
    while sum(len(b) for b in buckets.values()) < n and iters < max_iters:
        iters += 1
        fill = random.uniform(0.5, 0.95)
        for g in generate_playthrough_states(size=size, fill_ratio=fill, snapshot_interval=2):
            nv = count_valid_moves(g)
            if nv in buckets and len(buckets[nv]) < target[nv]:
                buckets[nv].append(g)
    flat = []
    for nv, items in buckets.items():
        for g in items:
            flat.append((g, nv))
    random.shuffle(flat)
    return flat


def parse_response(text: str, fmt: str) -> tuple[str, str]:
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    think_text = think_match.group(1).strip() if think_match else ""
    after = re.sub(r"<think>.*?</think>\s*", "", text, count=1, flags=re.DOTALL)
    extract = _extract_move_xml if fmt == "xml" else _extract_move_json
    return think_text, (extract(after) or "")


@torch.no_grad()
def generate_batch(model, tok, prompts: list[str], max_new_tokens: int, batch_size: int = 4) -> list[str]:
    """Batched generation returning the generated portion only (skip prompt prefix)."""
    out = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=False).to("cuda")
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
        for j in range(len(batch)):
            input_len = enc.attention_mask[j].sum().item()
            generated_ids = gen[j, input_len:]
            out.append(tok.decode(generated_ids, skip_special_tokens=False))
    return out


def evaluate_condition(
    model, tok, boards: list[tuple[Game2048, int]],
    fmt: str, enable_thinking: bool, max_new_tokens: int, batch_size: int,
    size: int,
) -> dict:
    sys_p = get_system_prompt(grid_size=size, target_tile=2048, output_format=fmt)
    prompts = []
    for game, _ in boards:
        user_p = get_user_prompt(game, output_format=fmt)
        prompt_text = tok.apply_chat_template(
            [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        prompts.append(prompt_text)

    completions = generate_batch(model, tok, prompts, max_new_tokens, batch_size=batch_size)

    per_difficulty = defaultdict(lambda: {"n": 0, "parsed": 0, "valid": 0, "has_think": 0, "len_sum": 0})
    overall = {"n": 0, "parsed": 0, "valid": 0, "has_think": 0, "len_sum": 0}
    examples = []
    for (game, nv), text in zip(boards, completions):
        think_text, move = parse_response(text, fmt)
        parsed = False
        moved = False
        if move:
            d = Game2048.parse_move(move)
            if d is not None:
                parsed = True
                test_game = Game2048(
                    size=game.size,
                    target_tile=game.target_tile,
                    grid=Grid(size=game.size, cells=[r[:] for r in game.grid.cells]),
                )
                moved = test_game.move(d)
        b = per_difficulty[nv]
        b["n"] += 1
        b["parsed"] += int(parsed)
        b["valid"] += int(moved)
        b["has_think"] += int(len(think_text) > 5)
        b["len_sum"] += len(text)
        overall["n"] += 1
        overall["parsed"] += int(parsed)
        overall["valid"] += int(moved)
        overall["has_think"] += int(len(think_text) > 5)
        overall["len_sum"] += len(text)
        if len(examples) < 3:
            examples.append({
                "valid_dirs": nv,
                "think_chars": len(think_text),
                "move": move,
                "parsed": parsed,
                "moved": moved,
                "text_snippet": text[:300],
            })
    return {
        "overall": overall,
        "per_difficulty": {str(k): v for k, v in per_difficulty.items()},
        "examples": examples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output_format", choices=["xml", "json"], required=True)
    ap.add_argument("--num_per_condition", type=int, default=100)
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument("--grid_sizes", type=int, nargs="+", default=[4, 5, 6])
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    print(f"Loading model: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    results = {"model": args.model, "output_format": args.output_format,
               "num_per_condition": args.num_per_condition, "conditions": {}}

    for size in args.grid_sizes:
        boards = make_balanced_boards(args.num_per_condition, size=size, seed=args.seed + size)
        print(f"\n=== Grid {size}x{size} (n={len(boards)}, "
              f"diff dist: {dict([(k, sum(1 for _, nv in boards if nv == k)) for k in (2, 3, 4)])}) ===")
        for think in [False, True]:
            tag = f"{size}x{size}_think={think}"
            print(f"  → {tag}")
            res = evaluate_condition(model, tok, boards, args.output_format, think,
                                     args.max_new_tokens if think else 256,
                                     args.batch_size, size)
            results["conditions"][tag] = res
            o = res["overall"]
            n = o["n"]
            print(f"     parsed={o['parsed']}/{n} ({100 * o['parsed'] / n:.0f}%), "
                  f"valid={o['valid']}/{n} ({100 * o['valid'] / n:.0f}%), "
                  f"has_think={o['has_think']}/{n} ({100 * o['has_think'] / n:.0f}%), "
                  f"avg_chars={o['len_sum'] / n:.0f}")
            for nv in (2, 3, 4):
                pd = res["per_difficulty"].get(str(nv))
                if pd and pd["n"]:
                    print(f"        diff={nv} valid: {pd['valid']}/{pd['n']} ({100 * pd['valid'] / pd['n']:.0f}%)")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved JSON results to {args.out_json}")


if __name__ == "__main__":
    main()
