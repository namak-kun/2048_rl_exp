#!/usr/bin/env python3
"""
Probe whether a LoRA-merged checkpoint preserves Qwen3's <think> blocks when
asked to think at inference time.

Generates a small number of completions in two modes (enable_thinking True/False)
on hold-out 2048 boards and prints:
  - Whether the response contains a non-empty <think>...</think> block
  - The parsed move (XML or JSON, depending on --output_format)
  - Whether that move is valid

Usage:
    python probe_thinking.py \
        --model lora_outputs_xml/merged \
        --output_format xml \
        --num_examples 10
"""

import argparse
import json
import os
import random
import re
import sys
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
    make_valid_move_reward,
    _extract_move_xml,
    _extract_move_json,
)


def make_probe_boards(n: int, size: int = 4, seed: int = 999) -> list[Game2048]:
    """Return n boards with exactly 2 valid moves (hardest)."""
    random.seed(seed)
    out = []
    iters = 0
    while len(out) < n and iters < n * 100:
        iters += 1
        fill_ratio = random.uniform(0.6, 0.95)
        states = generate_playthrough_states(size=size, fill_ratio=fill_ratio, snapshot_interval=2)
        for g in states:
            if count_valid_moves(g) == 2:
                out.append(g)
                if len(out) >= n:
                    break
    return out


def parse_response(text: str, fmt: str) -> tuple[str, str]:
    """Return (think_text, move_or_empty)."""
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    think_text = think_match.group(1).strip() if think_match else ""

    # The model output may have think pre-stripped (via reasoning parser) — handle both
    after_think = re.sub(r"<think>.*?</think>\s*", "", text, count=1, flags=re.DOTALL)
    extract = _extract_move_xml if fmt == "xml" else _extract_move_json
    move = extract(after_think) or ""
    return think_text, move


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output_format", choices=["xml", "json"], default="xml")
    ap.add_argument("--num_examples", type=int, default=10)
    ap.add_argument("--grid_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=999)
    args = ap.parse_args()

    print(f"Loading {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    boards = make_probe_boards(args.num_examples, size=args.grid_size, seed=args.seed)
    system_prompt = get_system_prompt(
        grid_size=args.grid_size, target_tile=2048, output_format=args.output_format
    )

    results = {True: [], False: []}

    for think_mode in [True, False]:
        print(f"\n{'=' * 60}")
        print(f"=== enable_thinking={think_mode} ===")
        print(f"{'=' * 60}")
        for i, game in enumerate(boards):
            user = get_user_prompt(game, output_format=args.output_format)
            msgs = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user},
            ]
            prompt_text = tok.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=think_mode,
            )
            inputs = tok(prompt_text, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
            full = tok.decode(out_ids[0], skip_special_tokens=True)
            gen_only = full[len(tok.decode(inputs.input_ids[0], skip_special_tokens=True)):]

            think_text, move = parse_response(gen_only, args.output_format)
            valid_dirs_count = count_valid_moves(game)

            # Check move validity
            move_valid = False
            move_changed = False
            if move:
                d = Game2048.parse_move(move)
                if d is not None:
                    test_game = Game2048(
                        size=game.size, target_tile=game.target_tile,
                        grid=Grid(size=game.size, cells=[r[:] for r in game.grid.cells]),
                    )
                    move_changed = test_game.move(d)
                    move_valid = True

            think_len = len(think_text)
            print(f"\n[#{i}] valid_dirs={valid_dirs_count} think_chars={think_len} move='{move}' "
                  f"parsed={move_valid} moved_tiles={move_changed}")
            if think_len > 0:
                snippet = think_text[:200].replace("\n", " ")
                print(f"   think snippet: {snippet}{'...' if think_len > 200 else ''}")
            results[think_mode].append({
                "valid_dirs": valid_dirs_count,
                "think_len": think_len,
                "has_think": think_len > 5,  # threshold to exclude empty-ish
                "move": move,
                "move_parsed": move_valid,
                "move_changed_board": move_changed,
            })

    print("\n" + "=" * 60)
    print("=== Summary ===")
    print("=" * 60)
    for mode, rs in results.items():
        n = len(rs)
        with_think = sum(1 for r in rs if r["has_think"])
        avg_think_len = sum(r["think_len"] for r in rs) / max(1, n)
        valid_format = sum(1 for r in rs if r["move_parsed"])
        valid_move = sum(1 for r in rs if r["move_changed_board"])
        print(f"\nenable_thinking={mode} (n={n}):")
        print(f"  responses with <think> content: {with_think}/{n} ({100 * with_think / n:.0f}%)")
        print(f"  avg think length (chars):       {avg_think_len:.1f}")
        print(f"  parseable move format:          {valid_format}/{n} ({100 * valid_format / n:.0f}%)")
        print(f"  valid moves (board changed):    {valid_move}/{n} ({100 * valid_move / n:.0f}%)")


if __name__ == "__main__":
    main()
