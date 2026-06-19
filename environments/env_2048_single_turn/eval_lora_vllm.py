#!/usr/bin/env python3
"""Fast batched evaluation using vLLM's offline engine."""

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

# Avoid prime_rl env-import quirks
os.environ.setdefault("VLLM_USE_V1", "1")

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


def make_balanced_boards(n: int, size: int, seed: int) -> list[tuple]:
    random.seed(seed)
    target = {2: int(n * 0.75), 3: int(n * 0.125), 4: int(n * 0.125)}
    target[2] += n - sum(target.values())
    buckets = {k: [] for k in target}
    iters = 0
    while sum(len(b) for b in buckets.values()) < n and iters < n * 100:
        iters += 1
        for g in generate_playthrough_states(size=size, fill_ratio=random.uniform(0.5, 0.95), snapshot_interval=2):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output_format", choices=["xml", "json"], required=True)
    ap.add_argument("--num_per_condition", type=int, default=100)
    ap.add_argument("--max_new_tokens_think", type=int, default=4096)
    ap.add_argument("--max_new_tokens_nothink", type=int, default=64)
    ap.add_argument("--grid_sizes", type=int, nargs="+", default=[4, 5, 6])
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.5)
    ap.add_argument("--max_model_len", type=int, default=8192)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"Loading vLLM model: {args.model}")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=False,
    )
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    results = {"model": args.model, "output_format": args.output_format,
               "num_per_condition": args.num_per_condition, "conditions": {}}

    for size in args.grid_sizes:
        boards = make_balanced_boards(args.num_per_condition, size=size, seed=args.seed + size)
        diff_dist = {k: sum(1 for _, nv in boards if nv == k) for k in (2, 3, 4)}
        print(f"\n=== Grid {size}x{size} (n={len(boards)}, diff={diff_dist}) ===")

        for think in [False, True]:
            tag = f"{size}x{size}_think={think}"
            print(f"  → {tag}", flush=True)
            sys_p = get_system_prompt(size, 2048, args.output_format)
            prompts = []
            for game, _ in boards:
                user_p = get_user_prompt(game, args.output_format)
                txt = tok.apply_chat_template(
                    [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=think,
                )
                prompts.append(txt)

            max_tokens = args.max_new_tokens_think if think else args.max_new_tokens_nothink
            sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=max_tokens)
            outs = llm.generate(prompts, sp, use_tqdm=True)

            per_difficulty = defaultdict(lambda: {"n": 0, "parsed": 0, "valid": 0, "has_think": 0, "len_sum": 0})
            overall = {"n": 0, "parsed": 0, "valid": 0, "has_think": 0, "len_sum": 0}
            examples = []
            for (game, nv), o in zip(boards, outs):
                text = o.outputs[0].text
                think_text, move = parse_response(text, args.output_format)
                parsed = False
                moved = False
                if move:
                    d = Game2048.parse_move(move)
                    if d is not None:
                        parsed = True
                        test_game = Game2048(
                            size=game.size, target_tile=game.target_tile,
                            grid=Grid(size=game.size, cells=[r[:] for r in game.grid.cells]),
                        )
                        moved = test_game.move(d)
                b = per_difficulty[nv]
                for k, v in [("n", 1), ("parsed", int(parsed)), ("valid", int(moved)),
                             ("has_think", int(len(think_text) > 5)), ("len_sum", len(text))]:
                    b[k] += v
                    overall[k] += v
                if len(examples) < 3:
                    examples.append({
                        "valid_dirs": nv, "think_chars": len(think_text),
                        "move": move, "parsed": parsed, "moved": moved,
                        "text_snippet": text[:300],
                    })
            results["conditions"][tag] = {
                "overall": overall,
                "per_difficulty": {str(k): v for k, v in per_difficulty.items()},
                "examples": examples,
            }
            n = overall["n"]
            print(f"     parsed={overall['parsed']}/{n} ({100 * overall['parsed'] / n:.0f}%), "
                  f"valid={overall['valid']}/{n} ({100 * overall['valid'] / n:.0f}%), "
                  f"has_think={overall['has_think']}/{n} ({100 * overall['has_think'] / n:.0f}%), "
                  f"avg_chars={overall['len_sum'] / n:.0f}", flush=True)
            for nv in (2, 3, 4):
                pd = per_difficulty.get(nv)
                if pd and pd["n"]:
                    print(f"        diff={nv} valid: {pd['valid']}/{pd['n']} ({100 * pd['valid'] / pd['n']:.0f}%)", flush=True)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved JSON to {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
