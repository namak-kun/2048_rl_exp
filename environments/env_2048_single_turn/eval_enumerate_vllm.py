#!/usr/bin/env python3
"""Fast vLLM eval for the enumerate-valid-moves task (JSON format).

Reports:
  - exact_match (EM)
  - F1 over the valid-moves set
  - blended (0.7 EM + 0.3 F1) matching the training reward
  - format_parsed: response had a parseable {"valid_moves": [...]} object
  - has_think: model produced a <think>...</think> block (with content)
  - avg completion length in chars
  - per-difficulty breakdown (2/3/4 valid moves)

Usage:
    python eval_enumerate_vllm.py \\
        --model Qwen/Qwen3-0.6B \\
        --num_per_condition 150 \\
        --grid_sizes 4 5 6 \\
        --out_json eval_results/base_enum.json
"""

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

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
    get_valid_direction_names,
    _extract_valid_moves_set,
    _f1,
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


def parse_response(text: str) -> tuple[str, frozenset | None]:
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    think_text = think_match.group(1).strip() if think_match else ""
    after = re.sub(r"<think>.*?</think>\s*", "", text, count=1, flags=re.DOTALL)
    pred = _extract_valid_moves_set(after)
    return think_text, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--num_per_condition", type=int, default=150)
    ap.add_argument("--max_new_tokens_think", type=int, default=4096)
    ap.add_argument("--max_new_tokens_nothink", type=int, default=256)
    ap.add_argument("--grid_sizes", type=int, nargs="+", default=[4, 5, 6])
    ap.add_argument("--think_modes", nargs="+", default=["false", "true"])
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.4)
    ap.add_argument("--max_model_len", type=int, default=8192)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"Loading vLLM model: {args.model}", flush=True)
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=False,
    )
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    results = {
        "model": args.model,
        "task": "enumerate_all",
        "num_per_condition": args.num_per_condition,
        "conditions": {},
    }

    for size in args.grid_sizes:
        boards = make_balanced_boards(args.num_per_condition, size=size, seed=args.seed + size)
        diff_dist = {k: sum(1 for _, nv in boards if nv == k) for k in (2, 3, 4)}
        print(f"\n=== Grid {size}x{size} (n={len(boards)}, diff={diff_dist}) ===", flush=True)

        for think_str in args.think_modes:
            think = think_str.lower() == "true"
            tag = f"{size}x{size}_think={think}"
            print(f"  → {tag}", flush=True)
            sys_p = get_system_prompt(size, 2048, output_format="json", task_type="enumerate_all")
            prompts = []
            true_sets = []
            for game, nv in boards:
                user_p = get_user_prompt(game, output_format="json", task_type="enumerate_all")
                txt = tok.apply_chat_template(
                    [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=think,
                )
                prompts.append(txt)
                true_sets.append(frozenset(get_valid_direction_names(game)))

            max_tokens = args.max_new_tokens_think if think else args.max_new_tokens_nothink
            sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=max_tokens)
            outs = llm.generate(prompts, sp, use_tqdm=True)

            per_difficulty = defaultdict(lambda: {"n": 0, "parsed": 0, "em": 0, "f1_sum": 0.0,
                                                  "blend_sum": 0.0, "has_think": 0, "len_sum": 0})
            overall = {"n": 0, "parsed": 0, "em": 0, "f1_sum": 0.0,
                       "blend_sum": 0.0, "has_think": 0, "len_sum": 0}
            examples = []
            for (game, nv), o, true_set in zip(boards, outs, true_sets):
                text = o.outputs[0].text
                think_text, pred = parse_response(text)
                parsed = pred is not None
                if parsed:
                    em = 1 if pred == true_set else 0
                    f1 = _f1(pred, true_set)
                else:
                    em, f1 = 0, 0.0
                blend = 0.7 * em + 0.3 * f1

                for d in (overall, per_difficulty[nv]):
                    d["n"] += 1
                    d["parsed"] += int(parsed)
                    d["em"] += em
                    d["f1_sum"] += f1
                    d["blend_sum"] += blend
                    d["has_think"] += int(len(think_text) > 5)
                    d["len_sum"] += len(text)

                if len(examples) < 3:
                    examples.append({
                        "valid_dirs": list(true_set),
                        "predicted": list(pred) if pred else None,
                        "em": em, "f1": f1,
                        "think_chars": len(think_text),
                        "finish_reason": o.outputs[0].finish_reason,
                        "text_snippet": text[:400],
                    })

            def fmt(d):
                n = max(1, d["n"])
                return {
                    "n": d["n"],
                    "parsed_pct": 100 * d["parsed"] / n,
                    "em_pct": 100 * d["em"] / n,
                    "f1_avg": d["f1_sum"] / n,
                    "blend_avg": d["blend_sum"] / n,
                    "has_think_pct": 100 * d["has_think"] / n,
                    "avg_chars": d["len_sum"] / n,
                }

            results["conditions"][tag] = {
                "overall": overall,
                "per_difficulty": {str(k): v for k, v in per_difficulty.items()},
                "examples": examples,
            }
            o = fmt(overall)
            print(f"     parsed={o['parsed_pct']:.0f}% EM={o['em_pct']:.0f}% F1={o['f1_avg']:.3f} "
                  f"blend={o['blend_avg']:.3f} think={o['has_think_pct']:.0f}% avg_chars={o['avg_chars']:.0f}",
                  flush=True)
            for nv in (2, 3, 4):
                pd = per_difficulty.get(nv)
                if pd and pd["n"]:
                    f = fmt(pd)
                    print(f"        diff={nv} (n={f['n']}): "
                          f"EM={f['em_pct']:.0f}% F1={f['f1_avg']:.3f} blend={f['blend_avg']:.3f}",
                          flush=True)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved JSON to {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
