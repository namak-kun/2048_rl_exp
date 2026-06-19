#!/usr/bin/env python3
"""Two diagnostics for the collapsed (predict-2-always) lora_prp model:

  Q1. PRECISION: when the model predicts 2 directions on a 3-valid or 4-valid board,
      are those 2 directions a SUBSET of the true valid set? (i.e. predicted moves are valid)
  Q2. THINK COVERAGE: in think mode, does the model mention all 4 directions
      ('up'/'down'/'left'/'right') in its reasoning, even if it only outputs 2?

Saves full think traces + final outputs for 4-valid boards so you can read what the
model is actually thinking.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("VLLM_USE_V1", "1")
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from env_2048_single_turn import (  # noqa: E402
    count_valid_moves,
    generate_playthrough_states,
    get_system_prompt,
    get_user_prompt,
    get_valid_direction_names,
    _extract_valid_moves_set,
)


def parse_response(text):
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    think_text = think_match.group(1).strip() if think_match else ""
    after = re.sub(r"<think>.*?</think>\s*", "", text, count=1, flags=re.DOTALL)
    pred = _extract_valid_moves_set(after)
    return think_text, pred


def make_boards(n_per_difficulty, sizes, seed=42):
    import random
    random.seed(seed)
    boards = []
    for size in sizes:
        target = {2: n_per_difficulty, 3: n_per_difficulty, 4: n_per_difficulty}
        bucket = {k: [] for k in target}
        iters = 0
        max_iters = n_per_difficulty * 200
        while sum(len(b) for b in bucket.values()) < sum(target.values()) and iters < max_iters:
            iters += 1
            for g in generate_playthrough_states(size=size, fill_ratio=random.uniform(0.4, 0.95), snapshot_interval=2):
                nv = count_valid_moves(g)
                if nv in bucket and len(bucket[nv]) < target[nv]:
                    bucket[nv].append(g)
        for nv, items in bucket.items():
            for g in items:
                boards.append((g, nv, size))
    return boards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n_per_difficulty", type=int, default=40,
                    help="boards per difficulty bucket per grid-size (so total = 3 * n * |sizes|)")
    ap.add_argument("--sizes", type=int, nargs="+", default=[4])
    ap.add_argument("--mode", choices=["think", "nothink", "both"], default="both")
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--max_tokens_think", type=int, default=4096)
    ap.add_argument("--max_tokens_nothink", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"Loading {args.model}", flush=True)
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.5, max_model_len=8192)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    boards = make_boards(args.n_per_difficulty, args.sizes, seed=args.seed)
    bucket_counts = Counter(nv for _, nv, _ in boards)
    print(f"Boards: total={len(boards)} per_difficulty={dict(bucket_counts)}", flush=True)

    modes = []
    if args.mode in ("think", "both"): modes.append(True)
    if args.mode in ("nothink", "both"): modes.append(False)

    results = {"model": args.model, "modes": {}}

    for think in modes:
        mode_name = "think" if think else "nothink"
        print(f"\n=== mode={mode_name} ===", flush=True)
        prompts = []
        truths = []
        for game, _, size in boards:
            sys_p = get_system_prompt(size, 2048, output_format="json", task_type="enumerate_all")
            user_p = get_user_prompt(game, output_format="json", task_type="enumerate_all")
            txt = tok.apply_chat_template(
                [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
                tokenize=False, add_generation_prompt=True, enable_thinking=think,
            )
            prompts.append(txt)
            truths.append(frozenset(get_valid_direction_names(game)))

        max_tokens = args.max_tokens_think if think else args.max_tokens_nothink
        sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=max_tokens)
        outs = llm.generate(prompts, sp, use_tqdm=True)

        # Per difficulty stats
        DIRS = ["up", "down", "left", "right"]
        per_diff = defaultdict(lambda: {
            "n": 0,
            "pred_size_counts": Counter(),
            "subset_of_true": 0,           # predicted ⊆ true (precision==1.0)
            "exact_match": 0,
            "think_mentions_all_4": 0,     # think text contains all 4 dir names
            "think_mentions_count_sum": 0, # avg # of distinct dir names mentioned in think
            "parsed": 0,
        })
        examples_to_save = defaultdict(list)  # per difficulty, save a handful

        for (game, nv, size), o, truth in zip(boards, outs, truths):
            text = o.outputs[0].text
            think_text, pred = parse_response(text)
            d = per_diff[nv]
            d["n"] += 1
            d["parsed"] += int(pred is not None)
            if pred is not None:
                d["pred_size_counts"][len(pred)] += 1
                if pred.issubset(truth) and len(pred) > 0:
                    d["subset_of_true"] += 1
                if pred == truth:
                    d["exact_match"] += 1
            mentioned = sum(1 for dir in DIRS if re.search(rf"\b{dir}\b", think_text.lower()))
            d["think_mentions_count_sum"] += mentioned
            if mentioned == 4:
                d["think_mentions_all_4"] += 1

            # Save up to 6 examples per (difficulty, size) for inspection
            key = (nv, size)
            if len(examples_to_save[key]) < 6:
                examples_to_save[key].append({
                    "size": size,
                    "true_size": nv,
                    "true_valid": sorted(truth),
                    "pred": sorted(pred) if pred else None,
                    "pred_size": len(pred) if pred else 0,
                    "subset_of_true": bool(pred and pred.issubset(truth) and len(pred) > 0),
                    "em": int(pred == truth),
                    "think_chars": len(think_text),
                    "think_text": think_text,
                    "final_text": re.sub(r"<think>.*?</think>\s*", "", text, count=1, flags=re.DOTALL)[:300],
                    "finish": o.outputs[0].finish_reason,
                })

        # Print summary
        print(f"\n{'diff':<6}{'n':<6}{'EM%':<7}{'subset%':<10}{'pred_sizes':<22}{'mean_dirs_in_think':<22}{'all4_in_think%':<14}")
        mode_out = {"per_difficulty": {}}
        for nv in sorted(per_diff.keys()):
            d = per_diff[nv]
            n = max(1, d["n"])
            ps = dict(d["pred_size_counts"])
            print(f"{nv:<6}{d['n']:<6}"
                  f"{100*d['exact_match']/n:<7.0f}"
                  f"{100*d['subset_of_true']/n:<10.0f}"
                  f"{str(ps):<22}"
                  f"{d['think_mentions_count_sum']/n:<22.2f}"
                  f"{100*d['think_mentions_all_4']/n:<14.0f}")
            mode_out["per_difficulty"][nv] = {
                "n": d["n"],
                "em_pct": 100 * d["exact_match"] / n,
                "subset_of_true_pct": 100 * d["subset_of_true"] / n,
                "pred_size_counts": ps,
                "mean_dirs_mentioned_in_think": d["think_mentions_count_sum"] / n,
                "all_4_in_think_pct": 100 * d["think_mentions_all_4"] / n,
                "parsed_pct": 100 * d["parsed"] / n,
            }

        # Save examples
        mode_out["examples"] = {f"diff{nv}_size{sz}": exs for (nv, sz), exs in examples_to_save.items()}
        results["modes"][mode_name] = mode_out

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
