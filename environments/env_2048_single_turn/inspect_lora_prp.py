#!/usr/bin/env python3
"""Inspect the lora+PR_penalty step_200 model: prediction sizes vs true sizes."""

import json
import os
import sys
from pathlib import Path
from collections import Counter

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


def make_boards(n_per_size, sizes, seed=42):
    import random
    random.seed(seed)
    boards = []
    for size in sizes:
        # Get a uniform-difficulty sample: equal numbers of 2/3/4-valid boards
        target = {2: n_per_size // 3 + n_per_size % 3, 3: n_per_size // 3, 4: n_per_size // 3}
        bucket = {k: [] for k in target}
        iters = 0
        while sum(len(b) for b in bucket.values()) < n_per_size and iters < n_per_size * 100:
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
    if len(sys.argv) < 2:
        print("usage: python inspect_lora_prp.py PATH_TO_CHECKPOINT [n_per_size]")
        sys.exit(1)
    model_path = sys.argv[1]
    n_per_size = int(sys.argv[2]) if len(sys.argv) > 2 else 90

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"Loading {model_path}", flush=True)
    llm = LLM(model=model_path, dtype="bfloat16", gpu_memory_utilization=0.5, max_model_len=8192)
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    boards = make_boards(n_per_size, [4, 5, 6])
    print(f"Boards: {len(boards)} total ({sum(1 for _,nv,_ in boards if nv==2)} 2-valid, "
          f"{sum(1 for _,nv,_ in boards if nv==3)} 3-valid, "
          f"{sum(1 for _,nv,_ in boards if nv==4)} 4-valid)")

    prompts = []
    truths = []
    for game, nv, size in boards:
        sys_p = get_system_prompt(size, 2048, output_format="json", task_type="enumerate_all")
        user_p = get_user_prompt(game, output_format="json", task_type="enumerate_all")
        txt = tok.apply_chat_template(
            [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        prompts.append(txt)
        truths.append(frozenset(get_valid_direction_names(game)))

    sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=4096)
    outs = llm.generate(prompts, sp, use_tqdm=True)

    # Analysis: per true-size, what does the model predict?
    confusion = {}  # confusion[true_size][pred_size] = count
    by_true_size_correct = Counter()
    by_true_size_total = Counter()
    # Also, when true=3, is the predicted 2-set a subset of the true 3-set?
    subset_check = {3: Counter(), 4: Counter()}  # bigger -> bigger means we count subsets
    examples_by_class = {}  # (true_size, pred_size) -> first example

    for (game, nv, size), out, true_set in zip(boards, outs, truths):
        text = out.outputs[0].text
        # strip think
        import re
        after = re.sub(r"<think>.*?</think>\s*", "", text, count=1, flags=re.DOTALL)
        pred = _extract_valid_moves_set(after)
        pred_size = len(pred) if pred is not None else -1

        confusion.setdefault(nv, Counter())[pred_size] += 1
        by_true_size_total[nv] += 1
        if pred == true_set:
            by_true_size_correct[nv] += 1

        # Subset check
        if pred is not None and nv in (3, 4):
            if pred <= true_set:
                subset_check[nv][len(pred)] += 1

        key = (nv, pred_size)
        if key not in examples_by_class:
            examples_by_class[key] = {
                "true": sorted(true_set),
                "pred": sorted(pred) if pred else None,
                "board": game.get_state_text(),
            }

    print("\n=== Confusion matrix: true set size vs predicted set size ===")
    print(f"{'true\\pred':<10} {'unparsed':>8} {'size=0':>6} {'size=1':>6} {'size=2':>6} {'size=3':>6} {'size=4':>6} {'total':>6}")
    for true_size in sorted(confusion.keys()):
        row = confusion[true_size]
        total = sum(row.values())
        unparsed = row.get(-1, 0)
        cells = [f"{row.get(s, 0):>6}" for s in [0, 1, 2, 3, 4]]
        print(f"{true_size:<10} {unparsed:>8} {'  '.join(cells)} {total:>6}")

    print("\n=== EM by true-size ===")
    for ts in sorted(by_true_size_total.keys()):
        n = by_true_size_total[ts]
        c = by_true_size_correct[ts]
        print(f"  true=|{ts}|: {c}/{n} ({100*c/n:.0f}%) exact")

    print("\n=== Subset check (does pred ⊆ true?) ===")
    print("  When true is 3-valid:")
    for sz, cnt in sorted(subset_check[3].items()):
        total = confusion[3].get(sz, 0)
        print(f"    pred |{sz}| was a subset of true: {cnt}/{total}")
    print("  When true is 4-valid:")
    for sz, cnt in sorted(subset_check[4].items()):
        total = confusion[4].get(sz, 0)
        print(f"    pred |{sz}| was a subset of true: {cnt}/{total}")

    print("\n=== Example (true, pred) pairs ===")
    for key in sorted(examples_by_class.keys()):
        ex = examples_by_class[key]
        print(f"\n  true_size={key[0]} pred_size={key[1]}: true={ex['true']} pred={ex['pred']}")


if __name__ == "__main__":
    main()
