#!/usr/bin/env python3
"""
Generate SFT training data for Stage 1: teaching the model valid moves.

Each example is a (board_state, valid_move) pair where the prompt asks for a move
and the completion is either:
  - XML  : `<move>direction</move>`
  - JSON : `{"move": "direction"}`

Uses the same playthrough-based generation as env_2048_single_turn.generate_single_turn_dataset,
with bucket-balanced difficulty (75% 2-valid, 12.5% 3-valid, 12.5% 4-valid) and balanced
answer directions across up/down/left/right.

Output is a JSONL file with {"prompt": [...], "completion": [...]} per line.

Usage:
    python generate_sft_data.py --output sft_data_xml --num_examples 10000 --output_format xml
    python generate_sft_data.py --output sft_data_json --num_examples 10000 --output_format json \\
        --grid_sizes 4 5 6 --grid_weights 0.6 0.25 0.15
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env_2048_single_turn import (
    Game2048,
    Grid,
    count_valid_moves,
    generate_playthrough_states,
    get_system_prompt,
    get_user_prompt,
)

DIRECTION_NAMES = {0: "up", 1: "right", 2: "down", 3: "left"}


def format_completion(direction: int, output_format: str) -> str:
    name = DIRECTION_NAMES[direction]
    if output_format == "xml":
        return f"<move>{name}</move>"
    if output_format == "json":
        return json.dumps({"move": name})
    raise ValueError(f"Unknown output_format: {output_format!r}")


def get_valid_directions(game: Game2048) -> list[int]:
    valid = []
    for direction in range(4):
        test_game = Game2048(
            size=game.size,
            target_tile=game.target_tile,
            grid=Grid(size=game.size, cells=[row[:] for row in game.grid.cells]),
        )
        if test_game.move(direction):
            valid.append(direction)
    return valid


def generate_sft_examples(
    num_examples: int,
    size: int,
    target_tile: int,
    output_format: str,
    seed: int,
) -> list[dict]:
    """Generate balanced-difficulty SFT examples for a given grid size.

    Buckets by (valid_move_count, chosen_direction) to balance both axes.
    """
    random.seed(seed)
    system_prompt = get_system_prompt(grid_size=size, target_tile=target_tile, output_format=output_format)

    target_counts = {
        2: int(num_examples * 0.75),
        3: int(num_examples * 0.125),
        4: int(num_examples * 0.125),
    }
    target_counts[2] += num_examples - sum(target_counts.values())

    buckets = {nv: {d: [] for d in range(4)} for nv in (2, 3, 4)}
    target_per_dir = {nv: (target_counts[nv] + 3) // 4 for nv in (2, 3, 4)}

    def all_full() -> bool:
        return all(
            sum(len(b) for b in buckets[nv].values()) >= target_counts[nv]
            for nv in (2, 3, 4)
        )

    iters = 0
    max_iters = max(2000, num_examples * 30)

    while not all_full() and iters < max_iters:
        iters += 1
        fill_ratio = random.uniform(0.4, 0.95)
        states = generate_playthrough_states(
            size=size,
            target_tile=target_tile,
            fill_ratio=fill_ratio,
            snapshot_interval=2,
        )
        for game in states:
            nv = count_valid_moves(game)
            if nv not in (2, 3, 4):
                continue
            if sum(len(b) for b in buckets[nv].values()) >= target_counts[nv]:
                continue
            valid_dirs = get_valid_directions(game)
            under = [d for d in valid_dirs if len(buckets[nv][d]) < target_per_dir[nv]]
            chosen = random.choice(under) if under else random.choice(valid_dirs)
            buckets[nv][chosen].append((game, chosen))

    examples = []
    for nv, dir_buckets in buckets.items():
        flat = []
        for items in dir_buckets.values():
            flat.extend(items)
        random.shuffle(flat)
        flat = flat[: target_counts[nv]]

        for game, direction in flat:
            user_content = get_user_prompt(game, output_format=output_format)
            examples.append({
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "completion": [
                    {"role": "assistant", "content": format_completion(direction, output_format)},
                ],
                # Tell Qwen3's chat template not to emit a <think></think> block
                # in the generation prompt. Without this, TRL's completion_mask
                # would include the empty think block as supervised tokens, which
                # trains the model to suppress reasoning entirely.
                "chat_template_kwargs": {"enable_thinking": False},
            })

    random.shuffle(examples)
    return examples


def main():
    parser = argparse.ArgumentParser(description="Generate SFT data for 2048 valid moves")
    parser.add_argument("--output", type=str, required=True, help="Output directory (will contain train.jsonl)")
    parser.add_argument("--num_examples", type=int, default=10000)
    parser.add_argument("--grid_sizes", type=int, nargs="+", default=[4])
    parser.add_argument("--grid_weights", type=float, nargs="+", default=None,
                        help="Weights for each grid size (default: equal)")
    parser.add_argument("--target_tile", type=int, default=2048)
    parser.add_argument("--output_format", choices=["xml", "json"], default="xml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.grid_weights is None:
        args.grid_weights = [1.0] * len(args.grid_sizes)
    if len(args.grid_weights) != len(args.grid_sizes):
        raise ValueError("--grid_weights must match --grid_sizes length")

    total_w = sum(args.grid_weights)
    weights = [w / total_w for w in args.grid_weights]

    all_examples = []
    for size, w in zip(args.grid_sizes, weights):
        n = int(args.num_examples * w)
        print(f"Generating {n} examples for {size}x{size} grid (format={args.output_format})...")
        ex = generate_sft_examples(
            num_examples=n,
            size=size,
            target_tile=args.target_tile,
            output_format=args.output_format,
            seed=args.seed + size,
        )
        all_examples.extend(ex)
        print(f"  generated {len(ex)} examples")

    random.seed(args.seed)
    random.shuffle(all_examples)

    dir_counts = {"up": 0, "right": 0, "down": 0, "left": 0}
    for ex in all_examples:
        content = ex["completion"][0]["content"]
        for d in dir_counts:
            if d in content:
                dir_counts[d] += 1
                break

    print(f"\nTotal examples: {len(all_examples)}")
    print("Direction distribution:")
    for d, c in sorted(dir_counts.items()):
        pct = 100 * c / len(all_examples) if all_examples else 0
        print(f"  {d:>6}: {c:>5} ({pct:.1f}%)")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train.jsonl"
    with open(out_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"\nSaved {len(all_examples)} examples to {out_path}")


if __name__ == "__main__":
    main()
