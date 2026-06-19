#!/usr/bin/env python3
"""For one board + one hint, print the full thinking traces from K samples
to understand WHY the model is or isn't following the hint.
"""
import argparse
import asyncio
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from env_2048_text import Game2048, get_system_prompt, _extract_move_json


async def main_async(args):
    from openai import AsyncOpenAI
    import random

    client = AsyncOpenAI(base_url=args.api_base, api_key="EMPTY")
    rng = random.Random(args.seed)

    # Same board generation as v3
    boards = []
    for i in range(3):
        game = Game2048(size=4, target_tile=2048)
        target_filled = rng.randint(8, 13)
        for _ in range(50):
            non_empty = sum(1 for r in range(4) for c in range(4) if game.grid.get(r, c) != 0)
            if non_empty >= target_filled or game.over:
                break
            valid_dirs = [d for d in range(4) if game.clone().move(d)]
            if not valid_dirs:
                break
            game.move(rng.choice(valid_dirs))
        boards.append(game)

    game = boards[args.board_idx]
    print(f"Board {args.board_idx}:")
    print(game.get_state_text())
    print()

    sys_p = get_system_prompt(grid_size=4, target_tile=2048, output_format="json")

    hint_text = f'A 2048 expert who has analyzed this exact board recommends the move "{args.hint}". Think carefully about why this move is the best — what merges does it create, what board structure does it preserve, what alternatives are worse and why? Use this analysis to decide your move.'

    user_p = (
        f"{game.get_state_text()}\n\n"
        f"{hint_text}\n\n"
        'What\'s your move? Respond with {"move": "direction"}.'
    )

    print(f"Hint='{args.hint}'  Sampling {args.k} times...\n")

    for i in range(args.k):
        resp = await client.chat.completions.create(
            model=args.model,
            messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
            temperature=1.0, top_p=0.95, max_tokens=2048,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )
        text = resp.choices[0].message.content or ""
        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        think_text = think_match.group(1) if think_match else ""
        no_think = re.sub(r"<think>.*?</think>\s*", "", text, count=1, flags=re.DOTALL)
        move = _extract_move_json(no_think)
        match = "✓" if move == args.hint else "✗"
        print(f"=== Sample {i} | model picked: {move} {match} ===")
        # First and last 800 chars of think
        if len(think_text) > 1600:
            print(f"THINK START (1st 800 chars):\n{think_text[:800]}")
            print(f"...\n[truncated middle]\n...")
            print(f"THINK END (last 800 chars):\n{think_text[-800:]}")
        else:
            print(f"THINK:\n{think_text}")
        print(f"\nFINAL: {no_think.strip()[:200]}")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--api_base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--board_idx", type=int, default=0)
    ap.add_argument("--hint", choices=["up","right","down","left"], default="down")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
