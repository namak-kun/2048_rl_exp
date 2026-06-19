#!/usr/bin/env python3
"""Test reasoning-style hint phrasings: invite the model to reason about the hint."""
import argparse
import asyncio
import json
import re
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from env_2048_text import Game2048, get_system_prompt, _extract_move_json


HINT_PHRASINGS = {
    "expert_explain":
        'A 2048 expert who has analyzed this exact board recommends the move "{H}". Think carefully about why this move is the best — what merges does it create, what board structure does it preserve, what alternatives are worse and why? Use this analysis to decide your move.',
    "expert_persuade":
        'A 2048 expert says the best move here is "{H}". Look at the board and think through why they might say this — examine each direction, see which merges happen, and verify the expert\'s pick.',
    "consider_then_decide":
        'Consider this: a strong player would choose "{H}" on this board. Before deciding your move, examine the board carefully — what does that move achieve, and is there a better alternative? Then make your decision.',
}


async def main_async(args):
    from openai import AsyncOpenAI
    import random

    client = AsyncOpenAI(base_url=args.api_base, api_key="EMPTY")
    rng = random.Random(args.seed)

    boards = []
    for i in range(args.n_boards):
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

    sys_p_base = get_system_prompt(grid_size=4, target_tile=2048, output_format="json")

    async def query(game, hint_text, k_samples):
        if hint_text is None:
            user_p = (
                f"{game.get_state_text()}\n\n"
                'What\'s your move? Respond with {"move": "direction"}.'
            )
        else:
            user_p = (
                f"{game.get_state_text()}\n\n"
                f"{hint_text}\n\n"
                'What\'s your move? Respond with {"move": "direction"}.'
            )
        results = []
        # Track whether thinking mentioned the expert's move
        for _ in range(k_samples):
            resp = await client.chat.completions.create(
                model=args.model,
                messages=[{"role": "system", "content": sys_p_base}, {"role": "user", "content": user_p}],
                temperature=args.temperature, top_p=0.95, max_tokens=args.max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            )
            text = resp.choices[0].message.content or ""
            think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
            think_text = think_match.group(1) if think_match else ""
            no_think = re.sub(r"<think>.*?</think>\s*", "", text, count=1, flags=re.DOTALL)
            move = _extract_move_json(no_think)
            results.append({"move": move, "think_chars": len(think_text), "completion_chars": len(text)})
        return results

    DIRS = ["up", "right", "down", "left"]
    all_results = []

    for b_idx, game in enumerate(boards):
        valid = []
        for d_name in DIRS:
            d_idx = {"up":0,"right":1,"down":2,"left":3}[d_name]
            if game.clone().move(d_idx):
                valid.append(d_name)
        print(f"\n{'='*70}")
        print(f"Board {b_idx} | Valid moves: {valid}")
        print(f"{'='*70}")
        print(game.get_state_text())

        per_phrasing = {}
        # No hint baseline
        rs = await query(game, None, args.k_samples)
        moves = [r["move"] for r in rs]
        cnt = dict(Counter(moves))
        avg_think = sum(r["think_chars"] for r in rs) / max(1, len(rs))
        per_phrasing["no_hint"] = {"moves": cnt, "avg_think_chars": avg_think}
        print(f"\n  no_hint: {cnt}  avg_think={avg_think:.0f} chars")

        for phrasing_name, template in HINT_PHRASINGS.items():
            print(f"\n  --- {phrasing_name} ---")
            phrasing_data = {}
            for hint in DIRS:
                hint_text = template.format(H=hint)
                rs = await query(game, hint_text, args.k_samples)
                moves = [r["move"] for r in rs]
                cnt = dict(Counter(moves))
                avg_think = sum(r["think_chars"] for r in rs) / max(1, len(rs))
                hint_compliance = cnt.get(hint, 0) / args.k_samples
                marker = " ✓" if hint_compliance >= 0.75 else ("" if hint_compliance >= 0.5 else " ✗")
                print(f"    hint={hint:<5} compliance={hint_compliance*100:>3.0f}%{marker} | think_avg={avg_think:>5.0f} | {cnt}")
                phrasing_data[hint] = {"moves": cnt, "avg_think_chars": avg_think}
            per_phrasing[phrasing_name] = phrasing_data

        all_results.append({
            "board_idx": b_idx,
            "valid_moves": valid,
            "per_phrasing": per_phrasing,
        })

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--api_base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--n_boards", type=int, default=3)
    ap.add_argument("--k_samples", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
