#!/usr/bin/env python3
"""Debug helper: run a few games against a vLLM endpoint and dump the FULL
trajectory so we can see what invalid moves actually look like.

Usage:
    python debug_invalids.py --model PATH --n 3
"""

import argparse
import asyncio
import json
import os
import sys
import re
from pathlib import Path

os.environ.setdefault("VLLM_USE_V1", "1")
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))


async def main_async(args):
    from env_2048_text import (
        load_environment,
        _extract_move_json,
        Game2048,
    )
    from openai import AsyncOpenAI

    env = load_environment(
        num_train_examples=args.n,
        num_eval_examples=args.n,
        max_moves=args.max_moves,
        max_invalid_moves=args.max_invalid_moves,
        grid_size=4,
        target_tile=2048,
        output_format="json",
        context_mode="markov",
    )
    client = AsyncOpenAI(base_url=args.api_base, api_key="EMPTY")

    inputs = env.get_eval_dataset(n=args.n)
    all_games = []
    for idx in range(args.n):
        info = inputs[idx]["info"]
        game = Game2048(size=4, target_tile=2048)
        from copy import deepcopy
        game.grid.cells = deepcopy(info["initial_grid"])
        game._update_max_tile()

        sys_p = inputs[idx]["prompt"][0]["content"]
        msg_history = [{"role": "system", "content": sys_p}]
        first_user = inputs[idx]["prompt"][1]["content"]
        msg_history.append({"role": "user", "content": first_user})

        events = []
        consecutive_invalid = 0
        for turn in range(args.max_moves):
            if game.over or game.won or consecutive_invalid >= args.max_invalid_moves:
                break
            # Markov: just send system + current state, not full history
            sys_msg = msg_history[0]
            user_msg = {
                "role": "user",
                "content": (
                    f"{game.get_state_text()}\n\n"
                    'What\'s your move? Respond with {"move": "direction"}.'
                ),
            }
            resp = await client.chat.completions.create(
                model=args.model,
                messages=[sys_msg, user_msg],
                temperature=0.7, top_p=0.9, max_tokens=2048,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            )
            content = resp.choices[0].message.content or ""
            move_str = _extract_move_json(content)
            kind = None
            if move_str is None:
                kind = "no_parse"
                consecutive_invalid += 1
            else:
                direction = game.parse_move(move_str)
                if direction is None:
                    kind = "bad_direction"
                    consecutive_invalid += 1
                else:
                    pre_grid = [row[:] for row in game.grid.cells]
                    moved = game.move(direction)
                    if not moved:
                        kind = "no_change"
                        consecutive_invalid += 1
                    else:
                        kind = "valid"
                        consecutive_invalid = 0
            events.append({
                "turn": turn,
                "kind": kind,
                "move_str": move_str,
                "consecutive_invalid": consecutive_invalid,
                "max_tile": game.max_tile,
                "score": game.score,
                "completion_tail": content[-200:],
            })
        all_games.append({
            "game_idx": idx,
            "final_max_tile": game.max_tile,
            "final_score": game.score,
            "total_turns": len(events),
            "ended_by": "won" if game.won else "over" if game.over else "too_many_invalid" if consecutive_invalid >= args.max_invalid_moves else "max_moves",
            "events": events,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_games, f, indent=2)

    # Summary
    from collections import Counter
    print(f"\n=== Summary ({args.n} games) ===")
    for g in all_games:
        kind_counts = Counter(e["kind"] for e in g["events"])
        print(f"Game {g['game_idx']}: max_tile={g['final_max_tile']:>4} turns={g['total_turns']:>4} end={g['ended_by']:<20} kinds={dict(kind_counts)}")

    print(f"\nFull trajectories saved to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--api_base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max_moves", type=int, default=500)
    ap.add_argument("--max_invalid_moves", type=int, default=10)
    ap.add_argument("--out", default="debug_invalids.json")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
