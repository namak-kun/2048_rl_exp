#!/usr/bin/env python3
"""Diagnostic eval: run multi-turn 2048 with very high max_moves to see what
max-tile the model can actually reach when not capped by turn budget.

Talks to an already-running vLLM OpenAI-compatible server.

Usage:
    python eval_unlimited.py --model PATH --n_games 16 --max_moves 1000 --out_json PATH
"""

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("VLLM_USE_V1", "1")

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))


async def main_async(args):
    from env_2048_text import load_environment
    from verifiers.types import ClientConfig

    env = load_environment(
        num_train_examples=args.n_games,
        num_eval_examples=args.n_games,
        max_moves=args.max_moves,
        max_invalid_moves=args.max_invalid_moves,
        grid_size=args.grid_size,
        target_tile=args.target_tile,
        output_format="json",
        context_mode="markov",
        max_tile_weight=0.8,
        score_weight=0.8,
        valid_moves_weight=0.2,
    )

    os.environ.setdefault("PRIME_API_KEY", "EMPTY")
    client = ClientConfig(
        client_type="openai_chat_completions",
        api_base_url=args.api_base,
        api_key_var="PRIME_API_KEY",
    )
    sampling_args = {
        "temperature": args.temperature,
        "top_p": 0.9,
        "max_tokens": args.max_tokens,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
    }

    print(f"Evaluating {args.n_games} games on model={args.model} (max_moves={args.max_moves})", flush=True)
    t0 = time.time()
    result = await env.evaluate(
        client=client,
        model=args.model,
        sampling_args=sampling_args,
        num_examples=args.n_games,
        rollouts_per_example=1,
        max_concurrent=args.max_concurrent,
    )
    elapsed = time.time() - t0
    outputs = result["outputs"]

    results = []
    tile_counter = Counter()
    stop_counter = Counter()
    for o in outputs:
        metrics = o.get("metrics") or {}
        max_tile = int(metrics.get("max_tile_value", 0))
        score = int(metrics.get("game_score", 0))
        turns = int(metrics.get("num_turns", 0))
        valid_ratio = metrics.get("valid_moves_ratio", 0.0)
        stop = o.get("stop_condition") or "unknown"
        tile_counter[max_tile] += 1
        stop_counter[stop] += 1
        results.append({
            "max_tile": max_tile,
            "score": score,
            "num_turns": turns,
            "valid_moves_ratio": valid_ratio,
            "reward": o.get("reward", 0),
            "is_truncated": o.get("is_truncated"),
            "stop": stop,
        })

    print(f"\nDone in {elapsed:.1f}s")
    print(f"Games completed: {len(results)}")
    print(f"\nMax-tile distribution: {dict(sorted(tile_counter.items()))}")
    print(f"Stop reasons: {dict(stop_counter)}")
    if results:
        import statistics as st
        tiles = [r["max_tile"] for r in results]
        scores = [r["score"] for r in results]
        turns = [r["num_turns"] for r in results]
        won = sum(1 for r in results if r["max_tile"] >= args.target_tile)
        print(f"\nMax tile   mean={st.mean(tiles):.0f}  median={int(st.median(tiles))}  range=[{min(tiles)}, {max(tiles)}]")
        print(f"Score      mean={st.mean(scores):.0f}  median={int(st.median(scores))}  range=[{min(scores)}, {max(scores)}]")
        print(f"Num turns  mean={st.mean(turns):.0f}  median={int(st.median(turns))}  range=[{min(turns)}, {max(turns)}]")
        print(f"Won (>={args.target_tile}): {won}/{len(results)} = {100*won/len(results):.0f}%")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({
                "model": args.model,
                "n_games": args.n_games,
                "max_moves": args.max_moves,
                "max_invalid_moves": args.max_invalid_moves,
                "grid_size": args.grid_size,
                "target_tile": args.target_tile,
                "temperature": args.temperature,
                "elapsed_sec": elapsed,
                "max_tile_distribution": {str(k): v for k, v in sorted(tile_counter.items())},
                "stop_reasons": dict(stop_counter),
                "results": results,
            }, f, indent=2)
        print(f"\nSaved {args.out_json}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--api_base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--n_games", type=int, default=16)
    ap.add_argument("--max_moves", type=int, default=1000)
    ap.add_argument("--max_invalid_moves", type=int, default=5)
    ap.add_argument("--grid_size", type=int, default=4)
    ap.add_argument("--target_tile", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--max_concurrent", type=int, default=16)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
