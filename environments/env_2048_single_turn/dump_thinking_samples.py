#!/usr/bin/env python3
"""Dump full think+answer responses from LoRA-XML and LoRA-JSON to markdown for inspection."""

import os
import random
import sys
from pathlib import Path

os.environ.setdefault("VLLM_USE_V1", "1")

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from env_2048_single_turn import (  # noqa: E402
    Game2048, Grid,
    count_valid_moves,
    generate_playthrough_states,
    get_system_prompt, get_user_prompt,
    _extract_move_xml, _extract_move_json,
)


def get_boards(n: int, sizes: list[int], seed: int) -> list[tuple[Game2048, int, int]]:
    """Return n boards: 60% 2-valid, 25% 3-valid, 15% 4-valid, balanced across sizes."""
    random.seed(seed)
    out = []
    iters = 0
    per_size = n // len(sizes)
    for size in sizes:
        bucket = {2: [], 3: [], 4: []}
        target = {2: int(per_size * 0.6), 3: int(per_size * 0.25), 4: int(per_size * 0.15)}
        target[2] += per_size - sum(target.values())
        while sum(len(b) for b in bucket.values()) < per_size and iters < n * 100:
            iters += 1
            for g in generate_playthrough_states(
                size=size, fill_ratio=random.uniform(0.5, 0.95), snapshot_interval=2
            ):
                nv = count_valid_moves(g)
                if nv in bucket and len(bucket[nv]) < target[nv]:
                    bucket[nv].append(g)
        for nv, games in bucket.items():
            for g in games:
                out.append((g, nv, size))
    random.shuffle(out)
    return out[:n]


def run(model_path: str, fmt: str, boards: list, out_file: Path, max_new_tokens: int):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"Loading {model_path}")
    llm = LLM(model=model_path, dtype="bfloat16",
              gpu_memory_utilization=0.34, max_model_len=6144, enforce_eager=False)
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    prompts = []
    for game, _nv, size in boards:
        sys_p = get_system_prompt(size, 2048, fmt)
        user_p = get_user_prompt(game, fmt)
        txt = tok.apply_chat_template(
            [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        prompts.append(txt)

    sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=max_new_tokens)
    outs = llm.generate(prompts, sp, use_tqdm=True)
    extract = _extract_move_xml if fmt == "xml" else _extract_move_json

    import re
    lines = []
    lines.append(f"# {model_path} | format={fmt} | enable_thinking=True | max_new_tokens={max_new_tokens}")
    lines.append("")
    lines.append(f"Inspection sample of {len(boards)} boards (mixed difficulty + grid sizes).")
    lines.append("")
    for i, ((game, nv, size), o) in enumerate(zip(boards, outs)):
        text = o.outputs[0].text
        m_think = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        think_text = m_think.group(1).strip() if m_think else ""
        had_close_think = m_think is not None
        # Strip <think>...</think> if it appeared, otherwise everything is reasoning
        after_think = re.sub(r"<think>.*?</think>\s*", "", text, count=1, flags=re.DOTALL)
        move = (extract(after_think) or "").strip()

        # Validate move
        move_valid = False
        move_moved = False
        if move:
            d = Game2048.parse_move(move)
            if d is not None:
                move_valid = True
                test_game = Game2048(size=game.size, target_tile=game.target_tile,
                                     grid=Grid(size=game.size, cells=[r[:] for r in game.grid.cells]))
                move_moved = test_game.move(d)

        lines.append(f"## Example #{i+1}")
        lines.append(f"- grid_size: {size}x{size}")
        lines.append(f"- valid_dirs: {nv}")
        lines.append(f"- finish_reason: {o.outputs[0].finish_reason}")
        lines.append(f"- gen_length: {len(text)} chars, {len(o.outputs[0].token_ids)} tokens")
        lines.append(f"- has_think_close: {had_close_think}, think_chars: {len(think_text)}")
        lines.append(f"- parsed_move: {move!r}, valid_format: {move_valid}, moved_tiles: {move_moved}")
        lines.append("")
        lines.append("### Board")
        lines.append("```")
        lines.append(game.get_state_text())
        lines.append("```")
        if think_text:
            lines.append("### Thinking")
            lines.append("```")
            lines.append(think_text)
            lines.append("```")
        else:
            lines.append("### Thinking")
            lines.append("(no closed <think>...</think> block found — model may have run out of tokens mid-thinking)")
        lines.append("### Answer (post-think)")
        lines.append("```")
        lines.append(after_think.strip()[:1500])
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines))
    print(f"Wrote {out_file}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--format", choices=["xml", "json"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument("--sizes", type=int, nargs="+", default=[4, 5, 6])
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    boards = get_boards(args.n, args.sizes, args.seed)
    run(args.model, args.format, boards, Path(args.out), args.max_new_tokens)
