#!/usr/bin/env python3
"""
On-Policy Self-Distillation (OPSD) for 2048.

A single LoRA-adapted model serves as both student and teacher. The student
sees only the board state; the teacher sees the board plus a hint with the
oracle's recommended move. We minimize the per-token Jensen-Shannon divergence
(JSD, copied from siyan-zhao/OPSD) between teacher and student logit
distributions over the student's own on-policy rollout tokens.

The "fixed teacher" trick (also from OPSD): we use the SAME model for both
forward passes; teacher pass goes through `model.disable_adapter()` so it
sees the base weights only, while student pass uses base+LoRA. This avoids
loading two model copies.

Usage:
    python train_opsd_2048.py \\
        --model PATH/to/base \\
        --output_dir PATH/to/out \\
        --num_steps 100 \\
        --batch_size 4

The base model should already have the `lora_outputs_json/merged` weights or
similar (i.e., the seed model from our pipeline). LoRA is applied fresh on top.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from env_2048_text import Game2048, get_system_prompt, _extract_move_json
from expectimax_oracle import expectimax_oracle


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generalized_jsd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor | None = None,
    beta: float = 0.5,
    temperature: float = 1.0,
    token_clip: float | None = None,
):
    """Generalized Jensen-Shannon Divergence loss for distillation.

    Direct port of `OPSDTrainer.generalized_jsd_loss` from siyan-zhao/OPSD.

    beta=0  → forward KL: KL(student || teacher) (preferred per OPSD paper)
    beta=1  → reverse KL
    beta=0.5 → standard JSD
    """
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature

    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    if beta == 0:
        jsd = F.kl_div(
            student_log_probs, teacher_log_probs,
            reduction="none", log_target=True,
        )
    elif beta == 1:
        jsd = F.kl_div(
            teacher_log_probs, student_log_probs,
            reduction="none", log_target=True,
        )
    else:
        beta_t = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
        mixture_log_probs = torch.logsumexp(
            torch.stack([
                student_log_probs + torch.log1p(-beta_t),
                teacher_log_probs + torch.log(beta_t),
            ]),
            dim=0,
        )
        kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
        jsd = beta * kl_teacher + (1 - beta) * kl_student

    if token_clip is not None and token_clip > 0:
        jsd = jsd.clamp(max=token_clip)

    if labels is not None:
        mask = labels != -100
        jsd_masked = jsd.sum(dim=-1)[mask]
        n_tokens = mask.sum().clamp(min=1)
        return jsd_masked.sum() / n_tokens

    return jsd.sum() / max(1, jsd.size(0) * jsd.size(1))


def random_2048_board(grid_size: int, target_tile: int, fill_ratio: float, seed: int) -> Game2048:
    """Generate a random 2048 board with the given fill ratio.

    Uses Game2048's playthrough mechanism: fresh game, random valid moves until
    fill_ratio * grid_size^2 cells are non-empty. We use this rather than purely
    random tile placements so the boards are reachable from real play.
    """
    rng = random.Random(seed)
    game = Game2048(size=grid_size, target_tile=target_tile)
    target_filled = max(2, int(fill_ratio * grid_size * grid_size))
    max_iters = 500
    for _ in range(max_iters):
        non_empty = sum(1 for r in range(grid_size) for c in range(grid_size) if game.grid.get(r, c) != 0)
        if non_empty >= target_filled or game.over or game.won:
            break
        valid = [d for d in range(4) if _direction_changes_board(game, d)]
        if not valid:
            break
        d = rng.choice(valid)
        game.move(d)
    return game


def _direction_changes_board(game: Game2048, direction: int) -> bool:
    """Return True iff applying `direction` would move at least one tile."""
    clone = game.clone()
    return clone.move(direction)


DIRECTION_NAMES = {0: "up", 1: "right", 2: "down", 3: "left"}
DIRECTION_INDICES = {v: k for k, v in DIRECTION_NAMES.items()}


def oracle_move(game: Game2048) -> tuple[str, int]:
    """One-step lookahead oracle: return (best_direction_name, score_gain).

    For each of the 4 directions, simulate one move and measure the score
    increase + ties broken by tile-count reduction (more merges = fewer tiles).
    Picks the highest score_gain. Ties broken alphabetically.
    """
    best = None
    for d_idx in range(4):
        clone = game.clone()
        before_score = clone.score
        moved = clone.move(d_idx)
        if not moved:
            continue
        gain = clone.score - before_score
        # secondary: prefer fewer tiles (more empties)
        empties = sum(1 for r in range(clone.size) for c in range(clone.size) if clone.grid.get(r, c) == 0)
        key = (gain, empties)
        if best is None or key > best[0]:
            best = (key, d_idx)
    if best is None:
        return ("up", 0)  # board frozen; fallback
    _, d_idx = best
    return (DIRECTION_NAMES[d_idx], best[0][0])


def build_student_messages(game: Game2048) -> list[dict]:
    sys_p = get_system_prompt(grid_size=game.size, target_tile=game.target_tile, output_format="json")
    user_p = (
        f"{game.get_state_text()}\n\n"
        'What\'s your move? Respond with {"move": "direction"}.'
    )
    return [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}]


def build_teacher_messages(game: Game2048, oracle_dir: str) -> list[dict]:
    sys_p = get_system_prompt(grid_size=game.size, target_tile=game.target_tile, output_format="json")
    user_p = (
        f"{game.get_state_text()}\n\n"
        f'Consider this: a strong player would choose "{oracle_dir}" on this board. '
        f'Before deciding your move, examine the board carefully — what does that move achieve, '
        f'and is there a better alternative? Then make your decision.\n\n'
        'What\'s your move? Respond with {"move": "direction"}.'
    )
    return [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}]


def render_messages(tokenizer, messages: list[dict], enable_thinking: bool) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )


@torch.no_grad()
def generate_student_completion(
    model,
    tokenizer,
    student_prompt_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    device,
) -> tuple[torch.Tensor, str]:
    """Generate from the student's view (LoRA adapter ON)."""
    inputs = tokenizer(student_prompt_text, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    full_ids = out[0]
    completion_ids = full_ids[inputs["input_ids"].shape[1]:]
    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=False)
    return completion_ids, completion_text


def make_lora(model, args):
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def train_step(
    model,
    tokenizer,
    args,
    device,
    step: int,
    rng_seed: int,
):
    """One training step: build batch, generate, JSD loss, backward."""
    boards = []
    oracles = []
    for i in range(args.batch_size):
        fill = random.uniform(args.min_fill_ratio, args.max_fill_ratio)
        game = random_2048_board(args.grid_size, args.target_tile, fill, seed=rng_seed + i)
        if not any(_direction_changes_board(game, d) for d in range(4)):
            game = random_2048_board(args.grid_size, args.target_tile, 0.5, seed=rng_seed + i + 999)
        boards.append(game)
        oracle_dir, oracle_info = expectimax_oracle(game, depth=args.oracle_depth)
        if oracle_dir is None:
            oracle_dir = "up"
        oracles.append((oracle_dir, oracle_info))

    losses = []
    log_info = []
    for game, (oracle_dir, oracle_info) in zip(boards, oracles):
        student_msgs = build_student_messages(game)
        teacher_msgs = build_teacher_messages(game, oracle_dir)
        student_text = render_messages(tokenizer, student_msgs, args.student_thinking)
        teacher_text = render_messages(tokenizer, teacher_msgs, args.teacher_thinking)

        # Generate student completion (LoRA on, no_grad)
        model.eval()
        completion_ids, completion_text = generate_student_completion(
            model, tokenizer, student_text,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            device=device,
        )
        model.train()

        # Build full sequences for both views
        student_prompt_ids = tokenizer(student_text, return_tensors="pt").input_ids[0].to(device)
        teacher_prompt_ids = tokenizer(teacher_text, return_tensors="pt").input_ids[0].to(device)
        completion_ids = completion_ids.to(device)

        student_full = torch.cat([student_prompt_ids, completion_ids], dim=0).unsqueeze(0)
        teacher_full = torch.cat([teacher_prompt_ids, completion_ids], dim=0).unsqueeze(0)

        # Build labels: -100 for prompt, real ids for completion
        student_labels = torch.full_like(student_full, -100)
        student_labels[0, len(student_prompt_ids):] = student_full[0, len(student_prompt_ids):]

        # Forward STUDENT (LoRA on, with grad)
        student_outputs = model(input_ids=student_full)
        # Logits for predicting completion tokens: shifted by 1
        # student_logits[t] predicts token at position t+1
        # We want logits over the completion tokens only.
        # student_full has shape [1, P_s + C]; logits has shape [1, P_s + C, V].
        # The logits that predict completion tokens i (for i in [0, C-1]) are at
        # positions [P_s - 1, P_s + C - 2].
        Ps = len(student_prompt_ids)
        Cs = len(completion_ids)
        student_logits_for_completion = student_outputs.logits[:, Ps - 1 : Ps - 1 + Cs, :]

        # Forward TEACHER (LoRA off, no grad)
        with torch.no_grad():
            with model.disable_adapter():
                teacher_outputs = model(input_ids=teacher_full)
        Pt = len(teacher_prompt_ids)
        teacher_logits_for_completion = teacher_outputs.logits[:, Pt - 1 : Pt - 1 + Cs, :]

        # JSD loss
        # Make a labels tensor of shape [1, Cs] with valid mask everywhere
        completion_labels = completion_ids.unsqueeze(0)  # [1, Cs]
        loss = generalized_jsd_loss(
            student_logits=student_logits_for_completion,
            teacher_logits=teacher_logits_for_completion,
            labels=completion_labels,
            beta=args.beta,
            temperature=args.kl_temp,
            token_clip=args.jsd_token_clip,
        )
        losses.append(loss)

        # Parse what student picked
        # Strip <think>...</think> for parsing
        no_think = re.sub(r"<think>.*?</think>\s*", "", completion_text, count=1, flags=re.DOTALL)
        student_move = _extract_move_json(no_think)
        log_info.append({
            "oracle": oracle_dir,
            "student_move": student_move,
            "match": int(student_move == oracle_dir) if student_move else 0,
            "completion_chars": len(completion_text),
            "completion_tokens": Cs,
            "oracle_per_dir": oracle_info.get("per_direction"),
            "oracle_depth": oracle_info.get("depth"),
        })

    if not losses:
        return None, log_info

    total_loss = torch.stack(losses).mean()
    total_loss.backward()
    return total_loss.detach(), log_info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Base model path (will get LoRA on top)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_steps", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_grad_norm", type=float, default=0.1)

    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--lora_alpha", type=float, default=64)

    ap.add_argument("--grid_size", type=int, default=4)
    ap.add_argument("--target_tile", type=int, default=2048)
    ap.add_argument("--min_fill_ratio", type=float, default=0.3)
    ap.add_argument("--max_fill_ratio", type=float, default=0.85)
    ap.add_argument("--oracle_depth", type=int, default=3, help="expectimax search depth")

    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=20)

    ap.add_argument("--beta", type=float, default=0.0, help="JSD beta (0=forward KL, 1=reverse, 0.5=JSD)")
    ap.add_argument("--kl_temp", type=float, default=1.0)
    ap.add_argument("--jsd_token_clip", type=float, default=0.05)

    ap.add_argument("--student_thinking", action="store_true", default=True)
    ap.add_argument("--teacher_thinking", action="store_true", default=True)

    ap.add_argument("--save_every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bf16", action="store_true", default=True)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading tokenizer + base model from {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        attn_implementation="flash_attention_2",
    )
    base_model.to(device)
    print(f"Applying LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})", flush=True)
    model = make_lora(base_model, args)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training_log.jsonl"
    log_f = open(log_path, "w")

    print(f"Starting training: {args.num_steps} steps, batch={args.batch_size}", flush=True)
    t0 = time.time()
    for step in range(args.num_steps):
        rng_seed = args.seed * 1000 + step
        random.seed(rng_seed)

        loss, batch_log = train_step(model, tokenizer, args, device, step, rng_seed)

        if loss is None:
            print(f"[step {step}] empty batch", flush=True)
            continue

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], args.max_grad_norm
        )
        optimizer.step()
        optimizer.zero_grad()

        match_rate = sum(b["match"] for b in batch_log) / max(1, len(batch_log))
        avg_chars = sum(b["completion_chars"] for b in batch_log) / max(1, len(batch_log))
        avg_tokens = sum(b["completion_tokens"] for b in batch_log) / max(1, len(batch_log))
        elapsed = time.time() - t0

        msg = {
            "step": step,
            "loss": float(loss),
            "match_rate": match_rate,
            "avg_completion_chars": avg_chars,
            "avg_completion_tokens": avg_tokens,
            "elapsed_sec": elapsed,
            "batch": batch_log,
        }
        log_f.write(json.dumps(msg) + "\n")
        log_f.flush()
        print(
            f"[step {step:>4d}] loss={float(loss):.4f}  match={match_rate*100:>5.1f}%  "
            f"avg_tok={avg_tokens:.0f}  elapsed={elapsed:.0f}s",
            flush=True,
        )

        if args.save_every > 0 and (step + 1) % args.save_every == 0:
            save_dir = output_dir / f"step_{step+1}"
            save_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Saving adapter to {save_dir}", flush=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

    log_f.close()
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Done. Final adapter saved to {final_dir}", flush=True)


if __name__ == "__main__":
    main()
