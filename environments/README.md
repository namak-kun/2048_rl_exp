# 2048-RL environments

This directory holds two `verifiers`-style RL environments for training small
LLMs (Qwen3-0.6B in our experiments) to play the 2048 puzzle game.

- **`env_2048_text/`** — multi-turn environment. LLM plays a full game,
  one move per turn, until the game ends.
- **`env_2048_single_turn/`** — single-turn environment. LLM is shown a
  board and either picks the best move (`pick_one`) or lists all valid moves
  (`enumerate_all`).

Both environments are independently `prime env install`-able and integrate
with [`prime-rl`](https://github.com/PrimeIntellect-ai/prime-rl) for GRPO
training.

## Project docs

- [`README.md` (this file)](README.md) — top-level orientation
- [`env_2048_text/README.md`](env_2048_text/README.md) — multi-turn env docs
- [`env_2048_single_turn/README.md`](env_2048_single_turn/README.md) — single-turn env docs
- [`CONFIGS.md`](CONFIGS.md) — what every TOML config does, and which one worked best
- [`EXPERIMENTS.md`](EXPERIMENTS.md) — running project log of every experiment

## High-level finding

After a long arc of stages (single-turn SFT → single-turn RL on enumerate
task → multi-turn GRPO → multi-turn OPSD), the **best achievable Stage-3
recipe with Qwen3-0.6B + GRPO** is:

- Seed: `rl_enum_lora_prp/step_200` (single-turn RL'd model that kept thinking)
- Config: [`env_2048_text/configs/rl_lora_thinkprp.toml`](env_2048_text/configs/rl_lora_thinkprp.toml)
- Result: mean eval ~0.62 at step 50 across 4x4@2k, 4x4@4k, 5x5@4k, 5x5@8k targets
- ~5 min/step, ~4 hours to step 50

GRPO plateaus around mean reward 0.65 — the model reaches tile ~50-100 on
average but never wins (tile 2048). The plateau appears structural to GRPO on
long-horizon games: rollouts in a group end up very similar, group-relative
advantages collapse, no learning signal.

We tried **OPSD** (On-Policy Self-Distillation with expectimax oracle hints)
as a non-GRPO alternative but it made the model worse — see
[`EXPERIMENTS.md`](EXPERIMENTS.md) Exp 17.

Future direction: value-function-based methods (PPO/REINFORCE with critic) may
help with the credit assignment issue that GRPO struggles with on long rollouts.
