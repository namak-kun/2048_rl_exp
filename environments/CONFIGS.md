# Config Guide

This document explains every TOML config in this repo: what it's for, which
seed model it uses, what reward shape, and how it performed.

Configs live under each environment's `configs/` directory. They are consumed
by `prime-rl`'s CLI: `uv run rl @ path/to/config.toml` (or `uv run sft @ ...`
for SFT configs).

Naming convention: `<task>_<seed>_<reward>[_<variant>].toml`.

Read [EXPERIMENTS.md](EXPERIMENTS.md) for the narrative arc and detailed
experimental findings.

---

## `env_2048_single_turn/configs/`

The single-turn env (one board → one move or one set-of-valid-moves).
Two task types:
- **pick_one** — output `{"move": "direction"}` for a single best move
- **enumerate_all** — output `{"valid_moves": [...]}` for the set of all valid moves

### SFT configs

| Config | Purpose | Seed | Result |
|---|---|---|---|
| `sft.toml` | Initial SFT recipe (legacy) | base Qwen3-0.6B | Superseded |
| `sft_lora.toml` | LoRA-SFT on pick_one JSON data | base Qwen3-0.6B | ✅ Best SFT model (100% valid in no-think) |
| `sft_json_full.toml` | Full-FT SFT on pick_one JSON data | base Qwen3-0.6B | Lost thinking ability |
| `sft_xml_full.toml` | Full-FT SFT on pick_one XML data | base Qwen3-0.6B | Lower quality than JSON variant |

**Winner**: `sft_lora.toml` — produces `lora_outputs_json/merged/`, which became
the seed model for later stages.

### Stage-2 enumerate-task RL configs

These RL the SFT'd model on the `enumerate_all` task (a stepping stone to
multi-turn). Format: `rl_enum_<seed>_<reward>[_<variant>].toml`.

Seed: `base` = Qwen3-0.6B, `lora` = our SFT'd merged checkpoint.
Reward: `em` = exact-match, `prp` = precision-recall penalty, `emprp` = combo.

| Config | Seed | Reward | Variant | Notes |
|---|---|---|---|---|
| `rl_enum_base_em.toml` | base | EM | - | Failed (sparse reward) |
| `rl_enum_base_prp.toml` | base | PRP | - | Collapsed to predict-all |
| `rl_enum_base_emprp.toml` | base | EM+PRP | - | Collapsed |
| `rl_enum_lora_em.toml` | LoRA-SFT | EM | - | Yappy; eventual truncation |
| `rl_enum_lora_prp.toml` | LoRA-SFT | PRP | - | **Best**: 76% EM, kept thinking. Used as later seed. |
| `rl_enum_lora_emprp.toml` | LoRA-SFT | EM+PRP | la=0.05 | Collapsed to 18-char `{"valid_moves":[]}` |
| `rl_enum_lora_emprp_la00.toml` | LoRA-SFT | EM+PRP | length_alpha=0 | weaker collapse |
| `rl_enum_lora_emprp_la01.toml` | LoRA-SFT | EM+PRP | length_alpha=0.1 | strong collapse |
| `rl_enum_lora_emprp_la02.toml` | LoRA-SFT | EM+PRP | length_alpha=0.2 | crashed at step 21 |
| `rl_enum_lora_emprp_uniform.toml` | LoRA-SFT | EM+PRP | uniform difficulty data 33/33/33 | Escaped collapse but lost precision |
| `rl_enum_lora_emprp_adaptive.toml` | LoRA-SFT | EM+PRP | online_difficulty_filtering | Partial escape, think-mode failed |

**Winner**: `rl_enum_lora_prp.toml` (Exp 13). 76% overall EM, 100% on the
dominant 2-valid case, kept thinking ON, used as the seed for stage-3 (multi-turn).
The "collapse" we initially flagged turned out to be a 100%-precision pattern
(model never picks invalid moves), it just predicts size-2 always.

### Stage-3 multi-turn from single-turn seed (legacy)

| Config | Purpose | Notes |
|---|---|---|
| `rl_stage2.toml` | Early multi-turn experiment | Superseded by configs in `env_2048_text/` |

### Reusable

| Config | Purpose |
|---|---|
| `rl.toml` | Generic RL template |

---

## `env_2048_text/configs/`

Multi-turn env (LLM plays full 2048 games).

### Curriculum-style stage configs (legacy)

| Config | Reward weights | Purpose |
|---|---|---|
| `valid_moves.toml` | 0.0 max_tile, 1.0 valid_moves | Stage 1: pure validity |
| `max_tile.toml` | 0.8 max_tile, 0.2 valid_moves | Stage 2: tile-target |
| `efficiency.toml` | 0.0/0.0, 1.0 efficiency | Stage 3: efficiency |
| `rl.toml` | 0.5/0.5 default | Default template |

These were the original three-stage curriculum, now superseded by direct full
multi-turn training (below).

### Stage-3 actual training configs

These were the configs we actually used for the main multi-turn RL runs.
All use JSON output format, markov context mode, max_moves=100, max_invalid_moves=5.

| Config | Seed | Reward | Result |
|---|---|---|---|
| `rl_base_json.toml` | base Qwen3-0.6B | 0.8 max_tile + 0.2 valid | **Best mean eval 0.626 at step 50.** Decay after step 200. |
| `rl_lora_json.toml` | `lora_outputs_json/merged` (SFT) | 0.8 max_tile + 0.2 valid | Peaked at step 50 mean 0.595; step 75 hit 0.623. |
| `rl_lora_thinkprp.toml` | `rl_enum_lora_prp/step_200` (stage-2 RL'd) | 0.8 max_tile + 0.2 valid | Best LoRA result so far: peak 0.621 mean at step 50 |
| `rl_lora_thinkprp_score.toml` | resume from `rl_lora_thinkprp/step_100` | 0.8 max_tile + 0.8 score + 0.2 valid | Score reward added; training reward jumped 0.66→0.70 but eval plateaued |

**Best stage-3 GRPO result overall**: `rl_lora_thinkprp` at step 50 — mean eval
0.621, completion length 661 chars, beat base full-FT (0.626 mean) with 1/3 the
wall-time. But all GRPO runs ultimately plateaued around mean reward 0.65,
which corresponds to avg max-tile ~50-100. None won (reached tile 2048).

### Notes on Stage 3 GRPO failure mode
All multi-turn GRPO runs hit a ceiling where:
- Best rollouts in a group cap around reward 0.75
- Variance across rollouts is narrow (std ~0.05)
- GRPO has no learning signal — every rollout in a group is similar
- Hypothesis: long rollouts are inherently sample-inefficient with GRPO;
  recovering from bad moves is hard, model needs to make many good moves in a row

See `EXPERIMENTS.md` for the OPSD attempt (Exp 17 — also negative).

---

## Trying it yourself

To reproduce the best Stage-3 GRPO recipe:
```bash
cd prime-rl  # assumes you've cloned prime-rl and the env packages are installed
uv run rl @ ../environments/env_2048_text/configs/rl_lora_thinkprp.toml
```

This needs the seed model `rl_enum_lora_prp/step_200` to exist — train that
first via `rl_enum_lora_prp.toml`, or use the chain:
```bash
# Stage 1: SFT
uv run sft @ ../environments/env_2048_single_turn/configs/sft_lora.toml
# Stage 2: single-turn RL on the enumerate task
uv run rl @ ../environments/env_2048_single_turn/configs/rl_enum_lora_prp.toml
# Stage 3: multi-turn RL
uv run rl @ ../environments/env_2048_text/configs/rl_lora_thinkprp.toml
```
