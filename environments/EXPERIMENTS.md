# 2048-RL Project Experiments Log

> Project-wide running document of every experiment, decision, and finding.
> Newest entries at the top. Each experiment is dated and tagged with status.
> Per-env detail logs may exist under each environment folder.

---

## Project Overview

We are building an RL pipeline that teaches Qwen3-0.6B to play 2048, using
[verifiers](https://github.com/PrimeIntellect-ai/verifiers) for environment
abstractions and [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) for
the RL trainer (GRPO).

Two environments live in `environments/`:
- `env_2048_text/` — **multi-turn** environment (LLM plays many moves)
- `env_2048_single_turn/` — **single-turn** environment (LLM sees a board, makes one move)

Original 4-stage curriculum:
1. SFT on valid moves (single-turn)
2. RL on valid moves with thinking (single-turn)
3. RL on multi-turn game with 3 rewards (max_tile, valid_moves, num_turns)
4. RL with more weight on num_turns

Current status: Stages 1 and 2 hit catastrophic-forgetting and policy-collapse
issues that we eventually solved with LoRA + a Qwen3 chat-template-masking
trick (see Exp 11). Stage 3 (multi-turn RL) is the next milestone.

---

## TL;DR (current state, June 19 2026)

- **Multi-turn env (`env_2048_text`)** supports `output_format="json"`, score
  reward, and oracle hints. Default reward = 0.5 max_tile + 0.5 valid_moves
  (configurable).
- **Best Stage 3 GRPO recipe**: `rl_lora_thinkprp.toml` from
  `rl_enum_lora_prp/step_200` seed. Mean eval ~0.62 at step 50, plateaus
  around mean reward 0.65.
- **OPSD attempt** (Exp 17) gave a negative result: model became more verbose
  with worse gameplay. Plain GRPO from a good seed remains the best baseline.
- **Best uncontested baseline**: base Qwen3-0.6B with no training reaches tile
  128 in 25% of games (mean tile 60) when given enough token budget. RL'd
  models match this at best, none reliably beat it.
- **Thinking preservation** is achieved by using LoRA (rank 16+) +
  `chat_template_kwargs={"enable_thinking": False}` per example. Full FT
  destroys thinking even with the masking trick.

## Open questions / future directions
- [ ] Value-function-based methods (PPO/REINFORCE-with-critic) for the
      long-rollout credit assignment problem GRPO struggles with.
- [ ] OPSD retry with max_new_tokens=8000+ during training (so student
      rollouts complete naturally and the model learns when to stop thinking).
- [ ] Process reward modeling — score intermediate states (e.g. expectimax
      score of post-move board) instead of just terminal reward.
- [ ] MCTS-style inference-time search using the trained policy as the
      action prior.
- [ ] Move to a larger base model (3B+) where the reasoning capacity isn't
      a fundamental bottleneck.

---

## Experiments (newest first)

### Exp 17 — On-Policy Self-Distillation (OPSD) with expectimax oracle
- **When**: 2026-06-16 → 2026-06-18
- **Env**: `env_2048_text` single-turn-style boards (random fill 0.3-0.85)
- **Status**: ❌ Negative result. OPSD made the model strictly worse on play
  quality. Useful artifacts: `train_opsd_2048.py`, `expectimax_oracle.py`.

**Motivation**: After Stage 3 GRPO plateaued (Exp 16), we hypothesized that
GRPO is sample-inefficient on long rollouts because group-relative advantages
collapse when all rollouts in a group reach similar end-states. OPSD
(Zhao et al., arxiv 2601.18734) is a non-GRPO alternative that distills a
teacher policy (model + privileged oracle hint) into a student policy
(model with no hint), via per-token JSD on the student's on-policy rollouts.

**Implementation**:
- `expectimax_oracle.py`: depth-N expectimax over the 2048 game tree with raw
  `game.score` as the leaf heuristic. depth=3 (560ms per query) used in training.
- Hint phrasing testing showed: base Qwen3-0.6B follows the
  `consider_then_decide` phrasing ~73% of the time, balanced across all 4
  directions. `directive` had higher compliance (~50% with strong "right" bias
  artifact). `rl_enum_lora_prp` (stage-2 seed) only got ~33% — its reasoning
  collapsed under prior RL.
- `train_opsd_2048.py`: standalone trainer using OPSD's JSD formula. Uses
  PEFT's `disable_adapter()` context as the "fixed teacher" trick — same model
  serves as both student (LoRA on) and teacher (LoRA off).

**Hint phrasing tested (on base Qwen3-0.6B)**:
| Phrasing | Avg compliance |
|---|---|
| weak | 44% |
| directive | 50% (but strong "right" bias) |
| expert_says | 46% |
| in_system | 31% |
| expert_explain | 33% |
| expert_persuade | 17% |
| **consider_then_decide** | **73%** (balanced) |

Chosen: `consider_then_decide` for actual training.

**Training**:
- Seed: base Qwen3-0.6B (NOT `rl_enum_lora_prp` — its reasoning was corrupted)
- LoRA rank 32, alpha 64, lr 5e-5
- JSD forward-KL (beta=0), jsd_token_clip=0.05
- batch_size=4, max_new_tokens=1024, oracle_depth=3
- 100 steps, ~6.5 hours wall-time

**Training metrics**:
| Step | Loss | AvgTok |
|---|---|---|
| 0 | +0.004 | 1024 |
| 25 | -0.061 | 1024 |
| 50 | -0.070 | 940 |
| 75 | -0.076 | 1024 |
| 100 | -0.087 | 775 |

Loss negative is expected per OPSD readme — per-vocab clipping creates a sum
that can go below 0 without changing gradient direction.

**Unlimited eval (max_moves=1000, max_invalid_moves=5, n=16 games)**:
| Model | Mean tile | Max tile | Mean score | Mean turns | Mean valid% |
|---|---|---|---|---|---|
| Base Qwen3 (max_tokens=2048) | 26 | 64 | 177 | 93 | 55 |
| Base Qwen3 (max_tokens=6000) | **60** | **128** | **577** | **174** | (better) |
| OPSD step_25 (max_tokens=2048) | 3 | 4 | 1 | 14 | 7 |
| OPSD step_50 (max_tokens=2048) | 3 | 4 | 1 | 13 | 5 |
| OPSD step_75 (max_tokens=2048) | 3 | 4 | 1 | 15 | 7 |
| OPSD step_100 (max_tokens=2048) | 8 | 32 | 24 | 36 | 22 |
| OPSD step_75 (max_tokens=6000) | 44 | 128 | 367 | 150 | (decent) |

**Findings**:
1. **OPSD destroyed valid_moves_ratio at low budget**. With max_tokens=2048
   the model became 7-22% valid (vs base 55%). Initially looked catastrophic.
2. **Trace inspection showed reasoning is actually COHERENT**. The model
   parses the board, considers merges, picks moves. Just over-thinks.
3. **OPSD made the model verbose**: step_50 thinks for 18000+ chars per turn
   (vs base 8000 max). Under tight token budgets, every turn truncates with
   no answer → invalid moves → games die fast.
4. **With matched generous budget (max_tokens=6000)**: base Qwen3 = 60 mean
   tile, OPSD step_75 = 44 mean tile. **Base is meaningfully better.**
5. **Token-cap during training is the root cause**: max_new_tokens=1024 in
   training meant student rollouts always saturated the cap. Model trained
   to elaborate reasoning indefinitely. Distillation reinforced this pattern.

**Net result**: OPSD as implemented is **worse than no training** by every
gameplay metric. Reasoning quality may improve subjectively, but at the cost
of unworkable verbosity.

**To make OPSD work, would need**:
- max_new_tokens=8000+ during training (so student rollouts complete naturally)
- Different teacher mechanism than free-form hint (e.g. force `</think>` after
  K tokens, or teacher-prefix injection)
- Or stronger oracle that produces meaningfully different teacher outputs
  (current: ~73% compliance means 27% of states give noisy teacher signal)

**Conclusion / next direction**:
GRPO on long rollouts is sample-inefficient because recovering from bad moves
is harder than not making them, and rollouts within a group end up broadly
similar. OPSD's premise (distill teacher → student) was reasonable but the
implementation here hit failure modes that compounded. A value-function-based
method (PPO/REINFORCE-with-critic) may better handle the per-token credit
assignment issue.

---


### Exp 16 — Stage 3 multi-turn RL: BASE and LoRA on real 2048 game
- **When**: 2026-06-12 → 2026-06-15 (BASE 281 steps; LoRA ongoing, at step ~40)
- **Env**: `env_2048_text` with new `output_format="json"`
- **Status**: BASE killed at step 281, LoRA running

**Goal**: First real Stage-3 run. Train Qwen3-0.6B to actually play multi-turn 2048
with the reward signal driving max-tile-reached. Compare base full-FT vs
LoRA-warm-start from the single-turn SFT pick_one model.

**Implementation**:
- Added `output_format: "xml" | "json"` to `env_2048_text`. JSON parser is the
  same tolerant `_extract_move_json` from single-turn env. Backward compat: XML
  is still default.
- Fixed `enable_thinking` plumbing: vLLM `extra_body.chat_template_kwargs.enable_thinking`
  is silently dropped unless `trust_request_chat_template=true` is set via
  `[inference.vllm_extra]`.
- All enum runs (Exp 13-15) accidentally trained with thinking ON because the
  template default is True even though our `extra_body.enable_thinking` was
  ignored. Intent matched accidentally.
- Trimmed `max_moves` (100/150/200/250 by env size), `max_invalid_moves=5`,
  eval `rollouts_per_example=2`, `max_tokens=2048`. Cut step time ~10× vs
  initial config.

**Configs**: `env_2048_text/configs/rl_base_json.toml`, `env_2048_text/configs/rl_lora_json.toml`

**Reward**: 0.8 * `max_tile_reward` + 0.2 * `valid_moves_ratio`
- `max_tile_reward = log2(max_tile) / log2(target)`
- `valid_moves_ratio = valid / (valid + invalid)`

**BASE results (Qwen3-0.6B full FT, 281 steps before kill):**

| Step | 4x4@2k | 4x4@4k | 5x5@4k | 5x5@8k | Mean | AvgLen |
|---|---|---|---|---|---|---|
| 0 | 0.460 | 0.403 | 0.489 | 0.490 | 0.461 | 1431 |
| 25 | 0.555 | 0.512 | 0.572 | 0.526 | 0.541 | 1275 |
| **50** | **0.579** | **0.602** | **0.673** | **0.650** | **0.626** | 737 |
| 100 | 0.497 | 0.497 | 0.617 | 0.582 | 0.548 | 1511 |
| **200** | 0.611 | 0.574 | 0.676 | 0.639 | **0.625** | 1358 |
| 275 | 0.453 | 0.512 | 0.574 | 0.566 | 0.526 | 1843 |

Training rewards bounced 0.40-0.57. `valid_moves_ratio` climbed from 50% → 80%+.
Avg max tile achieved went from ~15 to ~50-65.

**Saved BASE ckpts**: step_200 (mean 0.625), step_250 (mean 0.561). step_50 was
best (0.626) but lost to keep_last=2.

**LoRA results (from `lora_outputs_json/merged`, LoRA rank 16, lr 5e-5):**

Step 0 eval (the SFT seed evaluated with the multi-turn env):
| Env | LoRA step 0 | Base step 0 |
|---|---|---|
| 4x4-2048 | 0.417 | 0.460 |
| 4x4-4096 | 0.416 | 0.403 |
| 5x5-4096 | 0.415 | 0.489 |
| 5x5-8192 | 0.309 | 0.490 |
| **Mean** | **0.389** | **0.461** |

The SFT seed eval was WORSE than base at step 0. Counterintuitive but explained:
the SFT model thinks more (100% truncated at 2048 tokens, 1633 char avg vs
1431 base) and was trained for single-turn not multi-turn.

Step 25 eval:
| Env | LoRA step 25 | Base step 25 |
|---|---|---|
| 4x4-2048 | 0.593 | 0.555 |
| 4x4-4096 | 0.529 | 0.512 |
| 5x5-4096 | 0.595 | 0.572 |
| 5x5-8192 | 0.635 | 0.526 |
| **Mean** | **0.588** | **0.541** |

**LoRA at step 25 beats base at step 25 on every env.** Biggest gain is
5x5-8192 (+21%) where SFT prior really helps.

LoRA training trajectory:
| Step | max_tile_R | implied tile | valid% | reward | too-many-inv% |
|---|---|---|---|---|---|
| 0 | 0.446 | 30 | 60.6 | 0.478 | 75 |
| 9 | 0.545 | 64 | 79.8 | 0.596 | 3.1 |
| **17** | 0.562 | 73 | **85.0** | **0.620** | 6.2 |
| 25 | 0.523 | 54 | 79.4 | 0.577 | 9.4 |
| 39 | 0.511 | 49 | 77.6 | 0.564 | 41 |

**Key findings:**
1. **Valid% plateaus at ~80-85% in think mode** (vs SFT seed's 100% in no-think).
   The plateau is a thinking-mode ceiling, not RL-induced.
2. **SFT think-mode quality was already poor** (62-71% valid). RL improved this
   to 80-85% but couldn't reach the no-think 100%.
3. **Reward + valid% peaked at step 17 then declined** — same plateau pattern as
   base (peaked step 50, declined past step 200). LoRA-RL doesn't seem to
   escape the wall.
4. **Format errors are essentially solved** (too-many-invalid stop drops from
   75% at step 0 to 3-12% by step 17). Remaining ~20% invalid moves are
   "valid direction picked but doesn't move tiles on this board" — i.e.,
   gameplay mistakes, not format mistakes.
5. **LoRA training is ~2x slower per rollout than base** because the SFT'd
   model plays longer games (fewer invalid-move deaths) AND produces longer
   completions. ~17 min/step. ETA for 400 steps: ~4.7 days.

**Truncation**: 100% in all eval envs at step 0 and step 25. The thinking trace
hits the 2048-token cap every turn. Parser is tolerant enough to recover moves
from partial JSON, so reward signal still works. But cap is real bottleneck.

**Open question**: Should we restart from `rl_enum_lora_prp/weights/step_200/`
(the only stage-2 model that preserved thinking) instead? It already RL-trained
with thinking ON for 200 steps, even though on a different task format
(enumerate, not pick_one). Hypothesis: it might adapt to pick_one format
quickly because it already follows JSON-output instructions.

---


### Exp 15 — Data distribution variants: UNIFORM and ADAPTIVE difficulty
- **When**: 2026-06-11 evening
- **Env**: `env_2048_single_turn` enumerate task, em_plus_prp reward
- **Status**: ✅ done

**Goal**: Break the "predict size=2 always" collapse from Exp 14. Two hypotheses:
1. **UNIFORM**: Re-balance training data to 33/33/33 instead of 75/12.5/12.5. Removes the safe "match the mode" shortcut.
2. **ADAPTIVE**: Enable prime-rl's `online_difficulty_filtering=true`. Drops groups where all rollouts get the same reward (zero advantage). Should naturally drop "all collapsed to same size" groups.

**Implementation**:
- Added `difficulty_distribution` param to `load_environment` and `generate_single_turn_dataset`. Takes a 3-tuple (frac_2valid, frac_3valid, frac_4valid).
- Both runs from LoRA-JSON, em_plus_prp reward, no length penalty (to isolate distribution effects).

**vf-vllm eval (n=150 per condition):**

| Run | mode | 4x4 | 5x5 | 6x6 | D2 EM | D3 EM | D4 EM | chars |
|---|---|---|---|---|---|---|---|---|
| lora+PR (Exp 13) | both | 76% | 76% | 76% | **100%** | 0% | 0% | 33-3382 |
| lora+EM+PR la=.05 | both | 70% | 72% | 67% | 92% | 0% | 0% | 65-70 |
| lora+EM+PR UNIFORM (no-think) | - | 32% | 28% | 19% | 20% | **61%** | **30%** | 49 |
| lora+EM+PR UNIFORM (think) | - | 26% | 19% | 16% | 14% | **56%** | **28%** | 1016 |
| lora+EM+PR ADAPTIVE (no-think) | - | 41% | 50% | 49% | **51%** | **56%** | 11% | 170 |
| lora+EM+PR ADAPTIVE (think) | - | 1% | 0% | 0% | 0% | 0% | 0% | **13505** |

**Findings:**
1. **UNIFORM escaped the predict-2 collapse**: first run ever to have non-zero EM on D3 (61%) and D4 (30%). But traded D2 mastery (100% → 20%) for variable set size capability.
2. **ADAPTIVE also escaped**, in no-think mode: D2=51%, D3=56%, D4=11%. Best balance among the trap-escaping runs.
3. **ADAPTIVE think mode broke completely**: 0% across all conditions, 13505 chars avg, 100% truncated. The online filtering caused training instability — model swung between concise and runaway-yappy states.
4. **The data distribution dominates over reward formulation**. Same reward (em_plus_prp), but uniform/adaptive distribution alone breaks the trap that no reward tuning broke.
5. **D3 is now the EASIEST tier for uniform** (61% > 30% D4 > 20% D2). Model now over-predicts size 3.

**Interpretation**: The model under uniform learned to predict ~3 directions on average (the mean of the uniform distribution). Under adaptive, it learned a slightly broader distribution (predicts ~2-3) but with less stability.

**Best for D3/D4 performance**: UNIFORM (no-think). Combined EM across whole dataset is lower (~26%) but it actually generalizes.

**Best overall EM (counting D2 dominance)**: lora+PR (Exp 13) still wins at 76% because of the 75% D2 base rate.

**Open question**: a model that gets 76% by collapsing to D2 is not necessarily worse for downstream RL (Stage 3) than one that's 26% but generalizes. The right question is which transfers better. Could test by warm-starting Stage 3 RL from each.


### Exp 14 — RL with em_plus_prp reward and length-penalty sweep (5 runs)
- **When**: 2026-06-11 morning → afternoon
- **Env**: `env_2048_single_turn` enumerate task
- **Status**: ✅ done

**Motivation**: Exp 13's PR_penalty reward got 76% EM but collapsed to "always predict exactly 2 directions" (0% on 3/4-valid boards). Hypothesis: combine EM with PR_penalty so the perfect-match jump is bigger than the safe-2 policy's plateau, plus tune length_alpha to control eval-time runaway-think.

**Two bugs found and fixed before runs:**
1. **Eval-time max_tokens was unlimited** because `[orchestrator.eval.sampling]` wasn't set. Late-training models generated 40K-char outputs (10K+ tokens), evals took 30-60+ minutes each. Fix: set `max_tokens = 3072` in `[orchestrator.eval.sampling]` to match training.
2. **Predict-all-4 shortcut was actually a predict-same-size-every-time collapse** under any partial-credit reward (including F1 from Exp 13). All 5 variants in this exp collapsed similarly: model picks a fixed size, makes it match the modal training-data difficulty (size 2 in 75% of boards).

**Reward**: `em_plus_prp = EM + recall * max(0, 1 - 2*FP/4)`. Range [0, 2]. Exact match = 2.0; predict-all on 2-valid = 0.0; predict-2 on 2-valid (correct) = 1.0.

**Length reward** (when used): `-alpha * k_q * |y_i| / l_max` (group-level).

**Variants run (each 200 steps, batch 32 × 8 rollouts/example, lr 5e-6):**

| Run | starter | length_alpha | result |
|---|---|---|---|
| base_emprp | Qwen3 | 0.05 | collapsed to 25-char constant `<think>{json}</think>{json}` |
| lora_emprp | LoRA-JSON | 0.05 | collapsed to 18-char `{"valid_moves":[]}` constant |
| lora_emprp_la00 | LoRA-JSON | 0 | weaker collapse, ~400 char outputs |
| lora_emprp_la01 | LoRA-JSON | 0.1 | collapsed to 12-char constant |
| lora_emprp_la02 | LoRA-JSON | 0.2 | crashed at step 21 (disk full) |

**vLLM offline eval (n=150 per condition):**

| Run | 4x4 EM | 5x5 EM | 6x6 EM | D2 EM | D3 EM | D4 EM | AvgChars |
|---|---|---|---|---|---|---|---|
| lora+PR (Exp 13 baseline) | 75% | 76% | 76% | **100%** | 0% | 0% | 3382 |
| base+EM+PR la=.05 | 22% | 13% | 7% | 19% | 0% | 0% | 81 |
| lora+EM+PR la=.05 | 70% | 72% | 67% | 92% | 0% | 0% | 70 |
| lora+EM+PR la=0 | 67% | 63% | 57% | 82% | 0% | 0% | 386 |
| lora+EM+PR la=.1 | 68% | 65% | 59% | 84% | 0% | 0% | 453 |

**Findings:**
1. **EM+PR didn't escape the predict-2 trap.** All variants hit 0% on 3-valid and 4-valid boards regardless of length_alpha. The combined reward widens the gradient toward "exact" but in practice GRPO still finds and stays at the safe constant policy.
2. **Length_alpha mostly controls output style, not accuracy.** la=0.05 produces ~70-char outputs (super concise), la=0.1 produces ~450 chars, la=0 produces ~400 chars (similar). EM ranges 67-72% across all 3 — no clear monotonic relationship.
3. **base+EM+PR completely failed** (7-22% EM). The base Qwen3 needed too many initial right answers to escape the predict-all/predict-empty traps under GRPO.
4. **Length_alpha + length-reward formula has a perverse effect**: model generated `</think>` repeatedly as "thinking" tokens to satisfy the budget without doing real reasoning.
5. **Eval-time cap is critical infrastructure.** Without it, late-training evals take 30-60 min each because the model develops the habit of generating until context limit.
6. **Disk consumption**: each run dumps ~110GB (FSDP checkpoints + rollouts + wandb). 9 runs filled the 1.8TB disk to 100%, causing one crash. Future runs need cleanup-as-you-go.

**Conclusion**: The data distribution itself is the root cause. Training on 75% 2-valid means any reward that gives partial credit lets the model find a stable 2-valid-policy plateau. To break this, either (a) re-balance training to 33/33/33, (b) curriculum starting with 4-valid-only and ramping down, (c) abandon the enumerate task entirely.

**Best checkpoint still**: `rl_enum_lora_prp/weights/step_200/` (Exp 13) at 76% EM.


### Exp 13 — Stage 2 RL on enumerate-valid-moves task (4 runs)
- **When**: 2026-06-10 evening → 2026-06-11 early morning
- **Env**: `env_2048_single_turn` with `task_type=enumerate_all`
- **Status**: ✅ done (3 to completion, lora_em killed at step 140 due to runaway-think eval slowdown)
- **Goal**: RL on the new enumerate task, comparing PR_penalty reward vs pure EM, both from base Qwen3 and from LoRA-JSON SFT model.

**Reward designs explored:**
- **First attempt (`0.7 EM + 0.3 F1 + 0.1 length`)**: Killed early. Model converged
  to predicting `{"valid_moves":["up","down","left","right"]}` on every board within
  ~30 steps. F1 partial credit (~0.73 for "predict all") made it the trivial local
  optimum. Length reward reinforced collapse (predict-all is short).
- **Final design**:
  - **PR_penalty**: `recall × max(0, 1 - 2 × FP/4)`. Exact=1.0; predict-all on
    2-valid=0.0; predict-all on 3-valid=0.5; predict-all on 4-valid=1.0.
    Subset predictions still get recall credit. No length reward.
  - **EM-only**: 1.0/0.0 exact match.

**vLLM offline final eval (n=150 per condition, balanced 75/12.5/12.5)**:

  | Run | mode | EM% | F1 | 2-valid EM | 3-valid EM | 4-valid EM | avg chars |
  |---|---|---|---|---|---|---|---|
  | base+PR_penalty | no-think | 21% | 0.55 | 27% | 0% | 0% | 46 |
  | **lora+PR_penalty** | both | **76%** | **0.94** | **100%** | 0% | 0% | 3300 |
  | base+EM | both | 12% | 0.73 | 0% | 0% | 100% | 65 |
  | lora+EM step_140 | no-think | 41% | 0.56 | 54% | 0% | 0% | 590 |
  | lora+EM step_140 | think | 49% | 0.54 | 65% | 0% | 0% | 12800 |

  Compare to baselines (Exp 12):
  - Base+think: 8% EM. Base+no-think: 11% EM.
  - LoRA+think: 15% EM. LoRA+no-think: 63% EM.

**Key findings:**
1. **`lora+PR_penalty` hit 100% EM on 2-valid boards** (75% of the distribution).
   Massive +63 percentage points over the baseline, and the result that actually
   matters since 2-valid is by far the hardest case.
2. **All 4 runs found the same kind of shortcut: always predict the same SET SIZE.**
   - PR_penalty runs converge to "predict exactly 2 directions" (or sometimes 0/1)
   - base+EM converges to "predict all 4" (gets 100% on 12.5% of boards)
   - This is because the training distribution is 75% 2-valid, so the easy local
     optimum is "match the modal set size"
3. **PR_penalty is genuinely solving the 2-valid task** (the dominant + hardest one).
   100% on it requires actually computing which 2 directions are valid, not just
   guessing a fixed pair.
4. **Pure EM from base failed completely** (12% baseline, didn't move). Group reward
   variance from 8% baseline isn't enough for GRPO to find the gradient before
   collapsing to predict-all.
5. **Pure EM from LoRA partially worked** (~50% EM mid-training) but then started
   yapping (10-20% of eval responses got truncated at 4096 tokens). Reward
   hyperparameter is too volatile for stable convergence.
6. **Runaway-think problem at eval time**: in later training, some rollouts go to
   ~40K chars (would require >4096 tokens to close `</think>`). Eval times went
   from 45s → 1200s+ per condition. Length reward will be needed to control this.

**Next steps surfaced:**
- Re-balance training distribution to include more 3-valid and 4-valid boards so
  the model learns to predict variable set sizes
- Add length reward back (small, e.g. alpha=0.05) to control runaway-think
- Consider tightening the PR_penalty further (e.g. floor reward at 0 for ANY FP)


### Exp 12 — Baseline eval on enumerate-valid-moves task
- **When**: 2026-06-04 early
- **Env**: `env_2048_single_turn` (new `task_type=enumerate_all` mode)
- **Status**: ✅ done
- **Goal**: Establish baselines before RL. New task: model outputs `{"valid_moves": ["up", "left"]}` — the full set of valid directions. Reward = 0.7 × EM + 0.3 × set-F1.
- **Why this task**: pick_one collapses to constant policies in RL (Exp 08); enumerate has a hard ceiling that constant policies can't reach (max ~12.5% EM via "predict all 4"), so RL has a real hill to climb. Requires actual board analysis.
- **vLLM offline eval (n=150 per condition, balanced 75/12.5/12.5)**:

  | Model | mode | EM% | F1 | blend | parsed% | avg chars | closed-think% | opened-only-think% |
  |---|---|---|---|---|---|---|---|---|
  | Base Qwen3 | no-think | 11% | 0.685 | 0.280 | 100% | 56 | 0% | 0% |
  | Base Qwen3 | think | 8% | 0.525 | 0.213 | 84% | 9527 | 81% | ~19% |
  | LoRA-JSON | no-think | 63% | 0.899 | 0.709 | 100% | 37 | 0% | 0% |
  | LoRA-JSON | think | 15% | 0.442 | 0.235 | 65% | 10548 | 60% | ~40% |

  (opened-only-think = response begins `<think>` but the response was truncated before `</think>`; estimated as n − closed-think.)

- **Findings**:
  1. **Thinking actively HURTS both models** in baseline. Base goes 11→8% EM, LoRA goes 63→15% EM. Output explodes from ~50 chars to ~10k chars of waffle.
  2. **LoRA-JSON no-think (63% EM) is a strong free baseline** — the SFT-learned "what is a valid move" knowledge transfers to enumeration without any new training.
  3. **Per-difficulty pattern flips for LoRA no-think**: 75% EM on 2-valid boards, 17% EM on 4-valid (LoRA was trained to output ONE direction, so naming all 4 is OOD).
  4. **A huge fraction of think-mode failures are token-budget truncations**, not reasoning failures. LoRA-JSON 6×6 think: 53% of responses never close `</think>`. Length reward will have a huge effect.
  5. **Decision**: RL with thinking enabled is justified — there's plenty of headroom to climb from 8-15% EM, and the length reward attacks the dominant failure mode (yapping).


### Exp 11 — Full SFT XML vs JSON, with chat_template_kwargs masking
- **When**: 2026-06-03 late evening – early 2026-06-04
- **Env**: `env_2048_single_turn`
- **Status**: ✅ done
- **Goal**: Does the XML-vs-JSON thinking-preservation gap from LoRA (Exp 10)
  also hold under full FT? Or is it a LoRA-only effect?
- **Setup**: same 10k datasets and `chat_template_kwargs={"enable_thinking": False}`
  per-example as Exp 10, but full FT via prime-rl (200 steps, lr 2e-5, batch 64,
  2 GPU FSDP)
- **vf-eval results (n=150 per condition, balanced difficulty)**:
  | Model | 4x4 nothink | 4x4 think | 5x5 nothink | 5x5 think | 6x6 nothink | 6x6 think |
  |---|---|---|---|---|---|---|
  | Full SFT XML  | 0.778 | **0.000** | 0.868 | **0.000** | 0.958 | **0.000** |
  | Full SFT JSON | 1.000 | **0.000** | 1.000 | **0.000** | 1.000 | **0.000** |
- **Findings**:
  1. **Full SFT kills thinking entirely in BOTH formats** (0.000 reward across
     every think condition), even with the chat_template_kwargs masking.
  2. **Full SFT JSON outperforms full SFT XML on no-think** (1.000 vs 0.778-0.958).
     XML is harder to learn under full FT — possibly because tag emission is
     less compositionally natural than JSON for the small 0.6B model.
  3. **Revises Exp 10's headline**: thinking preservation comes from **LoRA
     itself**, not from JSON format. JSON helps further within LoRA but isn't
     the primary driver.

### Exp 10 — vf-eval parity check
- **When**: 2026-06-03 evening
- **Env**: `env_2048_single_turn`
- **Status**: ✅ done
- **Goal**: Confirm custom vLLM eval matches the verifiers framework that RL
  will see. Same checkpoints as Exp 09, evaluated with `vf-eval` against a
  vLLM OpenAI server using `--reasoning-parser qwen3`.
- **vf-eval results (n=150 per condition, balanced difficulty)**:
  | Model | 4x4 nothink | 4x4 think | 5x5 nothink | 5x5 think | 6x6 nothink | 6x6 think |
  |---|---|---|---|---|---|---|
  | Base XML  | 0.604 | 0.499 | 0.628 | 0.451 | 0.688 | 0.464 |
  | Base JSON | 0.622 | 0.505 | 0.628 | 0.433 | 0.610 | 0.511 |
  | LoRA XML  | 0.988 | 0.351 | 1.000 | 0.515 | 0.994 | 0.419 |
  | LoRA JSON | 1.000 | 0.649 | 0.994 | 0.639 | 0.994 | 0.605 |
- Numbers match the custom vLLM eval within sampling variance. Conclusion: our
  fast custom evaluator is trustworthy.
- vf-eval reward: 1.0 valid move, 0.1 parseable-but-no-tile-change, 0.0 unparseable.

### Exp 09 — LoRA SFT, XML vs JSON formats
- **When**: 2026-06-03 morning
- **Env**: `env_2048_single_turn`
- **Status**: ✅ done — first success at preserving thinking
- **Goal**: Train a LoRA on valid moves that preserves Qwen3's thinking
- **Setup**:
  - Added `output_format: xml|json` to env (XML default, JSON alternative)
  - HF transformers + PEFT + TRL `SFTTrainer` (prime-rl's SFT can't do LoRA
    without `MultiRunManager`; see Exp 06)
  - LoRA rank 16, alpha 32 on q/k/v/o/gate/up/down
  - 3 epochs, batch 8 × grad_accum 4, lr 2e-4
  - 10k examples per dataset, mixed 4×4/5×5/6×6 grids
  - **Key trick**: `chat_template_kwargs={"enable_thinking": False}` per example
    so TRL's loss mask excludes the empty `<think></think>` block that Qwen3
    auto-injects before the last assistant turn
- **Custom vLLM eval (n=100 per condition, hard balanced)**:
  | Model | 4x4 nothink | 4x4 think | 5x5 nothink | 6x6 nothink |
  |---|---|---|---|---|
  | Base XML  | 52% | 43% | 66% | 61% |
  | Base JSON | 63% | 45% | 62% | 54% |
  | LoRA XML  | **100%** | 42% | **100%** | **100%** |
  | LoRA JSON | **100%** | **71%** | **100%** | **99%** |
- **First headline** (later revised by Exp 11): LoRA-JSON beats LoRA-XML on
  thinking preservation (77% vs 48% have non-empty `<think>` blocks).
- **Refined story**: LoRA itself preserves thinking; JSON gives extra headroom.

### Exp 08 — Single-turn RL collapse analysis
- **When**: 2026-06-02 (prior session)
- **Env**: `env_2048_single_turn`
- **Status**: ❌ negative finding (informative)
- **Goal**: RL from base Qwen3 with thinking enabled, on single-turn valid-move task
- **Setup**: base Qwen3, `reasoning_parser=qwen3`, `enable_thinking=true`
- **Initial eval**: 4x4=0.620, 5x5=0.642, 6x6=0.658
- **Result**: Model collapsed to "always up" by step 10. Thinking blocks
  preserved in format but reasoning vacuous. Final eval ≈ 0.577 — matches the
  "always up" empirical accuracy on balanced data exactly (58%).
- **Why**: Any constant policy scores ~59.4% on balanced data
  (75% × 50% + 12.5% × 75% + 12.5% × 100%). RL has zero incentive to reason.
- **Implication**: Single-turn valid-move task is RL-trainable in theory but
  the reward shape collapses to a constant policy. Either change reward (best
  move, score gained, merges) or move to multi-turn for richer signal.

### Exp 07 — Full SFT (no-think) on valid moves
- **When**: 2026-06-02 (prior session)
- **Env**: `env_2048_single_turn`
- **Status**: ✅ task solved, ❌ thinking destroyed
- **Setup**: 200 steps full FT on 10k XML examples (no `chat_template_kwargs`
  field — empty `<think></think>` block was included in supervised tokens)
- **Result**: 100% valid moves on 4×4/5×5/6×6 test, BUT `<think>` block
  replaced by `<well>` garbage at inference time. Total catastrophic forgetting
  of the thinking template.
- **Motivated Exp 09 (LoRA approach)**.

### Exp 06 — Difficulty-balanced data generation
- **When**: 2026-06-02 (prior session)
- **Env**: `env_2048_single_turn`
- **Status**: ✅ done
- **Problem**: Random-board generation produced too-easy boards (~88% baseline
  success on untrained model). Multi-turn training was hitting ~0.72 valid-move
  ratio, so single-turn needed harder data.
- **First attempt**: `generate_state_with_valid_move_count()` — trial-and-error,
  failed for 5×5/6×6 (only 21% success at targeting 2-valid for 5×5).
- **Working approach** (`generate_playthrough_states()`): start from
  partially-filled board (varying fill ratio 0.4–0.95), play random valid
  moves, snapshot states every N moves. Naturally produces realistic boards
  at all difficulty levels.
- **Bucket distribution**: 75% hard (2 valid), 12.5% medium (3 valid),
  12.5% easy (4 valid). Works for ALL grid sizes (4×4, 5×5, 6×6).
- **Random baseline**: consistent at ~59.4% across all sizes.
- **1-valid-move states**: discovered they're practically impossible in 2048
  (a merge always re-enables the reverse direction; empty cells reachable from
  2+ directions). Hard floor is 2 valid moves. Removed dead generator code.

### Exp 05 — Fixed grid data bug in verifiers
- **When**: 2026-06-02 (prior session)
- **Env**: `env_2048_single_turn`
- **Status**: ✅ done (critical bug fix)
- **Symptom**: Rewards stuck at 0.1; model collapsed to "always up"; all
  moves returned `moved=False`.
- **Root cause**: `state["game"]` had all-zero grid cells. The dataset put
  `grid`, `size`, `score`, `target_tile` at the top level of rows, but
  verifiers `RolloutInput` only preserves `prompt`, `example_id`, `task`,
  `answer`, `info`. Everything else gets dropped silently.
- **Fix**: wrap custom data in `info` dict on the dataset side; read from
  `state["input"]["info"]` in `setup_state`.
- **Impact**: success rate jumped from ~9% to ~88% on untrained model. Made it
  clear the task was too easy → motivated Exp 06's difficulty balancing.

### Exp 04 — Fixed single-turn reward function (rewrite via verifiers framework)
- **When**: 2026-06-02 (prior session)
- **Env**: `env_2048_single_turn`
- **Status**: ✅ done
- **Issues fixed in this round**:
  1. Dataset prompts were strings; verifiers needs message lists.
  2. max_tokens=256 too tight for thinking model — went to 1024.
  3. `score_state()` was never called by verifiers; moved scoring logic
     directly into `valid_move_reward()` reward function.
  4. System prompt was missing reasoning example and `<move>` reminder.
- **Result**: First non-zero rewards (Avg@1 ≈ 0.09 on untrained model).

### Exp 03 — Single-turn environment creation
- **When**: 2026-06-01 (prior session)
- **Env**: `env_2048_single_turn` (new)
- **Status**: ✅ done
- **Goal**: Faster training for the "valid moves" stage of the curriculum by
  using a single-turn formulation (one board → one move) instead of the
  multi-turn game.
- **Setup**: Created package with play-forward board generation (start fresh,
  play 0–100 random moves, take resulting state). Initial batch_size=64,
  rollouts_per_example=8.
- **Tooling added in same session**: `num_moves_reward`, `efficiency_reward`,
  reward weights configurable via TOML. Curriculum config files for the
  multi-turn env: `valid_moves.toml`, `max_tile.toml`, `efficiency.toml`.

### Exp 02 — Debugging zero-gradient RL training
- **When**: 2026-06-01 (prior session)
- **Env**: `env_2048_text` (multi-turn)
- **Status**: ✅ root cause found and documented
- **Symptom**: Training ran 100 steps with `grad_norm=0.0`, `loss=0.0`,
  `entropy=NaN` — no learning. Happened in both full and markov context modes.
- **Investigation path**: rollouts complete with reward variation (0.18-0.25),
  but completion_mask was ALL zeros (sum=0) for all 512 examples. ~25,863
  "Aborted rollout" errors in train log.
- **Root cause**: `get_prompt_messages()` returned raw dicts instead of
  Pydantic `Message` objects (`SystemMessage`, `UserMessage`, etc.). The vLLM
  client's `to_native_prompt()` does `isinstance` checks → raises
  `ValueError("Invalid chat message")` → `state["error"]` is set → trajectory
  processing zeros out `completion_mask` → zero trainable tokens.
- **Fix**: wrap returns with `normalize_messages()` from
  `verifiers.utils.message_utils`, OR return Pydantic Message objects directly.
- **Documented in**: `env_2048_text/PITFALLS.md` (the most painful debug of
  the project so far — worth reading).

### Exp 01b — Multi-turn context modes + full-history logging
- **When**: 2026-05-31 / 2026-06-01 (prior sessions)
- **Env**: `env_2048_text`
- **Status**: ✅ done
- **Key work**:
  - Implemented MARKOV mode but it sent only `[system, current_state]` to the
    LLM. Initial bug: it returned early without calling `env_response`, so
    game state never updated. Fix: always call `super().get_prompt_messages()`
    first, then transform output.
  - Even after fix, completion logs only had last turn pair (because trajectory
    stores what `get_prompt_messages` returns). Fix: manually maintain
    `_full_messages_for_logging` in state; override `render_completion()` to
    use it.
  - LAST_K mode initially used placeholder env responses; rewrote to pull real
    env responses from full_messages.
- **Result**: All 4 context modes (FULL, MARKOV, LAST_K, SUMMARY) preserve
  full history in logs while sending condensed prompts to the LLM. Test
  coverage for each mode.

### Exp 01 — 2048 multi-turn environment with context modes (initial build)
- **When**: 2026-05-30 / 2026-05-31 (prior session)
- **Env**: `env_2048_text` (new)
- **Status**: ✅ done
- **Goal**: Build an RL environment for training LLMs to play 2048
- **Setup**:
  - Studied the [2048 reference implementation](https://github.com/gabrielecirulli/2048) for game mechanics
  - Created `Grid`, `Game2048`, `Game2048Env(vf.MultiTurnEnv)` classes
  - Configurable `grid_size`, `target_tile`
  - 4 context modes: FULL (default history), MARKOV (stateless),
    LAST_K (last k turns), SUMMARY (LLM-generated summaries every N turns)
  - Reward functions: `win_reward`, `max_tile_reward`, `score_reward`,
    `valid_moves_ratio`, `efficiency_reward`
  - Comprehensive test suite (18 test functions)
- **2048 mechanics that matter**:
  - New tiles: 2 (90%) or 4 (10%), unaffected by max tile
  - Tiles merge at most once per move (tracked via merged flag)
  - Traversal order: process opposite to movement direction
  - Game over: no empty cells AND no adjacent matches

---

## Comparison matrix on 4x4 hard-balanced (vf-eval, n=150)

| Model | no-think | think (4096 tok) |
|---|---|---|
| Base Qwen3 XML  | 0.604 | 0.499 |
| Base Qwen3 JSON | 0.622 | 0.505 |
| Full SFT XML    | 0.778 | **0.000** |
| Full SFT JSON   | 1.000 | **0.000** |
| LoRA XML        | 0.988 | 0.351 |
| **LoRA JSON**   | **1.000** | **0.649** |

---

## Key technical decisions & gotchas (durable knowledge)

### Verifiers / message-handling
- **`RolloutInput` only preserves these fields**: `prompt`, `example_id`,
  `task`, `answer`, `info`. Put anything else in `info` or it gets dropped.
- **`get_prompt_messages()` MUST return Pydantic `Message` objects**, not raw
  dicts. Raw dicts → zero gradients (silent). Use `normalize_messages()` to
  convert. See `env_2048_text/PITFALLS.md`.
- **Trajectory stores what `get_prompt_messages` returns.** In non-FULL
  context modes, override `render_completion()` to use a separately-maintained
  full-history state field, otherwise logs only have the last pair.

### Qwen3-specific
- **Qwen3 chat template auto-injects `<think>\n\n</think>` before the last
  assistant turn** regardless of `enable_thinking` setting. To prevent SFT
  from training on that empty block, pass
  `chat_template_kwargs={"enable_thinking": False}` per example so the
  prompt-side and prompt+completion-side renders include the same prefix.
- **Even with the masking trick, full FT destroys thinking.** Use LoRA
  (rank 16+) if thinking preservation matters.
- **Inference-time thinking control**: `reasoning_parser="qwen3"` in vLLM
  config separates `<think>` from content; `enable_thinking=true` in
  `extra_body` is the chat template kwarg.

### prime-rl operational
- **Launch**: `cd prime-rl && PATH="$(pwd)/.venv/bin:$PATH" UV_FROZEN=1 .venv/bin/rl @ <config.toml>`.
  `UV_FROZEN=1` is critical (broken `deep-gemm` URL otherwise).
- **prime-rl SFT loads datasets via `load_dataset` (not `load_from_disk`).**
  Save as parquet directory, not via HF `save_to_disk`.
- **prime-rl's SFT trainer cannot use LoRA** without `MultiRunManager`
  (only initialized in the RL pipeline). Use HF TRL + PEFT in a standalone
  script. `merge_and_unload()` produces a regular HF checkpoint that vLLM and
  prime-rl can both load.
- **Checkpoints** at `<output_dir>/weights/step_X/` (vLLM-loadable);
  `<output_dir>/run_default/checkpoints/step_X/` is the orchestrator state.
- **Reward/std IS computed** (orchestrator.py:697) but only logged to wandb
  monitor, not stdout.
- **wandb offline mode** via `[trainer.wandb] offline = true`.

### 2048 specifics
- **1-valid-move states are practically impossible.** A merge always re-enables
  the reverse direction along that axis; empty cells are reachable from at
  least 2 directions. Hard floor is 2 valid moves.
- **Any constant move on balanced data → ~59.4%.** This is why naive RL on
  the single-turn valid-move task collapses to "always up".

### Evaluation
- **vLLM offline engine for fast iteration** — 10-50× faster than HF
  `model.generate` thanks to continuous batching.
- **vf-eval for parity with RL** — runs against a vLLM OpenAI server using
  `--reasoning-parser qwen3` and matches the reward path RL training will see.
- **vf-eval can hang/timeout when run in parallel**: env servers use ports
  that may collide. For parallel runs, spawn each in its own session via
  `setsid nohup ... < /dev/null` and use distinct ports.
- **Sharing GPUs** (~28GB occupied per A6000 by another user): vLLM server
  must use `--gpu-memory-utilization 0.34` to fit.

---

## File map

### Environments
- `env_2048_text/`
  - `env_2048_text.py` — Game logic + MultiTurnEnv with 4 context modes
  - `test_env.py` — 18 test functions covering all modes
  - `PITFALLS.md` — durable knowledge from debugging multi-turn RL
  - `configs/{rl,valid_moves,max_tile,efficiency}.toml` — curriculum configs

- `env_2048_single_turn/`
  - `env_2048_single_turn.py` — Single-turn env, supports `output_format=xml|json`
  - `generate_sft_data.py` — Playthrough-based balanced data generator
  - `train_lora_sft.py` — Standalone TRL + PEFT LoRA SFT script with sanity check
  - `eval_lora_vllm.py` — Fast vLLM offline evaluator (used in Exp 09)
  - `eval_lora.sh` — vf-eval against a vLLM server (used in Exps 10, 11)
  - `probe_thinking.py` — Qualitative HF probe for thinking-block inspection
  - `configs/{rl,rl_stage2,sft,sft_xml_full,sft_json_full,sft_lora}.toml`
  - `EXPERIMENTS.md` — Detailed log for this env
  - Checkpoints:
    - `lora_outputs_{xml,json}/{adapter,merged}/` — LoRA SFT (Exp 09)
    - `sft_outputs_{xml,json}_full/weights/step_200/` — Full SFT (Exp 11)
    - `sft_outputs/weights/step_200/` — Older full SFT (Exp 07)
    - `rl_outputs_new_data/weights/step_X/` — Earlier RL run
    - `rl_stage2_v2/` — Stage 2 RL collapse (Exp 08)
  - Datasets:
    - JSONL (LoRA train): `sft_data_{xml,json}/train.jsonl`
    - Parquet (prime-rl SFT): `sft_data_{xml,json}_parquet/train.parquet`
  - Logs: `logs/`, `eval_results/`, `probe_results/`, `rollout_examples/`

### External
- `prime-rl/` — RL trainer (vendored). Key files we reference often:
  - `src/prime_rl/trainer/sft/{train.py,data.py}` — SFT trainer
  - `src/prime_rl/trainer/rl/train.py` — RL trainer (with LoRA support)
  - `src/prime_rl/trainer/lora.py` — LoRA wiring (RL-only via MultiRunManager)
  - `src/prime_rl/trainer/runs.py` — MultiRunManager singleton
  - `src/prime_rl/orchestrator/orchestrator.py` — reward logging
  - `src/prime_rl/entrypoints/rl.py` — RL launcher (needs `UV_FROZEN=1`)
- `verifiers/` — Environment framework (vendored). Key files:
  - `verifiers/types.py` — `RolloutInput`, `AssistantMessage.reasoning_content`
  - `verifiers/utils/message_utils.py` — `normalize_messages`
  - `verifiers/envs/multiturn_env.py` — parent class for `Game2048Env`
  - `verifiers/clients/openai_chat_completions_client.py` — `to_native_prompt`
