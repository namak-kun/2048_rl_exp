# env-2048-single-turn

### Overview
- **Environment ID**: `env-2048-single-turn`
- **Short description**: Single-turn 2048 task — show a board, ask the model
  to either pick the next move or enumerate all valid moves
- **Tags**: game, 2048, single-turn, reasoning, train, eval

### Datasets
- **Source**: Procedurally generated mid-game 2048 boards via
  `generate_playthrough_states` (random valid plays from fresh start, snapshot
  every N moves)
- **Configurable**: grid size, target tile, fill ratio, difficulty distribution

### Task types

The env supports two task modes, controlled by `task_type`:

#### `pick_one` (default)
Show a board, model outputs one move. Used for SFT and as a multi-turn building block.

Prompt template (JSON format):
```
[System: 2048 rules]
Score: 196
Moves: 37
Max Tile: 32

+----+----+----+----+
| .  | 2  | .  | 2  |
+----+----+----+----+
| 16 | 8  | 4  | .  |
+----+----+----+----+
| 4  | 4  | 32 | 2  |
+----+----+----+----+

What's your move? Respond with {"move": "direction"}.
```

Expected response: `{"move": "right"}`.

#### `enumerate_all`
Show a board, model outputs the SET of all valid directions (moves that change the board).

Prompt: same board + "Respond with {"valid_moves": [...]}."
Expected response: `{"valid_moves": ["up", "right"]}`.

Reward types: `em` (exact set match), `prp` (precision-recall penalty),
`em_plus_prp` (combo). See `make_em_only_reward`, `make_pr_penalty_reward`,
`make_em_plus_prp_reward` in `env_2048_single_turn.py`.

### Output format
Same dual support as the multi-turn env:
- `"xml"` (default): `<move>up</move>` / `<valid_moves>...</valid_moves>`
- `"json"`: `{"move": "up"}` / `{"valid_moves": [...]}`

### Difficulty distribution (for enumerate task)
Train data fill ratio is uniform random per example; difficulty (number of
valid moves on the board) follows the natural play distribution. To rebalance:
```python
load_environment(
    task_type="enumerate_all",
    difficulty_distribution=(0.34, 0.33, 0.33),  # equalize 2/3/4-valid boards
)
```

### Scripts

#### Data generation
- `generate_sft_data.py` — Build SFT dataset of (board, expert_move) pairs.
  Used to produce `sft_data_json/` and `sft_data_xml/`.

#### Training
- `train_lora_sft.py` — Standalone HF TRL SFTTrainer + PEFT LoRA trainer for
  the single-turn task. Outputs both raw adapter and merged checkpoint.

#### Eval
- `eval_lora_vllm.py` — Fast vLLM eval for `pick_one` task. Reports parsed-rate,
  valid-rate, has_think rate, completion length.
- `eval_enumerate_vllm.py` — Same for `enumerate_all` task. Reports EM, F1,
  per-difficulty breakdown.
- `eval_lora_full.py` — Full HF-only eval (slower) used for sanity checks.
- `eval_ckpt.sh`, `eval_lora.sh` — Driver shells

#### Analysis
- `diagnose_collapse.py` — Probes a RL'd model for the "predict-N constant"
  failure: per-difficulty bucket, what's the model's prediction size distribution
  and how often is the prediction a valid subset of true valid moves.
- `inspect_lora_prp.py` — Inspection of the lora_prp RL'd checkpoint
  (the one that "collapsed" but actually had 100% precision).
- `probe_thinking.py`, `dump_thinking_samples.py` — Dump model thinking traces
  for qualitative inspection.

### Quickstart

#### Run a quick eval with `prime eval`
```bash
prime eval run env-2048-single-turn -m PrimeIntellect/Qwen3-0.6B -n 20 \
  -a '{"output_format": "json", "task_type": "pick_one", "grid_size": 4}'
```

#### Train SFT LoRA
```bash
cd /path/to/env_2048_single_turn
python generate_sft_data.py --output_dir sft_data_json --output_format json
python train_lora_sft.py \
    --data_dir sft_data_json \
    --output_dir lora_outputs_json \
    --model Qwen/Qwen3-0.6B --merge
```

### Configs
See [`../CONFIGS.md`](../CONFIGS.md) for a full breakdown of each TOML config:
which seed model, reward, hyperparameters, and result.

### Project log
See [`../EXPERIMENTS.md`](../EXPERIMENTS.md) for the full experimental
narrative (Exp 6-15 cover the single-turn work).

### Key findings
1. **LoRA SFT (rank 16+) preserves thinking**; full-FT SFT destroys it,
   even with `chat_template_kwargs={enable_thinking: False}` to mask empty
   `<think></think>` blocks from supervised tokens. See `EXPERIMENTS.md` Exp 6.
2. **PR_penalty reward** on the enumerate task hits 100% EM on the dominant
   2-valid bucket but collapses to "predict size=2 always" on 3/4-valid
   boards. The 2-direction predictions are 100% subset-of-valid (never
   hallucinates invalid moves) — the model just predicts the modal set size.
3. **Data distribution dominates over reward design**: balancing train data
   to 33/33/33 (vs the natural 75/12.5/12.5) breaks the predict-2 collapse
   but introduces hallucinations on the easy case. See Exp 15.
