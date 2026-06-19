# 2048 RL Experiments Log

> Running document. Newest at the top.
> Maintained by Copilot for human reference. Edit freely.

---

## TL;DR (current state)

- **Stage 1 (no-think SFT)** is **solved**: both LoRA models hit 99-100% valid moves on hard balanced boards across all grid sizes.
- **Thinking preservation is LoRA-specific, NOT JSON-specific**. Full SFT in either format destroys thinking (0.000 on think conditions). LoRA in either format preserves thinking (XML: 35-52%, JSON: 60-65%). JSON does help further within LoRA.
- The `chat_template_kwargs={"enable_thinking": False}` trick is necessary but NOT sufficient — full FT still kills thinking, only LoRA preserves it.
- **Verified parity** between custom vLLM eval and vf-eval (the verifiers framework prime-rl will use). Numbers within sampling variance.
- Best checkpoint so far: `lora_outputs_json/merged/` — drop-in for vLLM/prime-rl.

## Open questions / things to test
- [ ] Why does full SFT XML get only 0.778 on no-think 4x4 (vs 0.988 for LoRA-XML, 1.000 for full SFT JSON)? Format-specific learning difficulty under full FT?
- [ ] Stage 2 RL on LoRA-JSON merged model — does the preserved thinking + valid-move skill help RL stay on-policy and avoid collapse?
- [ ] Does a larger LoRA rank (32, 64) help thinking preservation further?
- [ ] Could partial LoRA (e.g., only MLP modules, not attention) preserve thinking even better?
- [ ] What does "average response length" look like across conditions in vf-eval?
- [ ] Should we bake JSON format as the default in the env (XML is back-compat default)?

---

## Experiments

### Exp 06 — Full SFT XML vs JSON (with chat_template masking) (✅ done — 2026-06-03 late evening)
- **Date**: 2026-06-03 ~23:00–01:30
- **Goal**: Test if the XML-vs-JSON thinking-preservation gap holds for full FT, OR if it's a LoRA-only effect
- **Setup**: Same 10k datasets and `chat_template_kwargs={"enable_thinking": False}` masking as LoRA runs, but full FT via prime-rl (200 steps, lr 2e-5, batch 64, 2 GPU FSDP)
- **vf-eval results (n=150 per condition, balanced difficulty)**:

  | Model | 4x4 nothink | 4x4 think | 5x5 nothink | 5x5 think | 6x6 nothink | 6x6 think |
  |---|---|---|---|---|---|---|
  | Full SFT XML  | 0.778 | **0.000** | 0.868 | **0.000** | 0.958 | **0.000** |
  | Full SFT JSON | 1.000 | **0.000** | 1.000 | **0.000** | 1.000 | **0.000** |
- **Headline 1**: Full SFT kills thinking ENTIRELY in both formats (0.000 on every think condition).
- **Headline 2**: Full SFT JSON > Full SFT XML on no-think (1.000 vs 0.778-0.958). XML may be harder to learn under full FT for some reason — possibly because the JSON format is simpler/more atomic for the model.
- **Conclusion**: The thinking-preservation effect of JSON observed in Exp 04 was actually a LoRA-specific phenomenon. The XML-vs-JSON gap within LoRA exists but is small. The DOMINANT factor for thinking preservation is **using LoRA**, not the format choice.

### Exp 05 — vf-eval parity check (✅ done)
- **Goal**: Confirm custom vLLM evaluator results match the verifiers/prime-rl eval framework
- **Setup**: Same checkpoints as Exp 04, eval via `vf-eval` against a vLLM OpenAI server with `--reasoning-parser qwen3`
- **vf-eval results (n=150 per condition, balanced difficulty)**:

  | Model | 4x4 nothink | 4x4 think | 5x5 nothink | 5x5 think | 6x6 nothink | 6x6 think |
  |---|---|---|---|---|---|---|
  | Base XML    | 0.604 | 0.499 | 0.628 | 0.451 | 0.688 | 0.464 |
  | Base JSON   | 0.622 | 0.505 | 0.628 | 0.433 | 0.610 | 0.511 |
  | LoRA XML    | 0.988 | 0.351 | 1.000 | 0.515 | 0.994 | 0.419 |
  | LoRA JSON   | 1.000 | 0.649 | 0.994 | 0.639 | 0.994 | 0.605 |
- Numbers match the custom vLLM eval within sampling variance.
- Note: vf-eval reward = 1.0 for valid move, 0.1 for parseable-but-no-tile-change, 0.0 for unparseable. That's why "base" think conditions are 0.4-0.5 (mix of valid + parseable-but-invalid).

### Exp 04 — LoRA SFT, XML vs JSON (✅ done — 2026-06-03 morning)
- **Goal**: Train a LoRA on the valid-move task that preserves Qwen3's thinking
- **Setup**:
  - HF transformers + PEFT + TRL `SFTTrainer` (standalone, prime-rl's SFT can't use LoRA without `MultiRunManager`)
  - LoRA rank 16, alpha 32 on q/k/v/o/gate/up/down
  - 3 epochs, batch 8 × grad_accum 4, lr 2e-4
  - 10k examples per dataset, mixed 4×4/5×5/6×6 grids
  - **Key trick**: `chat_template_kwargs={"enable_thinking": False}` per example so TRL's loss mask excludes the empty `<think></think>` block
- **Result (custom vLLM eval, n=100 per condition, hard balanced; see Exp 05 for matching vf-eval numbers)**:
  - 4x4 no-think LoRA-XML: 100%; LoRA-JSON: 100%
  - 4x4 think LoRA-XML: 42%; LoRA-JSON: 71%
  - JSON preserves thinking content at 77% vs XML's 48%
- **Headline (revised after Exp 06)**: LoRA preserves thinking via parameter-efficient adaptation. JSON provides additional headroom within LoRA but isn't the primary driver.

### Exp 03 — Full SFT XML (no-think, no chat_template masking) (✅ done earlier; reference)
- 200 steps full FT on 10k XML examples (no `chat_template_kwargs` field — empty `<think></think>` block included in supervised tokens)
- Result: 100% valid moves on test, but `<think>` block replaced by `<well>` garbage — full catastrophic forgetting and TEMPLATE corruption
- Conclusion that motivated Exp 04: need LoRA + proper chat-template masking
- Note: Exp 06 added masking + still saw full thinking destruction, so masking alone is insufficient

### Exp 02 — RL Stage 2 from base Qwen3 + thinking (❌ failed)
- Config: base Qwen3, `reasoning_parser=qwen3`, `enable_thinking=true`
- Result: collapsed to "always up" by step 10; thinking format preserved but vacuous
- Root cause: single-turn valid-move task lets any constant policy score ~58% on balanced data → no incentive to reason

### Exp 01 — Single-turn env + playthrough data generation (✅ done earlier)
- Built `generate_playthrough_states()`: start partially-filled, play random moves, snapshot
- Produces balanced 75%/12.5%/12.5% (2/3/4-valid) data on ALL grid sizes (4×4, 5×5, 6×6)
- Random baseline 59.4% on all sizes

---

## Comparison matrix (vf-eval, n=150, 4x4 hard-balanced)

| Model | no-think | think |
|---|---|---|
| Base XML | 0.604 | 0.499 |
| Base JSON | 0.622 | 0.505 |
| **Full SFT XML** | 0.778 | **0.000** |
| **Full SFT JSON** | 1.000 | **0.000** |
| **LoRA XML** | 0.988 | 0.351 |
| **LoRA JSON** | **1.000** | **0.649** |

## Key technical decisions (durable knowledge)

- **Qwen3 chat template** auto-injects `<think>\n\n</think>` before the last assistant turn no matter what `enable_thinking` is set to. To prevent SFT from training on that empty block, pass `chat_template_kwargs={"enable_thinking": False}` per example so the prompt and the prompt+completion render with the same prefix → only the answer ends up in `completion_mask`.
- **Even with the masking trick, full FT destroys thinking.** Use LoRA (rank 16+) if thinking preservation matters.
- **prime-rl's SFT trainer cannot use LoRA** without the `MultiRunManager` being initialized (only done in the RL pipeline). Use HF TRL + PEFT in a standalone script. `merge_and_unload()` produces a regular HF checkpoint that vLLM and prime-rl can load.
- **prime-rl SFT data loading** uses `load_dataset`, not `load_from_disk`. Save your data as parquet (not via `save_to_disk`) for prime-rl compatibility.
- **1-valid-move 2048 states are impossible**. A merge always enables the reverse direction; empty cells are reachable from 2+ directions. Hard floor is 2 valid moves.
- **Use vLLM offline engine for eval** — 10-50× faster than HuggingFace `model.generate` thanks to continuous batching. For verifiers-parity eval, use `vf-eval` against a vLLM OpenAI server with `--reasoning-parser qwen3`.
- **vf-eval can hang/timeout** when running in parallel (env servers use ports that may collide). For parallel runs, spawn each in its own session via `setsid` and use distinct ports.
- **nohup alone doesn't fully detach.** Use `setsid nohup ... < /dev/null` and `disown` for processes that need to survive parent shell exits, especially when child processes use `trap cleanup EXIT`.
- **Sharing GPUs with other users**: with ~28GB occupied per A6000 by another user, our vLLM server must use `--gpu-memory-utilization 0.34` to fit alongside.

## File map
- Env: `environments/env_2048_single_turn/env_2048_single_turn.py` (supports `output_format=xml|json`)
- Data gen: `generate_sft_data.py` (jsonl → `sft_data_{xml,json}/train.jsonl`)
- LoRA SFT: `train_lora_sft.py` (standalone TRL+PEFT, includes sanity check)
- Configs: `configs/{sft_xml_full,sft_json_full}.toml` (prime-rl full SFT)
- Fast eval: `eval_lora_vllm.py` (vLLM offline; used in Exp 04)
- VF-parity eval: `eval_lora.sh` (vf-eval against vLLM server; used in Exp 05/06)
- Probe: `probe_thinking.py` (qualitative HF-based; useful for inspection)
- Checkpoints:
  - LoRA adapters/merged: `lora_outputs_{xml,json}/{adapter,merged}/`
  - Full SFT: `sft_outputs_{xml,json}_full/weights/step_200/`
- Eval logs: `logs/vfeval_*.log` + `eval_results/*.json` (custom eval only)
- Datasets (jsonl): `sft_data_{xml,json}/train.jsonl`
- Datasets (parquet for prime-rl): `sft_data_{xml,json}_parquet/train.parquet`
