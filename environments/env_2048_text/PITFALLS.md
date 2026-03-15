# Pitfalls in Building a Verifiers Multi-Turn RL Environment

This document captures issues encountered while building the 2048 text environment for RL training with the `verifiers` library and `prime-rl` framework.

---

## 1. Raw Dicts vs Message Objects (Zero Gradient Bug)

### Symptom
- RL training showed `grad_norm = 0.0` for ALL steps
- `loss = 0.0`, `entropy = NaN`
- No learning happening despite rollouts completing

### Root Cause
`get_prompt_messages()` returned raw Python dicts instead of Pydantic `Message` objects:

```python
# BROKEN:
return [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
]
```

### Why It Broke
1. The rollout loop stores whatever `get_prompt_messages()` returns into `state["trajectory"]`
2. Later, the token client reconstructs messages from trajectory for token-level processing
3. `to_native_prompt()` does `isinstance(message, SystemMessage)` checks
4. Raw dicts fail these checks → `ValueError("Invalid chat message")`
5. Error is caught → `state["error"]` is set
6. In trajectory processing: `has_error = True` → `completion_mask = [0, 0, 0, ...]`
7. Zero trainable tokens → zero loss → zero gradients

### Fix
Always wrap returns with `normalize_messages()`:

```python
# FIXED:
from verifiers.utils.message_utils import normalize_messages

return normalize_messages([
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
], field_name="prompt")
```

### Key Insight
The parent `MultiTurnEnv.get_prompt_messages()` always normalizes its returns. When you override this method, you must do the same.

---

## 2. History Not Accumulating in Non-FULL Context Modes

### Symptom
- `render_completion()` only showed the last turn instead of full conversation
- Logged completions were truncated, making debugging difficult

### Root Cause
In non-FULL context modes (MARKOV, LAST_K, SUMMARY), we return truncated prompts. The parent's `get_prompt_messages()` rebuilds history from `trajectory[-1]["prompt"]`, which is already truncated.

```python
# Parent's logic:
prev_turn_prompt = state["trajectory"][-1]["prompt"]  # Only [system, current_state] in markov!
# ... builds from this truncated base
```

### Fix
Manually maintain full history in a separate state field:

```python
async def get_prompt_messages(self, state: State) -> Messages:
    # Accumulate full history ourselves
    full_messages = state.get("_full_messages_for_logging", [])
    
    # Add last completion
    last_completion = normalize_messages(trajectory[-1]["completion"])
    full_messages = concat_messages([full_messages, last_completion])
    
    # Add env response
    env_response = await self.env_response(full_messages, state)
    full_messages = concat_messages([full_messages, env_response_messages])
    
    # Store for logging
    state["_full_messages_for_logging"] = list(full_messages)
    
    # Return truncated version for model (e.g., just current state for markov)
    if self.context_mode == ContextMode.MARKOV:
        return normalize_messages([...])  # truncated
```

Then in `render_completion()`, use `_full_messages_for_logging` instead of rebuilding from trajectory.

---

## 3. LAST_K Mode Using Placeholder Text

### Symptom
- LAST_K mode showed generic placeholder messages instead of actual game states
- History looked like `[system, "Previous turn 1", "Response 1", ...]`

### Root Cause
Initial implementation tried to rebuild history from trajectory, but trajectory only stored truncated prompts. The rebuild logic used placeholder text.

### Fix
Use the accumulated `_full_messages_for_logging` which contains actual env responses:

```python
def _build_last_k_messages(self, state: State, game: Game2048) -> Messages:
    full_messages = state.get("_full_messages_for_logging", [])
    
    # Extract last 2*k messages (k turns of user+assistant pairs)
    system_msg = full_messages[0]
    rest = full_messages[1:]
    
    if len(rest) <= 2 * k:
        return normalize_messages(full_messages)
    
    last_messages = rest[-(2 * k):]
    return normalize_messages([system_msg] + last_messages)
```

---

## 4. Debugging Zero Gradients - Investigation Process

When you see zero gradients in RL training, check these in order:

### Step 1: Check completion_mask
```python
# In a rollout file or debug output:
print(sum(sample["completion_mask"]))  # Should be > 0
```

If all zeros, tokens are being masked out.

### Step 2: Check for errors
```python
# In rollout data:
print(sample.get("error"))  # Should be None
```

Errors cause all tokens to be masked.

### Step 3: Check logs for "Aborted rollout"
```bash
grep -c "Aborted rollout" outputs/logs/train/*.log
```

If many aborted rollouts, find the error message:
```bash
grep "Aborted rollout" outputs/logs/train/*.log | head -5
```

### Step 4: Common error patterns
- `ValueError("Invalid chat message: {...}")` → Raw dicts instead of Message objects
- `OverlongPromptError` → Prompt too long, increase `seq_len`
- `ModelError` → Check the wrapped exception for details

---

## 5. env_response Returns Don't Need Normalization (Usually)

### Clarification
The `env_response()` method CAN return raw dicts because the parent's `get_prompt_messages()` normalizes them:

```python
# In parent's get_prompt_messages():
env_response = await self.env_response(messages, state)
env_response_messages = normalize_messages(env_response)  # Parent normalizes!
```

However, if you override `get_prompt_messages()` and call `env_response()` yourself, you should normalize:

```python
# In your override:
env_response = await self.env_response(full_messages, state)
env_response_messages = normalize_messages(env_response)  # You must normalize!
```

---

## 6. Context Mode Considerations

### FULL Mode
- Stores complete conversation history
- Memory grows with game length
- Best for debugging and analysis

### MARKOV Mode
- Only current state shown to model
- Constant memory footprint
- Model can't learn from history patterns
- Note: Move counter (`Moves: N`) leaks some temporal info

### LAST_K Mode
- Shows last K turns of history
- Balance between context and memory
- Must carefully maintain actual history for logging

### SUMMARY Mode
- Requires additional LLM call for summarization
- Adds latency and cost
- Summary quality affects learning

---

## 7. Testing Multi-Turn Environments

### Test Prompt Generation
```python
async def test_context_modes():
    env = Game2048Env(context_mode="markov")
    state = create_test_state()
    
    messages = await env.get_prompt_messages(state)
    
    # Verify messages are normalized
    from verifiers.types import Message
    for msg in messages:
        assert isinstance(msg, Message), f"Got {type(msg)}"
```

### Test History Accumulation
```python
async def test_history_grows():
    # Simulate multiple turns
    for turn in range(5):
        await env.get_prompt_messages(state)
        # Add mock completion to trajectory
        
    full_msgs = state.get("_full_messages_for_logging", [])
    # Should grow: 2 (initial) + 2*turns
    assert len(full_msgs) == 2 + 2 * 5
```

---

## 8. Configuration Pitfalls

### Eval Uses Different Config
Your training config might use one context mode, but eval configs can differ:

```toml
# Training
[[orchestrator.env]]
args = { context_mode = "markov" }

# Eval - might accidentally use different mode!
[[orchestrator.eval.env]]
args = { context_mode = "full" }  # Different!
```

### seq_len Must Accommodate Full Conversations
Even in MARKOV mode, if you're doing token-level training, the full conversation history affects tokenization:

```toml
seq_len = 4096  # Must fit: prompt + all completions + env responses
```

---

## 9. Logging and Debugging Tips

### Enable Verbose Logging
```python
import logging
logging.getLogger("verifiers").setLevel(logging.DEBUG)
```

### Check Rollout Files
```bash
# Find a rollout file
ls outputs/rollouts/

# Check completion_mask
python -c "
import json
with open('outputs/rollouts/step_0/0.jsonl') as f:
    for line in f:
        d = json.loads(line)
        mask_sum = sum(d.get('completion_mask', []))
        print(f'mask_sum={mask_sum}, error={d.get(\"error\")}')"
```

### Check WandB Metrics
Key metrics to watch:
- `grad_norm`: Should be > 0
- `loss`: Should be > 0  
- `entropy`: Should be finite (not NaN)
- `reward_mean`: Indicates rollout quality

---

## Summary Checklist

When building a multi-turn environment:

- [ ] `get_prompt_messages()` returns normalized `Message` objects
- [ ] Override maintains compatibility with parent's expectations
- [ ] History accumulation works correctly for logging
- [ ] `env_response()` returns are normalized when used directly
- [ ] Tests verify message types, not just content
- [ ] Eval configs match training mode expectations
- [ ] seq_len is sufficient for full conversations
