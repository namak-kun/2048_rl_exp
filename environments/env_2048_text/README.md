# env-2048-text

### Overview
- **Environment ID**: `env-2048-text`
- **Short description**: RL environment for training LLMs to play the 2048 puzzle game using ASCII text representation
- **Tags**: game, 2048, multi-turn, reasoning, train, eval

### Datasets
- **Primary dataset(s)**: Procedurally generated 2048 game instances
- **Source links**: N/A (generated on-the-fly)
- **Split sizes**: Configurable via `num_train_examples` and `num_eval_examples`

### Task
- **Type**: multi-turn
- **Output format expectations**: XML tags for move extraction (`<move>up/down/left/right</move>`) and optional reasoning (`<reasoning>...</reasoning>`)
- **Rubric overview**: 
  - `win_reward` (weight=0.4): Binary reward for reaching target tile
  - `max_tile_reward` (weight=0.3): Log-scaled reward based on highest tile achieved
  - `score_reward` (weight=0.2): Normalized game score
  - `valid_moves_ratio` (weight=0.1): Ratio of valid to invalid moves

### Game Rules
The 2048 game follows standard rules:
- NxN grid with numbered tiles (default 4x4, configurable)
- Tiles slide in one of four directions (up, down, left, right)
- When two tiles with the same value collide, they merge into one tile with double the value
- After each move, a new tile (2 or 4) appears in a random empty cell
- Goal: Create a tile with the target value (default 2048, configurable)
- Game ends when no more moves are possible

### Board Representation
The board is displayed as an ASCII table. Example 4x4 grid:
```
+----+----+----+----+
| .  | 2  | .  | .  |
+----+----+----+----+
| .  | 4  | .  | 2  |
+----+----+----+----+
| .  | .  | .  | .  |
+----+----+----+----+
| .  | .  | .  | .  |
+----+----+----+----+
```
Where `.` represents empty cells.

### Context Management Modes

Since 2048 is a **Markov game** (current state contains all information needed), the environment supports multiple context management strategies:

| Mode | Description | Use Case |
|------|-------------|----------|
| `full` | Full conversation history (default) | Standard multi-turn, good for learning patterns |
| `markov` | Current state only, no history | **Recommended for 2048** - forces pure state-based reasoning |
| `last_k` | Last K turns of history | Balance between context and efficiency |
| `summary` | LLM-generated summary + current state | Compressed strategic context with game analysis |

**Recommendation**: Use `context_mode="markov"` for 2048 since the game is fully observable and Markovian.

### Quickstart
Run an evaluation with default settings (4x4 grid, target 2048):

```bash
prime eval run env-2048-text
```

Configure model, grid size, target, and context mode:

```bash
# Standard with Markov context (recommended for 2048)
prime eval run env-2048-text \
  -m gpt-4.1-mini \
  -a '{"context_mode": "markov", "grid_size": 4, "target_tile": 2048}'

# With last 3 turns of history
prime eval run env-2048-text \
  -m gpt-4.1-mini \
  -a '{"context_mode": "last_k", "last_k_turns": 3}'

# With LLM-generated summary
prime eval run env-2048-text \
  -m gpt-4.1-mini \
  -a '{"context_mode": "summary", "summary_interval": 5}'

# Easier 3x3 game with lower target
prime eval run env-2048-text \
  -m gpt-4.1-mini \
  -a '{"grid_size": 3, "target_tile": 512, "max_moves": 200}'
```

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_train_examples` | int | `100` | Number of training game instances |
| `num_eval_examples` | int | `20` | Number of evaluation game instances |
| `max_moves` | int | `500` | Maximum moves allowed per game |
| `target_tile` | int | `2048` | Target tile value to win (e.g., 512, 1024, 2048, 4096) |
| `grid_size` | int | `4` | Size of the grid (e.g., 3 for 3x3, 4 for 4x4, 5 for 5x5) |
| `context_mode` | str | `"full"` | Context management: `"full"`, `"markov"`, `"last_k"`, or `"summary"` |
| `last_k_turns` | int | `3` | Number of turns to keep when `context_mode="last_k"` |
| `summary_interval` | int | `5` | Regenerate LLM summary every N turns (for `context_mode="summary"`) |
| `summary_model` | str | `None` | Model to use for summarization (defaults to main model) |
| `system_prompt` | str | (auto) | Custom system prompt (auto-generated if not provided) |
| `seed` | int | `42` | Random seed for reproducibility |

### Context Mode Details

#### `full` (default)
Sends the complete conversation history to the model. Each turn includes all previous messages.
- **Pros**: Model can learn from patterns across the game
- **Cons**: Context grows linearly, expensive for long games

#### `markov` (recommended for 2048)
Sends only the system prompt and current game state. No history.
- **Pros**: Minimal tokens, forces state-based reasoning, ideal for Markov games
- **Cons**: Model can't learn from past mistakes within a game

#### `last_k`
Keeps the last K turns of conversation history.
- **Pros**: Bounded context, some history for pattern recognition
- **Cons**: May miss important earlier context

#### `summary` (LLM-generated)
Periodically calls the LLM to generate a strategic summary of game history, then includes it with the current state.
- **Pros**: Compressed but informative context, includes strategic analysis
- **Cons**: Additional API calls for summarization

The summary is regenerated every `summary_interval` turns (default: 5). You can use a cheaper model for summaries via `summary_model`.

Example summary prompt sent to model:
```
Game Summary:
<LLM-generated analysis of moves, patterns, and strategy>

Current State:
Score: 1234
Moves: 45
Max Tile: 512

+----+----+----+----+
| 2  | 4  | 8  | 2  |
...

What's your move?
```

### Difficulty Levels
You can adjust difficulty by changing grid size and target tile:

| Difficulty | Grid Size | Target | Typical Moves |
| ---------- | --------- | ------ | ------------- |
| Easy | 3x3 | 256-512 | 50-150 |
| Medium | 4x4 | 1024 | 200-400 |
| Standard | 4x4 | 2048 | 400-800 |
| Hard | 4x4 | 4096 | 800-1500 |
| Expert | 5x5 | 4096+ | 1000-2000 |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward (weighted sum of win, max_tile, score, valid_moves) |
| `win_reward` | 1.0 if target tile reached, 0.0 otherwise |
| `max_tile_reward` | Log-scaled progress toward target (0.0 to 1.0) |
| `score_reward` | Normalized game score (scaled by target tile) |
| `valid_moves_ratio` | Proportion of valid moves out of all attempts |
| `efficiency_reward` | For wins only: bonus for fewer moves |
| `num_turns` | Total number of turns in the game |

### Example Interaction

#### With `context_mode="markov"` (current state only)
Each turn the model only sees:
```
[System prompt explaining rules]

Score: 156
Moves: 12
Max Tile: 64

+----+----+----+----+
| 2  | 4  | 8  | 2  |
+----+----+----+----+
| 4  | 16 | 32 | 4  |
+----+----+----+----+
| 2  | 8  | 64 | 8  |
+----+----+----+----+
| .  | 2  | 4  | 2  |
+----+----+----+----+

What's your move? Remember to put your move inside <move>...</move> tags.
```

#### Assistant Response
```
<reasoning>The 64 tile is in the third row. I should keep building toward the bottom-right corner. Moving right will merge the 4s and keep tiles organized.</reasoning>
<move>right</move>
```

copilot --resume=81c1da82-cd8b-4a8c-91d6-b6e50c83288f