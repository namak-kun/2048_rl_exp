"""
2048 Single-Turn Environment for LLM Training

A simple single-turn environment where the LLM sees a game state and makes one move.
Used for curriculum stage 1: learning valid move format and basic game mechanics.
"""

import math
import random
import re
from typing import Literal, Optional

from datasets import Dataset

import verifiers as vf
from verifiers.types import State


# =============================================================================
# Game Logic (simplified from main env)
# =============================================================================

class Grid:
    """NxN grid for the 2048 game."""

    def __init__(self, size: int = 4, cells: Optional[list] = None):
        self.size = size
        if cells is not None:
            self.cells = [row[:] for row in cells]
        else:
            self.cells = [[0] * size for _ in range(size)]

    def copy(self):
        return Grid(self.size, self.cells)

    def available_cells(self):
        return [(i, j) for i in range(self.size) for j in range(self.size) if self.cells[i][j] == 0]

    def random_available_cell(self):
        available = self.available_cells()
        return random.choice(available) if available else None

    def insert_tile(self, pos, value):
        self.cells[pos[0]][pos[1]] = value

    @property
    def max_value(self):
        return max(max(row) for row in self.cells)


class Game2048:
    """2048 game logic."""

    def __init__(self, size: int = 4, target_tile: int = 2048, grid: Optional[Grid] = None):
        self.size = size
        self.target_tile = target_tile
        self.grid = grid if grid else Grid(size)
        self.score = 0
        self.over = False
        self.won = False

        if grid is None:
            self.add_random_tile()
            self.add_random_tile()

    def add_random_tile(self):
        cell = self.grid.random_available_cell()
        if cell:
            value = 4 if random.random() < 0.1 else 2
            self.grid.insert_tile(cell, value)

    def move(self, direction: int) -> bool:
        """Execute move. Returns True if board changed."""
        old_cells = [row[:] for row in self.grid.cells]
        
        if direction == 0:    # up
            self._move_up()
        elif direction == 1:  # right
            self._move_right()
        elif direction == 2:  # down
            self._move_down()
        elif direction == 3:  # left
            self._move_left()

        moved = self.grid.cells != old_cells
        if moved:
            self.add_random_tile()
            if self.grid.max_value >= self.target_tile:
                self.won = True
            if not self._moves_available():
                self.over = True
        return moved

    def _slide_row_left(self, row):
        """Slide and merge a row to the left."""
        non_zero = [x for x in row if x != 0]
        merged = []
        skip = False
        for i, val in enumerate(non_zero):
            if skip:
                skip = False
                continue
            if i + 1 < len(non_zero) and non_zero[i + 1] == val:
                merged.append(val * 2)
                self.score += val * 2
                skip = True
            else:
                merged.append(val)
        return merged + [0] * (len(row) - len(merged))

    def _move_left(self):
        self.grid.cells = [self._slide_row_left(row) for row in self.grid.cells]

    def _move_right(self):
        self.grid.cells = [self._slide_row_left(row[::-1])[::-1] for row in self.grid.cells]

    def _move_up(self):
        self.grid.cells = list(map(list, zip(*[self._slide_row_left(list(col)) for col in zip(*self.grid.cells)])))

    def _move_down(self):
        self.grid.cells = list(map(list, zip(*[self._slide_row_left(list(col)[::-1])[::-1] for col in zip(*self.grid.cells)])))

    def _moves_available(self) -> bool:
        if self.grid.available_cells():
            return True
        for i in range(self.size):
            for j in range(self.size):
                val = self.grid.cells[i][j]
                if j + 1 < self.size and self.grid.cells[i][j + 1] == val:
                    return True
                if i + 1 < self.size and self.grid.cells[i + 1][j] == val:
                    return True
        return False

    @property
    def max_tile(self):
        return self.grid.max_value

    def get_state_text(self) -> str:
        """Render board as ASCII table."""
        max_val = self.grid.max_value
        cell_width = max(4, len(str(max_val)) + 2)
        
        lines = []
        horizontal = "+" + (("-" * cell_width + "+") * self.size)
        
        for row in self.grid.cells:
            lines.append(horizontal)
            cells = []
            for val in row:
                if val == 0:
                    cells.append(" " * cell_width)
                else:
                    cells.append(str(val).center(cell_width))
            lines.append("|" + "|".join(cells) + "|")
        lines.append(horizontal)
        
        return "\n".join(lines)

    @staticmethod
    def parse_move(move_str: str) -> Optional[int]:
        """Parse move string to direction int."""
        move_map = {
            "up": 0, "u": 0, "north": 0, "n": 0,
            "right": 1, "r": 1, "east": 1, "e": 1,
            "down": 2, "d": 2, "south": 2, "s": 2,
            "left": 3, "l": 3, "west": 3, "w": 3,
        }
        return move_map.get(move_str.lower().strip())


# =============================================================================
# Random State Generation
# =============================================================================

def generate_random_game_state(
    size: int = 4,
    target_tile: int = 2048,
    min_moves: int = 0,
    max_moves: int = 100,
) -> Game2048:
    """
    Generate a random mid-game state by playing random moves from start.
    
    This guarantees:
    - Realistic board configurations
    - At least one valid move exists (unless game over, which we retry)
    - Varying board fullness naturally
    """
    for _ in range(100):  # retry if we hit game over
        game = Game2048(size=size, target_tile=target_tile)
        
        # Play random number of moves
        num_moves = random.randint(min_moves, max_moves)
        directions = [0, 1, 2, 3]  # up, right, down, left
        
        for _ in range(num_moves):
            if game.over or game.won:
                break
            # Try random moves until one works
            random.shuffle(directions)
            for d in directions:
                if game.move(d):
                    break
        
        # Verify game isn't over (has valid moves)
        if not game.over and game._moves_available():
            return game
    
    # Fallback: return fresh game
    return Game2048(size=size, target_tile=target_tile)


def generate_dense_random_state(
    size: int = 4,
    target_tile: int = 2048,
    fill_ratio: float = 0.8,
    max_tile_power: int = 8,
) -> Game2048:
    """
    Generate a random board state with specified fill ratio.
    
    Does NOT simulate gameplay - just places random tiles.
    Good for generating hard states where not all moves work.
    
    Args:
        size: Board size
        target_tile: Target tile for win condition
        fill_ratio: Fraction of cells to fill (0.0 to 1.0)
        max_tile_power: Max power of 2 for tiles (8 = up to 256)
    """
    cells = [[0] * size for _ in range(size)]
    num_tiles = int(size * size * fill_ratio)
    
    # Random positions to fill
    positions = random.sample(range(size * size), num_tiles)
    for pos in positions:
        r, c = pos // size, pos % size
        # Random power of 2 (2, 4, 8, ... up to 2^max_tile_power)
        cells[r][c] = 2 ** random.randint(1, max_tile_power)
    
    grid = Grid(size=size, cells=cells)
    game = Game2048(size=size, target_tile=target_tile, grid=grid)
    return game



DIRECTION_NAMES = ("up", "right", "down", "left")
DIRECTION_TO_INT = {name: i for i, name in enumerate(DIRECTION_NAMES)}


def get_valid_directions(game: Game2048) -> list[int]:
    """Return list of valid direction ints (0..3) for a game state."""
    valid = []
    for direction in range(4):
        test_game = Game2048(
            size=game.size,
            target_tile=game.target_tile,
            grid=Grid(size=game.size, cells=[row[:] for row in game.grid.cells]),
        )
        if test_game.move(direction):
            valid.append(direction)
    return valid


def get_valid_direction_names(game: Game2048) -> list[str]:
    return [DIRECTION_NAMES[d] for d in get_valid_directions(game)]


def count_valid_moves(game: Game2048) -> int:
    """Count how many directions result in a valid move."""
    return len(get_valid_directions(game))


def generate_playthrough_states(
    size: int = 4,
    target_tile: int = 2048,
    fill_ratio: float = 0.7,
    max_tile_power: int = 10,
    snapshot_interval: int = 3,
) -> list[Game2048]:
    """
    Generate realistic board states by starting from a partially-filled board
    and playing random valid moves, snapshotting along the way.
    
    This produces a range of difficulties naturally:
    - Early snapshots: more empty space, more valid moves (easier)
    - Later snapshots: board fills up, fewer valid moves (harder)
    
    Args:
        size: Board size
        target_tile: Target tile for win condition
        fill_ratio: Initial fill ratio (0.0 to 1.0)
        max_tile_power: Max power of 2 for initial tiles (10 = up to 1024)
        snapshot_interval: Take a snapshot every N moves
    
    Returns:
        List of Game2048 states sampled from the playthrough
    """
    # Step 1: Create initial board
    game = generate_dense_random_state(
        size=size,
        target_tile=target_tile,
        fill_ratio=fill_ratio,
        max_tile_power=max_tile_power,
    )
    
    # Step 2: Play random valid moves, snapshotting along the way
    snapshots = []
    directions = [0, 1, 2, 3]
    move_count = 0
    
    while not game.over and not game.won:
        nv = count_valid_moves(game)
        if nv == 0:
            break
        
        # Snapshot current state
        if move_count % snapshot_interval == 0:
            snapshot = Game2048(
                size=size,
                target_tile=target_tile,
                grid=game.grid.copy(),
            )
            snapshot.score = game.score
            snapshots.append(snapshot)
        
        # Play a random valid move
        random.shuffle(directions)
        for d in directions:
            if game.move(d):
                break
        move_count += 1
    
    return snapshots


def generate_single_turn_dataset(
    num_examples: int,
    size: int = 4,
    target_tile: int = 2048,
    min_moves: int = 0,
    max_moves: int = 100,
    seed: int = 42,
    balanced_difficulty: bool = False,
    output_format: str = "xml",
    task_type: str = "pick_one",
    difficulty_distribution: tuple[float, float, float] = (0.75, 0.125, 0.125),
) -> Dataset:
    """
    Generate dataset of random game states for single-turn training.
    
    Uses playthrough-based generation: starts from partially-filled boards
    with varying fill ratios, plays random moves, and snapshots states along
    the way. This naturally produces a range of difficulties for any grid size.
    
    Args:
        num_examples: Number of examples to generate
        size: Board size
        target_tile: Target tile for win condition
        min_moves: (unused, kept for config compatibility)
        max_moves: (unused, kept for config compatibility)
        seed: Random seed
        balanced_difficulty: If True, ensures even distribution across 
                           difficulty levels (2/3/4 valid moves) using
                           difficulty_distribution.
        output_format: "xml" or "json" — controls the user prompt instruction
        task_type: "pick_one" (default) or "enumerate_all"
        difficulty_distribution: (frac_2valid, frac_3valid, frac_4valid). Must sum to 1.
                                 Default (0.75, 0.125, 0.125). For uniform: (0.34, 0.33, 0.33).
    """
    if task_type not in ("pick_one", "enumerate_all"):
        raise ValueError(f"task_type must be 'pick_one' or 'enumerate_all', got {task_type!r}")
    if task_type == "enumerate_all" and output_format != "json":
        raise ValueError("task_type=enumerate_all requires output_format='json'")
    if balanced_difficulty:
        if len(difficulty_distribution) != 3 or abs(sum(difficulty_distribution) - 1.0) > 1e-6:
            raise ValueError(
                f"difficulty_distribution must be 3 floats summing to 1, got {difficulty_distribution!r}"
            )

    random.seed(seed)

    def make_example(game: "Game2048", nv: int) -> dict:
        prompt_content = get_user_prompt(game, output_format=output_format, task_type=task_type)
        valid_dirs = get_valid_direction_names(game)
        info = {
            "grid": [row[:] for row in game.grid.cells],
            "score": game.score,
            "size": size,
            "target_tile": target_tile,
            "valid_move_count": nv,
            "valid_directions": valid_dirs,
        }
        return {
            "prompt": [{"role": "user", "content": prompt_content}],
            "info": info,
        }

    examples = []
    
    if balanced_difficulty:
        # Collect states into difficulty buckets
        buckets = {2: [], 3: [], 4: []}
        f2, f3, f4 = difficulty_distribution
        target_counts = {
            2: int(num_examples * f2),
            3: int(num_examples * f3),
            4: num_examples - int(num_examples * f2) - int(num_examples * f3),
        }
        
        max_iters = num_examples * 50
        iters = 0
        
        while iters < max_iters:
            # Check if all buckets are full
            if all(len(buckets[k]) >= target_counts[k] for k in buckets):
                break
            
            iters += 1
            
            # Vary fill ratio to get diverse boards
            fill_ratio = random.uniform(0.4, 0.95)
            states = generate_playthrough_states(
                size=size,
                target_tile=target_tile,
                fill_ratio=fill_ratio,
                snapshot_interval=2,
            )
            
            for game in states:
                nv = count_valid_moves(game)
                if nv < 2 or nv > 4:
                    continue
                if len(buckets[nv]) >= target_counts[nv]:
                    continue
                buckets[nv].append(game)
        
        # Build examples from buckets
        for nv in [2, 3, 4]:
            for game in buckets[nv][:target_counts[nv]]:
                examples.append(make_example(game, nv))

        random.shuffle(examples)
    else:
        # Unbalanced: just generate diverse states
        while len(examples) < num_examples:
            fill_ratio = random.uniform(0.3, 0.95)
            states = generate_playthrough_states(
                size=size,
                target_tile=target_tile,
                fill_ratio=fill_ratio,
                snapshot_interval=3,
            )

            for game in states:
                if len(examples) >= num_examples:
                    break
                nv = count_valid_moves(game)
                if nv == 0:
                    continue

                examples.append(make_example(game, nv))

    return Dataset.from_list(examples)


# =============================================================================
# System Prompt
# =============================================================================

_RULES_BLOCK = """## Rules:
- The board is a {grid_size}x{grid_size} grid
- Tiles slide as far as possible in the chosen direction
- When two tiles with the same value collide, they merge into one tile with double the value
- After each move, a new tile (2 or 4) appears in a random empty cell
- The game ends when no more moves are possible

Valid moves are: up, down, left, right.
up slides all tiles to the top, down slides to the bottom, left slides to the left, and right slides to the right."""


def get_system_prompt(
    grid_size: int = 4,
    target_tile: int = 2048,
    output_format: str = "xml",
    task_type: str = "pick_one",
) -> str:
    rules = _RULES_BLOCK.format(grid_size=grid_size)
    intro = (
        f"You are playing the 2048 puzzle game. Your goal is to combine tiles by sliding them "
        f"in one of four directions (up, down, left, right) to create a tile with the value {target_tile}."
    )

    if task_type == "pick_one":
        if output_format == "xml":
            format_block = """## How to Play:
Look at the current game state and choose your next move. Respond with your move inside <move>...</move> tags.

Example response:
<move>up</move>"""
        elif output_format == "json":
            format_block = """## How to Play:
Look at the current game state and choose your next move. Respond with a JSON object containing a single key "move" whose value is the direction.

Example response:
{"move": "up"}"""
        else:
            raise ValueError(f"Unknown output_format: {output_format!r}. Expected 'xml' or 'json'.")
    elif task_type == "enumerate_all":
        if output_format != "json":
            raise ValueError("task_type='enumerate_all' requires output_format='json'")
        format_block = """## Task:
Look at the current game state and determine ALL directions that result in a valid move (a move that changes the board). A direction is valid if at least one tile would slide or merge in that direction.

Respond with a JSON object containing a single key "valid_moves" whose value is the list of all valid directions (any subset of ["up", "down", "left", "right"]).

Example response (when both up and left are valid):
{"valid_moves": ["up", "left"]}

The order of directions in the list does not matter, but the set must exactly match the truly valid directions for this board."""
    else:
        raise ValueError(f"Unknown task_type: {task_type!r}. Expected 'pick_one' or 'enumerate_all'.")

    return f"{intro}\n\n{rules}\n\n{format_block}\n"


def get_user_prompt(game: "Game2048", output_format: str = "xml", task_type: str = "pick_one") -> str:
    if task_type == "pick_one":
        if output_format == "xml":
            instr = "What's your move? Respond with <move>direction</move>."
        else:
            instr = 'What\'s your move? Respond with {"move": "direction"}.'
    else:
        instr = 'Which directions are valid moves on this board? Respond with {"valid_moves": [...]}.'
    return f"""Current board:
{game.get_state_text()}

Score: {game.score}
Max tile: {game.max_tile}

{instr}"""


# =============================================================================
# Reward Functions
# =============================================================================

def _extract_move_xml(text: str) -> Optional[str]:
    match = re.search(r"<move>\s*(.*?)\s*</move>", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def _extract_move_json(text: str) -> Optional[str]:
    # Find the first balanced { ... } block and try to parse it.
    # Fallback to a regex on a "move" field if json parsing fails (e.g., model wrapped in markdown).
    import json as _json

    # Strip common code fences
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    candidates = []
    if fenced:
        candidates.append(fenced.group(1))
    brace = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if brace:
        candidates.append(brace.group(0))

    for cand in candidates:
        try:
            data = _json.loads(cand)
        except ValueError:
            continue
        if isinstance(data, dict) and "move" in data and isinstance(data["move"], str):
            return data["move"].strip()

    # Last resort: regex-extract the value of a "move" key
    kv = re.search(r'"move"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    if kv:
        return kv.group(1).strip()
    return None


def make_valid_move_reward(output_format: str = "xml"):
    """Build a reward function bound to the given output format."""

    extract = _extract_move_xml if output_format == "xml" else _extract_move_json

    def valid_move_reward(state: State, **kwargs) -> float:
        """
        Reward for making a valid move.

        Returns:
            1.0 if move was valid (correct format + actually moved tiles)
            0.1 if format was correct but move didn't change board
            0.0 if format was wrong or unparseable
        """
        game: Game2048 = state.get("game")
        if game is None:
            return 0.0

        completion = state.get("completion", [])

        completion_text = ""
        for msg in completion:
            if hasattr(msg, "content"):
                completion_text = msg.content or ""
            elif isinstance(msg, dict):
                completion_text = msg.get("content", "") or ""

        move_str = extract(completion_text)
        if move_str is None:
            return 0.0

        direction = Game2048.parse_move(move_str)
        if direction is None:
            return 0.0

        game_copy = Game2048(size=game.size, target_tile=game.target_tile, grid=game.grid.copy())
        game_copy.score = game.score
        moved = game_copy.move(direction)

        return 1.0 if moved else 0.1

    valid_move_reward.__name__ = f"valid_move_reward_{output_format}"
    return valid_move_reward


# ---------------------------------------------------------------------------
# Enumerate-valid-moves task
# ---------------------------------------------------------------------------

_VALID_DIR_NAMES = set(DIRECTION_NAMES)


def _extract_valid_moves_set(text: str) -> Optional[frozenset]:
    """Extract a set of direction names from a model response.

    Returns frozenset of canonical lowercase direction names, or None if no
    parseable JSON list found. Unknown entries are dropped but presence of
    only-unknown entries still yields an empty set (not None) so we
    distinguish "model produced JSON but with garbage" from "no JSON at all".
    """
    import json as _json

    def _coerce(item) -> Optional[str]:
        if not isinstance(item, str):
            return None
        s = item.strip().lower()
        if s in _VALID_DIR_NAMES:
            return s
        # short aliases u/d/l/r
        alias = {"u": "up", "d": "down", "l": "left", "r": "right"}
        return alias.get(s)

    # Try fenced JSON, then any balanced-ish JSON object
    candidates = []
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1))
    # naive object grab — won't handle nested braces but we don't need that here
    for m in re.finditer(r"\{[^{}]*\}", text, re.DOTALL):
        candidates.append(m.group(0))

    for cand in candidates:
        try:
            data = _json.loads(cand)
        except ValueError:
            continue
        if not isinstance(data, dict):
            continue
        for key in ("valid_moves", "moves", "valid"):
            if key in data and isinstance(data[key], list):
                items = [_coerce(x) for x in data[key]]
                items = [x for x in items if x is not None]
                return frozenset(items)

    # Fallback: regex over a list literal after "valid_moves"
    lm = re.search(r'"valid_moves"\s*:\s*\[([^\]]*)\]', text, re.IGNORECASE | re.DOTALL)
    if lm:
        raw_items = re.findall(r'"([^"]+)"', lm.group(1))
        items = [_coerce(x) for x in raw_items]
        items = [x for x in items if x is not None]
        return frozenset(items)

    return None


def _completion_text(state: State) -> str:
    """Get the assistant answer text from a state's completion.

    With Qwen3's reasoning parser, the post-think answer lives in `content`
    and the think block in `reasoning_content`. We return the last assistant
    message's content; if that's None (model never closed </think>), we fall
    back to the reasoning_content so the parser can at least try to extract
    a JSON object from there.
    """
    last_content = ""
    for msg in state.get("completion", []):
        content = msg.content if hasattr(msg, "content") else (msg.get("content") if isinstance(msg, dict) else None)
        reasoning = (
            msg.reasoning_content
            if hasattr(msg, "reasoning_content")
            else (msg.get("reasoning_content") if isinstance(msg, dict) else None)
        )
        last_content = content if content else (reasoning or "")
    return last_content or ""


def _f1(pred: frozenset, true: frozenset) -> float:
    if not pred and not true:
        return 1.0
    if not pred or not true:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    precision = tp / len(pred)
    recall = tp / len(true)
    return 2 * precision * recall / (precision + recall)


def get_true_valid_set(state: State) -> frozenset:
    """Pull the true valid-moves set from state, handling both live State and serialized RolloutOutput.

    Live State has `state['input']['info']`; RolloutOutput has `state['info']` directly.
    """
    info = state.get("info")
    if not info:
        info = state.get("input", {}).get("info", {})
    if isinstance(info, str):
        import json as _json
        info = _json.loads(info)
    true_dirs = info.get("valid_directions")
    if true_dirs is None:
        game: Game2048 = state.get("game")
        if game is not None:
            true_dirs = get_valid_direction_names(game)
        else:
            true_dirs = []
    return frozenset(true_dirs)


def make_enumerate_reward(em_weight: float = 0.7, f1_weight: float = 0.3):
    """Build a reward function for the enumerate-valid-moves task.

    Reward = em_weight * EM + f1_weight * F1, both in [0, 1].
    Returns 0.0 if response is unparseable.
    """

    def enumerate_valid_moves_reward(state: State, **kwargs) -> float:
        true_set = get_true_valid_set(state)
        pred_set = _extract_valid_moves_set(_completion_text(state))
        if pred_set is None:
            return 0.0
        em = 1.0 if pred_set == true_set else 0.0
        f1 = _f1(pred_set, true_set)
        return em_weight * em + f1_weight * f1

    enumerate_valid_moves_reward.__name__ = (
        f"enumerate_valid_moves_reward_em{em_weight:g}_f1{f1_weight:g}"
    )
    return enumerate_valid_moves_reward


def make_em_plus_prp_reward(fp_penalty: float = 2.0):
    """EM + PR_penalty combined.

    Reward = (1.0 if exact else 0.0) + recall * max(0, 1 - fp_penalty * FP/4)
    Range: [0, 2]. Exact match = 2.0; collapsed "always 2 correct on a 3-valid"
    gets ~0.67 from PR only.
    """

    def enumerate_em_plus_prp_reward(state: State, **kwargs) -> float:
        true_set = get_true_valid_set(state)
        pred_set = _extract_valid_moves_set(_completion_text(state))
        if pred_set is None:
            return 0.0
        if not true_set:
            return 2.0 if not pred_set else 0.0
        em = 1.0 if pred_set == true_set else 0.0
        tp = len(pred_set & true_set)
        fp = len(pred_set - true_set)
        recall = tp / len(true_set)
        wrong_frac = fp / 4
        pr = max(0.0, recall * (1.0 - fp_penalty * wrong_frac))
        return em + pr

    enumerate_em_plus_prp_reward.__name__ = f"enumerate_em_plus_prp_reward_fp{fp_penalty:g}"
    return enumerate_em_plus_prp_reward


def make_em_only_reward():
    """Build a pure-EM reward for the enumerate task."""

    def enumerate_em_reward(state: State, **kwargs) -> float:
        true_set = get_true_valid_set(state)
        pred_set = _extract_valid_moves_set(_completion_text(state))
        if pred_set is None:
            return 0.0
        return 1.0 if pred_set == true_set else 0.0

    return enumerate_em_reward


def make_pr_penalty_reward(fp_penalty: float = 2.0):
    """Recall * max(0, 1 - fp_penalty * FP / 4) reward.

    Designed to kill the 'predict all 4' shortcut:
      - exact match  -> 1.0
      - strict subset of true -> recall (0.5 for missing one of two, etc.)
      - over-prediction multiplicatively penalized by 1 - fp_penalty * FP/4
      - at fp_penalty=2, 2+ wrong predictions -> 0
      - predict-all-4 on 2-valid -> recall=1 * (1 - 2*2/4) = 0
      - predict-all-4 on 3-valid -> 1 * (1 - 2*1/4) = 0.5
      - predict-all-4 on 4-valid -> 1.0
    """

    def enumerate_pr_penalty_reward(state: State, **kwargs) -> float:
        true_set = get_true_valid_set(state)
        pred_set = _extract_valid_moves_set(_completion_text(state))
        if pred_set is None:
            return 0.0
        if not true_set:
            return 1.0 if not pred_set else 0.0
        tp = len(pred_set & true_set)
        fp = len(pred_set - true_set)
        recall = tp / len(true_set)
        wrong_frac = fp / 4
        return max(0.0, recall * (1.0 - fp_penalty * wrong_frac))

    enumerate_pr_penalty_reward.__name__ = f"enumerate_pr_penalty_reward_fp{fp_penalty:g}"
    return enumerate_pr_penalty_reward


# ---------------------------------------------------------------------------
# Length / anti-yapping reward (group-level)
# ---------------------------------------------------------------------------

def make_length_reward(
    alpha: float = 0.1,
    task_reward_fn=None,
    min_l_max: int = 64,
):
    """Group-level length-penalty reward.

    Reward per rollout i:  -alpha * k_q * (|y_i| / l_max)

    Where, computed across the group of rollouts for the same prompt:
      k_q   = mean(task_reward_fn(state) > 0)
      l_max = max(|y_i|, min_l_max)
      |y_i| = token count of completion (falls back to char count if tokens unavailable)

    Notes:
      - Returns a list (one per state) so verifiers routes this as a GroupRewardFunc
        (group routing key off plural param names like `states`).
      - When all rollouts fail (k_q=0), no penalty is applied: thinking is free
        when the model still hasn't figured out the task.
      - When alpha=0, the function is a no-op.
    """

    if task_reward_fn is None:
        raise ValueError("task_reward_fn is required to compute k_q (group pass rate)")

    async def length_reward(states, **kwargs) -> list[float]:
        if alpha == 0.0:
            return [0.0] * len(states)

        lens = [rollout_completion_length(s) for s in states]
        l_max = max(max(lens), min_l_max)

        task_scores = []
        for s in states:
            score = task_reward_fn(s)
            # task_reward_fn may be async (it usually isn't here)
            if hasattr(score, "__await__"):
                score = await score
            task_scores.append(float(score))
        k_q = sum(1 for s in task_scores if s > 0) / len(task_scores)

        return [-alpha * k_q * (lens[i] / l_max) for i in range(len(states))]

    length_reward.__name__ = f"length_reward_alpha{alpha:g}"
    return length_reward


def rollout_completion_length(state: State) -> int:
    """Total assistant-side length for this rollout.

    Prefers trajectory token_ids (the source of truth for what the trainer sees).
    Falls back to character count over completion message `content` plus
    `reasoning_content` (so Qwen3's `<think>...</think>` block is counted even
    when split out by the reasoning parser).
    """
    total = 0
    for step in state.get("trajectory", []) or []:
        tokens = step.get("tokens") if isinstance(step, dict) else None
        if tokens and tokens.get("completion_ids"):
            total += len(tokens["completion_ids"])
    if total > 0:
        return total

    for msg in state.get("completion", []):
        content = msg.content if hasattr(msg, "content") else (msg.get("content") if isinstance(msg, dict) else None)
        reasoning = (
            msg.reasoning_content
            if hasattr(msg, "reasoning_content")
            else (msg.get("reasoning_content") if isinstance(msg, dict) else None)
        )
        total += len(content or "")
        total += len(reasoning or "")
    return max(total, 1)


# =============================================================================
# Environment
# =============================================================================

class Game2048SingleTurnEnv(vf.SingleTurnEnv):
    """Single-turn environment for learning valid 2048 moves."""

    env_id = "env-2048-single-turn"

    def __init__(
        self,
        grid_size: int = 4,
        target_tile: int = 2048,
        **kwargs,
    ):
        # Ensure env_id is set even when the caller passes nothing — Environment's
        # init replaces the class-level env_id attribute with "" otherwise.
        kwargs.setdefault("env_id", self.env_id)
        super().__init__(**kwargs)
        self.grid_size = grid_size
        self.target_tile = target_tile
    
    async def setup_state(self, state: State) -> State:
        """Initialize game from dataset row."""
        state = await super().setup_state(state)

        info = state["input"].get("info", {})
        if isinstance(info, str):
            import json
            info = json.loads(info)

        grid_cells = info.get("grid")
        size = info.get("size", self.grid_size)
        target = info.get("target_tile", self.target_tile)

        if grid_cells is None:
            raise ValueError(
                f"setup_state: dataset row is missing 'grid' in info. info keys: "
                f"{list(info.keys()) if isinstance(info, dict) else type(info)!r}"
            )

        grid = Grid(size=size, cells=grid_cells)
        game = Game2048(size=size, target_tile=target, grid=grid)
        game.score = info.get("score", 0)

        state["game"] = game
        state["move_reward"] = 0.0

        return state


# =============================================================================
# Environment Loader
# =============================================================================

def load_environment(
    num_train_examples: int = 1000,
    num_eval_examples: int = 100,
    grid_size: int = 4,
    target_tile: int = 2048,
    min_moves: int = 0,
    max_moves: int = 100,
    seed: int = 42,
    balanced_difficulty: bool = False,
    output_format: str = "xml",
    task_type: str = "pick_one",
    reward_type: str = "blend",
    em_weight: float = 0.7,
    f1_weight: float = 0.3,
    pr_penalty_fp: float = 2.0,
    length_alpha: float = 0.0,
    difficulty_distribution: tuple[float, float, float] = (0.75, 0.125, 0.125),
    **kwargs,
) -> vf.Environment:
    """Load the 2048 single-turn environment.

    Args:
        output_format: "xml" → `<move>direction</move>` (default, back-compat)
                       "json" → `{"move": "direction"}` (pick_one) or
                                `{"valid_moves": [...]}` (enumerate_all)
        task_type:     "pick_one" (default) — output ONE valid direction.
                       "enumerate_all" — output the SET of all valid directions
                       (requires output_format="json").
        reward_type:   Which enumerate-task reward to use.
                       "blend" — em_weight*EM + f1_weight*F1 (NOT recommended:
                                 collapses to predict-all-4 under GRPO).
                       "em" — exact-match only (1.0/0.0).
                       "pr_penalty" — recall * max(0, 1 - pr_penalty_fp*FP/4).
                                      Kills the predict-all shortcut while still
                                      giving partial credit to strict subsets.
                       "em_plus_prp" — EM + PR_penalty (range [0,2]).
                       Only used when task_type="enumerate_all".
        em_weight, f1_weight: Weights for reward_type="blend".
        pr_penalty_fp: FP multiplier for reward_type="pr_penalty" (default 2.0).
        length_alpha:  Strength of the anti-yapping length reward. 0 disables it.
                       Per rollout: -alpha * k_q * (|y_i| / l_max), where k_q is
                       the group's task-reward pass-rate.
        difficulty_distribution: Tuple (frac_2valid, frac_3valid, frac_4valid)
                       used when balanced_difficulty=True. Default (.75,.125,.125).
                       Try (0.34, 0.33, 0.33) for uniform.
    """
    if output_format not in ("xml", "json"):
        raise ValueError(f"output_format must be 'xml' or 'json', got {output_format!r}")
    if task_type not in ("pick_one", "enumerate_all"):
        raise ValueError(f"task_type must be 'pick_one' or 'enumerate_all', got {task_type!r}")
    if task_type == "enumerate_all" and output_format != "json":
        raise ValueError("task_type='enumerate_all' requires output_format='json'")
    if reward_type not in ("blend", "em", "pr_penalty", "em_plus_prp"):
        raise ValueError(f"reward_type must be 'blend', 'em', 'pr_penalty', or 'em_plus_prp', got {reward_type!r}")

    system_prompt = get_system_prompt(
        grid_size=grid_size,
        target_tile=target_tile,
        output_format=output_format,
        task_type=task_type,
    )

    train_dataset = generate_single_turn_dataset(
        num_train_examples,
        size=grid_size,
        target_tile=target_tile,
        min_moves=min_moves,
        max_moves=max_moves,
        seed=seed,
        balanced_difficulty=balanced_difficulty,
        output_format=output_format,
        task_type=task_type,
        difficulty_distribution=difficulty_distribution,
    )
    eval_dataset = generate_single_turn_dataset(
        num_eval_examples,
        size=grid_size,
        target_tile=target_tile,
        min_moves=min_moves,
        max_moves=max_moves,
        seed=seed + 10000,
        balanced_difficulty=balanced_difficulty,
        output_format=output_format,
        task_type=task_type,
        difficulty_distribution=difficulty_distribution,
    )

    if task_type == "pick_one":
        if output_format == "xml":
            parser = vf.XMLParser(fields=["move"], answer_field="move")
        else:
            parser = vf.Parser()
        task_reward = make_valid_move_reward(output_format)
    else:
        parser = vf.Parser()
        if reward_type == "em":
            task_reward = make_em_only_reward()
        elif reward_type == "pr_penalty":
            task_reward = make_pr_penalty_reward(fp_penalty=pr_penalty_fp)
        elif reward_type == "em_plus_prp":
            task_reward = make_em_plus_prp_reward(fp_penalty=pr_penalty_fp)
        else:
            task_reward = make_enumerate_reward(em_weight=em_weight, f1_weight=f1_weight)

    rubric = vf.Rubric(parser=parser)
    rubric.add_reward_func(task_reward, weight=1.0)
    if length_alpha > 0.0:
        rubric.add_reward_func(
            make_length_reward(alpha=length_alpha, task_reward_fn=task_reward),
            weight=1.0,
        )

    return Game2048SingleTurnEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        grid_size=grid_size,
        target_tile=target_tile,
    )
