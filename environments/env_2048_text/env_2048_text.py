"""
2048 Text Environment for LLM Training

A multi-turn RL environment where an LLM plays the 2048 game in text mode.
The game state is presented as an ASCII table, and the LLM makes moves
(up/down/left/right) to try to reach the 2048 tile.
"""

import json as _json
import random
import re
from copy import deepcopy
from enum import Enum
from functools import partial
from typing import List, Literal, Optional, Tuple

import numpy as np
from datasets import Dataset

import verifiers as vf
from verifiers.types import Messages, State
from verifiers.utils.message_utils import concat_messages, normalize_messages


# =============================================================================
# Context Modes
# =============================================================================

class ContextMode(str, Enum):
    """Context management modes for the 2048 environment."""
    FULL = "full"           # Full conversation history (default MultiTurnEnv behavior)
    MARKOV = "markov"       # Current state only (no history)
    LAST_K = "last_k"       # Last K turns of history
    SUMMARY = "summary"     # LLM-generated summary of game history


SUMMARY_SYSTEM_PROMPT = """You are a game analyst. Summarize this 2048 game history concisely.
Focus on: key merges achieved, strategy patterns observed, and what to do next.
Keep it under 100 words. Be direct and actionable for the player."""


# =============================================================================
# Game Logic
# =============================================================================


class Grid:
    """NxN grid for the 2048 game."""

    def __init__(self, size: int = 4, cells: Optional[List[List[int]]] = None):
        self.size = size
        if cells is not None:
            self.cells = deepcopy(cells)
        else:
            self.cells = [[0 for _ in range(size)] for _ in range(size)]

    def empty_cells(self) -> List[Tuple[int, int]]:
        """Return list of (x, y) positions that are empty (value 0)."""
        result = []
        for x in range(self.size):
            for y in range(self.size):
                if self.cells[x][y] == 0:
                    result.append((x, y))
        return result

    def has_empty_cells(self) -> bool:
        """Check if there are any empty cells."""
        return len(self.empty_cells()) > 0

    def random_empty_cell(self) -> Optional[Tuple[int, int]]:
        """Get a random empty cell position."""
        empty = self.empty_cells()
        if empty:
            return random.choice(empty)
        return None

    def get(self, x: int, y: int) -> int:
        """Get value at position."""
        return self.cells[x][y]

    def set(self, x: int, y: int, value: int):
        """Set value at position."""
        self.cells[x][y] = value

    def within_bounds(self, x: int, y: int) -> bool:
        """Check if position is within grid bounds."""
        return 0 <= x < self.size and 0 <= y < self.size

    def clone(self) -> "Grid":
        """Create a deep copy of the grid."""
        return Grid(self.size, self.cells)


class Game2048:
    """
    2048 game engine implementing the full game logic.
    Based on the JavaScript reference implementation.
    """

    # Direction vectors: 0=up, 1=right, 2=down, 3=left
    VECTORS = {
        0: (0, -1),   # up
        1: (1, 0),    # right
        2: (0, 1),    # down
        3: (-1, 0),   # left
    }

    DIRECTION_NAMES = {
        "up": 0, "u": 0, "0": 0,
        "right": 1, "r": 1, "1": 1,
        "down": 2, "d": 2, "2": 2,
        "left": 3, "l": 3, "3": 3,
    }

    def __init__(self, size: int = 4, start_tiles: int = 2, target_tile: int = 2048):
        self.size = size
        self.start_tiles = start_tiles
        self.target_tile = target_tile
        self.grid = Grid(size)
        self.score = 0
        self.over = False
        self.won = False
        self.move_count = 0
        self.max_tile = 0

        # Add initial tiles
        for _ in range(self.start_tiles):
            self.add_random_tile()

        self._update_max_tile()

    def _update_max_tile(self):
        """Update the maximum tile value on the board."""
        max_val = 0
        for x in range(self.size):
            for y in range(self.size):
                max_val = max(max_val, self.grid.get(x, y))
        self.max_tile = max_val

    def add_random_tile(self):
        """Add a random tile (90% chance of 2, 10% chance of 4) to an empty cell."""
        pos = self.grid.random_empty_cell()
        if pos:
            value = 2 if random.random() < 0.9 else 4
            self.grid.set(pos[0], pos[1], value)

    def _build_traversals(self, vector: Tuple[int, int]) -> Tuple[List[int], List[int]]:
        """
        Build traversal order based on movement direction.
        We process tiles in reverse order of movement to avoid interference.
        """
        x_traversal = list(range(self.size))
        y_traversal = list(range(self.size))

        # Reverse traversal direction if moving right or down
        if vector[0] == 1:  # moving right
            x_traversal.reverse()
        if vector[1] == 1:  # moving down
            y_traversal.reverse()

        return x_traversal, y_traversal

    def _find_farthest_position(
        self, x: int, y: int, vector: Tuple[int, int]
    ) -> Tuple[Tuple[int, int], Optional[Tuple[int, int]]]:
        """
        Find the farthest position a tile can move to in the given direction.
        Returns (farthest_empty_pos, next_pos_if_occupied).
        """
        prev_x, prev_y = x, y
        while True:
            next_x = prev_x + vector[0]
            next_y = prev_y + vector[1]
            if not self.grid.within_bounds(next_x, next_y) or self.grid.get(next_x, next_y) != 0:
                break
            prev_x, prev_y = next_x, next_y

        # Check if there's a tile at the next position (for merging)
        next_x = prev_x + vector[0]
        next_y = prev_y + vector[1]
        next_pos = None
        if self.grid.within_bounds(next_x, next_y) and self.grid.get(next_x, next_y) != 0:
            next_pos = (next_x, next_y)

        return (prev_x, prev_y), next_pos

    def move(self, direction: int) -> bool:
        """
        Execute a move in the given direction.
        Returns True if any tiles moved, False otherwise.
        """
        if self.over:
            return False

        vector = self.VECTORS[direction]
        x_traversal, y_traversal = self._build_traversals(vector)

        moved = False
        # Track which tiles have been merged this turn
        merged = [[False for _ in range(self.size)] for _ in range(self.size)]

        for x in x_traversal:
            for y in y_traversal:
                value = self.grid.get(x, y)
                if value == 0:
                    continue

                farthest, next_pos = self._find_farthest_position(x, y, vector)

                # Check for merge
                if next_pos:
                    next_x, next_y = next_pos
                    next_value = self.grid.get(next_x, next_y)
                    if next_value == value and not merged[next_x][next_y]:
                        # Merge!
                        new_value = value * 2
                        self.grid.set(next_x, next_y, new_value)
                        self.grid.set(x, y, 0)
                        merged[next_x][next_y] = True
                        self.score += new_value

                        if new_value >= self.target_tile:
                            self.won = True

                        moved = True
                        continue

                # Move to farthest position if different from current
                if farthest != (x, y):
                    self.grid.set(farthest[0], farthest[1], value)
                    self.grid.set(x, y, 0)
                    moved = True

        if moved:
            self.add_random_tile()
            self.move_count += 1
            self._update_max_tile()

            if not self._moves_available():
                self.over = True

        return moved

    def _moves_available(self) -> bool:
        """Check if any moves are available."""
        if self.grid.has_empty_cells():
            return True
        return self._tile_matches_available()

    def _tile_matches_available(self) -> bool:
        """Check if any adjacent tiles can be merged."""
        for x in range(self.size):
            for y in range(self.size):
                value = self.grid.get(x, y)
                if value == 0:
                    continue
                # Check all 4 directions
                for direction in range(4):
                    vector = self.VECTORS[direction]
                    nx, ny = x + vector[0], y + vector[1]
                    if self.grid.within_bounds(nx, ny) and self.grid.get(nx, ny) == value:
                        return True
        return False

    def parse_move(self, move_str: str) -> Optional[int]:
        """Parse a move string into a direction. Returns None if invalid."""
        move_str = move_str.strip().lower()
        return self.DIRECTION_NAMES.get(move_str)

    def to_ascii(self) -> str:
        """
        Render the game state as an ASCII table.
        """
        # Calculate column width based on largest number
        max_val = max(
            self.grid.get(x, y)
            for x in range(self.size)
            for y in range(self.size)
        )
        col_width = max(4, len(str(max_val)) + 2)

        lines = []
        separator = "+" + (("-" * col_width + "+") * self.size)

        for y in range(self.size):
            lines.append(separator)
            row = "|"
            for x in range(self.size):
                val = self.grid.get(x, y)
                cell = str(val) if val > 0 else "."
                row += cell.center(col_width) + "|"
            lines.append(row)
        lines.append(separator)

        return "\n".join(lines)

    def get_state_text(self) -> str:
        """Get full game state as text."""
        parts = [
            f"Score: {self.score}",
            f"Moves: {self.move_count}",
            f"Max Tile: {self.max_tile}",
            "",
            self.to_ascii(),
        ]

        if self.won:
            parts.append("\n🎉 You reached 2048! You can keep playing.")
        if self.over:
            parts.append("\n❌ Game Over! No more moves available.")

        return "\n".join(parts)

    def clone(self) -> "Game2048":
        """Create a copy of the game state."""
        new_game = Game2048.__new__(Game2048)
        new_game.size = self.size
        new_game.start_tiles = self.start_tiles
        new_game.target_tile = self.target_tile
        new_game.grid = self.grid.clone()
        new_game.score = self.score
        new_game.over = self.over
        new_game.won = self.won
        new_game.move_count = self.move_count
        new_game.max_tile = self.max_tile
        return new_game
    
    def to_matrix(self) -> np.ndarray:
        """Get the grid as a 2D numpy array of integers."""
        return np.array(self.grid.cells)


# =============================================================================
# Verifiers Environment
# =============================================================================

DEFAULT_SYSTEM_PROMPT_XML = """You are playing the 2048 puzzle game. Your goal is to combine tiles by sliding them in one of four directions (up, down, left, right) to create a tile with the value {target_tile}.

## Rules:
- The board is a {grid_size}x{grid_size} grid
- Tiles slide as far as possible in the chosen direction
- When two tiles with the same value collide, they merge into one tile with double the value
- After each move, a new tile (2 or 4) appears in a random empty cell
- The game ends when no more moves are possible

## How to Play:
Look at the current game state and choose your next move. Respond with your move inside <move>...</move> tags.

Valid moves are: up, down, left, right (or u, d, l, r for short)
up slides all tiles to the top, down slides to the bottom,left slides to the left, and right slides to the right.

Example response:
<reasoning>The highest tile (512) is in the top-left corner. I should keep it there by moving up to merge the two 64 tiles below.</reasoning>
<move>up</move>
"""

DEFAULT_SYSTEM_PROMPT_JSON = """You are playing the 2048 puzzle game. Your goal is to combine tiles by sliding them in one of four directions (up, down, left, right) to create a tile with the value {target_tile}.

## Rules:
- The board is a {grid_size}x{grid_size} grid
- Tiles slide as far as possible in the chosen direction
- When two tiles with the same value collide, they merge into one tile with double the value
- After each move, a new tile (2 or 4) appears in a random empty cell
- The game ends when no more moves are possible

## How to Play:
Look at the current game state and choose your next move. Respond with a JSON object containing a single key "move" whose value is the direction.

Valid moves are: up, down, left, right (or u, d, l, r for short).
up slides all tiles to the top, down slides to the bottom, left slides to the left, and right slides to the right.

Example response:
{{"move": "up"}}
"""

# Backward compatibility alias
DEFAULT_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT_XML


def _format_instruction(output_format: str, first: bool = False) -> str:
    """One-line instruction telling the model how to format its move."""
    if output_format == "json":
        if first:
            return 'What\'s your first move? Respond with {"move": "direction"}.'
        return 'What\'s your move? Respond with {"move": "direction"}.'
    if first:
        return "What's your first move? Remember to put your move inside <move>...</move> tags."
    return "What's your move? Remember to put your move inside <move>...</move> tags."


def _extract_move_xml(text: str) -> Optional[str]:
    match = re.search(r"<move>\s*(.*?)\s*</move>", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def _extract_move_json(text: str) -> Optional[str]:
    """Extract a move string from a JSON-formatted response. Tolerant to code fences."""
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

    # Last-resort regex
    kv = re.search(r'"move"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    if kv:
        return kv.group(1).strip()
    return None


def _extract_move(text: str, output_format: str) -> Optional[str]:
    if output_format == "json":
        return _extract_move_json(text)
    return _extract_move_xml(text)


def _format_help(output_format: str) -> str:
    """Short string used in error messages to remind the model of the format."""
    if output_format == "json":
        return 'Respond with a JSON object like {"move": "up"} (use up/down/left/right).'
    return "Use <move>up/down/left/right</move>."


def get_system_prompt(
    grid_size: int = 4,
    target_tile: int = 2048,
    output_format: str = "xml",
) -> str:
    """Generate a system prompt with the specified grid size, target tile, and output format."""
    if output_format == "json":
        template = DEFAULT_SYSTEM_PROMPT_JSON
    elif output_format == "xml":
        template = DEFAULT_SYSTEM_PROMPT_XML
    else:
        raise ValueError(f"Unknown output_format: {output_format!r}. Expected 'xml' or 'json'.")
    return template.format(grid_size=grid_size, target_tile=target_tile)


def generate_game_dataset(
    num_examples: int,
    grid_size: int = 4,
    target_tile: int = 2048,
    seed: int = 42,
    output_format: str = "xml",
) -> Dataset:
    """
    Generate a dataset of 2048 game starting positions.
    Each example is a fresh game with initial tiles.
    
    Args:
        num_examples: Number of game instances to generate
        grid_size: Size of the grid (e.g., 4 for 4x4, 5 for 5x5)
        target_tile: Target tile value to win (e.g., 2048, 1024, 4096)
        seed: Random seed for reproducibility
        output_format: "xml" or "json" — controls the move format hint in the
            initial user prompt (must match Game2048Env.output_format).
    """
    random.seed(seed)
    
    data = {
        "task": [],
        "prompt": [],
        "answer": [],  # We don't have a specific answer, but include for compatibility
        "info": [],
    }

    move_hint = _format_instruction(output_format, first=True)

    for i in range(num_examples):
        # Create a new game with specified size and target
        game = Game2048(size=grid_size, target_tile=target_tile)
        
        initial_state = game.get_state_text()
        
        prompt_content = f"""Let's play 2048! Here's the starting board ({grid_size}x{grid_size} grid, target: {target_tile}):

{initial_state}

{move_hint}"""

        data["task"].append("2048")
        data["prompt"].append([{"role": "user", "content": prompt_content}])
        data["answer"].append(str(target_tile))  # Target tile as answer
        data["info"].append({
            "game_id": i,
            "initial_grid": deepcopy(game.grid.cells),
            "grid_size": grid_size,
            "target_tile": target_tile,
            "seed": seed + i,
        })

    return Dataset.from_dict(data)


class Game2048Env(vf.MultiTurnEnv):
    """
    Multi-turn environment for playing 2048.
    
    The LLM receives the game state as an ASCII grid and responds with moves.
    The environment processes moves and returns updated game state.
    
    Supports configurable grid sizes (e.g., 3x3, 4x4, 5x5) and target tiles.
    
    Context Modes:
        - "full": Full conversation history (default)
        - "markov": Current state only (no history) - ideal for 2048
        - "last_k": Last K turns of history
        - "summary": LLM-generated summary of game history
    """

    def __init__(
        self,
        max_moves: int = 500,
        target_tile: int = 2048,
        grid_size: int = 4,
        context_mode: Literal["full", "markov", "last_k", "summary"] = "full",
        last_k_turns: int = 3,
        summary_interval: int = 5,
        summary_model: Optional[str] = None,
        max_invalid_moves: int = 10,
        output_format: Literal["xml", "json"] = "xml",
        **kwargs,
    ):
        """
        Initialize the 2048 environment.
        
        Args:
            max_moves: Maximum moves allowed per game
            target_tile: Target tile value to win
            grid_size: Size of the grid (e.g., 4 for 4x4)
            context_mode: How to manage conversation context
                - "full": Send full conversation history
                - "markov": Send only current state (recommended for 2048)
                - "last_k": Send last K turns of history
                - "summary": LLM-generated summary of game history
            last_k_turns: Number of turns to keep when context_mode="last_k"
            summary_interval: How often to regenerate LLM summary (every N turns)
            summary_model: Model to use for summarization (defaults to same as main model)
            max_invalid_moves: Stop if model makes this many invalid moves total
            output_format: "xml" (<move>...</move>) or "json" ({"move": "..."}).
                Controls how moves are parsed from assistant messages and how
                error/continuation user messages are phrased.
        """
        # max_turns in parent class controls the game length
        super().__init__(max_turns=max_moves, **kwargs)
        self.target_tile = target_tile
        self.grid_size = grid_size
        self.context_mode = ContextMode(context_mode)
        self.last_k_turns = last_k_turns
        self.summary_interval = summary_interval
        self.summary_model = summary_model
        self.max_invalid_moves = max_invalid_moves
        if output_format not in ("xml", "json"):
            raise ValueError(f"Unknown output_format: {output_format!r}. Expected 'xml' or 'json'.")
        self.output_format = output_format

    async def setup_state(self, state: State) -> State:
        """Initialize game state for this rollout."""
        info = state["info"]
        
        # Get grid size and target from info (per-example) or fall back to env defaults
        grid_size = info.get("grid_size", self.grid_size)
        target_tile = info.get("target_tile", self.target_tile)
        
        # Create game with specified size and target
        game = Game2048(size=grid_size, target_tile=target_tile)
        
        # If initial grid is provided, restore it
        if "initial_grid" in info:
            game.grid.cells = deepcopy(info["initial_grid"])
            game._update_max_tile()
        
        # Seed random for this game instance
        if "seed" in info:
            random.seed(info["seed"])
        
        # Store game in state
        state["game"] = game
        state["invalid_moves"] = 0  # Total invalid moves (for metrics)
        state["consecutive_invalid_moves"] = 0  # Consecutive invalid moves (for termination)
        state["valid_moves"] = 0
        state["move_history"] = []  # For tracking moves
        state["llm_summary"] = None  # Cached LLM summary
        state["last_summary_turn"] = 0  # When we last generated summary
        
        return state

    def _get_current_state_prompt(self, game: Game2048) -> str:
        """Generate a prompt with just the current game state."""
        return f"""{game.get_state_text()}

{_format_instruction(self.output_format)}"""

    async def _generate_llm_summary(self, state: State) -> str:
        """Generate an LLM-based summary of the game history."""
        trajectory = state.get("trajectory", [])
        if not trajectory:
            return "Game just started. No history yet."
        
        # Build a condensed history for the summarizer
        game: Game2048 = state["game"]
        move_history = state.get("move_history", [])
        
        history_text = f"""2048 Game History:
- Total moves: {game.move_count}
- Current score: {game.score}
- Max tile achieved: {game.max_tile}
- Target tile: {game.target_tile}
- Move sequence: {', '.join(move_history[-20:])}{"..." if len(move_history) > 20 else ""}
"""
        
        # Get client and model from state
        client = state.get("client")
        model = self.summary_model or state.get("model")
        
        if not client or not model:
            # Fallback to simple summary if no client available
            return f"Moves: {game.move_count} | Score: {game.score} | Max: {game.max_tile}"
        
        try:
            # Call LLM for summary
            response = await client.generate(
                model=model,
                messages=[
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": history_text},
                ],
                max_tokens=150,
                temperature=0.3,
            )
            summary = response.message.content.strip()
            return summary
        except Exception as e:
            # Fallback on error
            return f"Moves: {game.move_count} | Score: {game.score} | Max: {game.max_tile} | Recent: {', '.join(move_history[-5:])}"

    async def _get_summary_prompt(self, state: State) -> str:
        """Generate a prompt with LLM summary + current state."""
        game: Game2048 = state["game"]
        current_turn = len(state.get("trajectory", []))
        
        # Check if we need to regenerate summary
        needs_new_summary = (
            state.get("llm_summary") is None or
            current_turn - state.get("last_summary_turn", 0) >= self.summary_interval
        )
        
        if needs_new_summary:
            state["llm_summary"] = await self._generate_llm_summary(state)
            state["last_summary_turn"] = current_turn
        
        summary = state.get("llm_summary", "")
        
        return f"""Game Summary:
{summary}

Current State:
{game.get_state_text()}

{_format_instruction(self.output_format)}"""

    async def get_prompt_messages(self, state: State) -> Messages:
        """
        Generate prompt messages based on the configured context mode.
        
        Overrides the default MultiTurnEnv behavior to support different
        context management strategies.
        
        For non-FULL modes, we maintain full history ourselves because parent
        builds from trajectory[-1]["prompt"] which is truncated.
        """
        # For full context mode, just use parent directly
        if self.context_mode == ContextMode.FULL:
            return await super().get_prompt_messages(state)
        
        trajectory = state["trajectory"]
        
        # First turn: initialize with original prompt
        if len(trajectory) == 0:
            full_messages = normalize_messages(state["prompt"], field_name="state.prompt")
            state["_full_messages_for_logging"] = list(full_messages)
            return full_messages
        
        # Get existing full messages (accumulated history)
        full_messages = state.get("_full_messages_for_logging", [])
        if not full_messages:
            # Fallback - shouldn't happen but just in case
            full_messages = list(normalize_messages(state["prompt"], field_name="state.prompt"))
        
        # Add the last completion from trajectory
        last_completion = normalize_messages(
            trajectory[-1]["completion"], 
            field_name="trajectory.completion"
        )
        full_messages = concat_messages([full_messages, last_completion])
        
        # Call env_response to process the move and update game state
        env_response = await self.env_response(full_messages, state)
        env_response_messages = normalize_messages(env_response, field_name="env_response")
        full_messages = concat_messages([full_messages, env_response_messages])
        
        # Store updated full messages for logging
        state["_full_messages_for_logging"] = list(full_messages)
        
        # Check if game ended (final_env_response set by env_response)
        if state.get("final_env_response") is not None:
            return full_messages
        
        game: Game2048 = state.get("game")
        
        if self.context_mode == ContextMode.MARKOV:
            # Markov: only system prompt + current state
            return normalize_messages([
                {"role": "system", "content": state.get("system_prompt", self.system_prompt)},
                {"role": "user", "content": self._get_current_state_prompt(game)},
            ], field_name="markov_prompt")
        
        elif self.context_mode == ContextMode.LAST_K:
            # Last K turns: system + last K (user, assistant) pairs + current state
            return self._build_last_k_messages(state, game)
        
        elif self.context_mode == ContextMode.SUMMARY:
            # Summary: system + summary + current state
            summary_prompt = await self._get_summary_prompt(state)
            return normalize_messages([
                {"role": "system", "content": state.get("system_prompt", self.system_prompt)},
                {"role": "user", "content": summary_prompt},
            ], field_name="summary_prompt")
        
        return full_messages
    
    async def render_completion(self, state: State):
        """
        Override to use full conversation history for logging in non-FULL modes.
        """
        # For FULL mode, use parent's implementation
        if self.context_mode == ContextMode.FULL:
            await super().render_completion(state)
            return
        
        if len(state["trajectory"]) == 0:
            state["completion"] = []
            return
        
        # Get the full messages we stored
        full_messages = state.get("_full_messages_for_logging")
        
        if full_messages is None:
            # Fallback to parent behavior
            await super().render_completion(state)
            return
        
        # Add the last assistant completion
        last_completion = normalize_messages(
            state["trajectory"][-1]["completion"], 
            field_name="trajectory.completion"
        )
        full_conversation = concat_messages([full_messages, last_completion])
        
        # Add final env response if present
        if state.get("final_env_response"):
            full_conversation = concat_messages([
                full_conversation,
                normalize_messages(state["final_env_response"], field_name="final_env_response")
            ])
        
        # Remove initial prompt to get just the "completion" part
        prompt_messages = normalize_messages(state["prompt"], field_name="state.prompt")
        state["completion"] = full_conversation[len(prompt_messages):]
    
    def _build_last_k_messages(self, state: State, game: Game2048) -> Messages:
        """Build messages with last K turns of history using actual env responses."""
        # full_messages contains: [system, user0, asst0, user1, asst1, ..., asst_{N-1}, userN]
        # where each userI (i > 0) is the env_response showing state after turn i-1
        full_messages = state.get("_full_messages_for_logging", [])
        
        if len(full_messages) < 2:
            # Not enough history, return what we have
            return full_messages
        
        # Count how many turns we have (each turn = 1 assistant + 1 user after the initial)
        # Structure: system, user0, [asst0, user1], [asst1, user2], ..., [asst_{N-1}, userN]
        # Pairs after initial: (len - 2) // 2 complete turns, possibly +1 if ends with userN
        
        # We want: system + last k turns worth of (user, assistant) pairs + current user
        # A "turn" in our context = assistant response + env response (user message)
        
        k = self.last_k_turns
        
        # Extract messages after system prompt
        system_msg = full_messages[0]
        rest = full_messages[1:]  # [user0, asst0, user1, asst1, ..., userN]
        
        # We want the last 2*k messages from rest (k assistant + k user messages)
        # Plus we always want to end with the current state (last user message)
        if len(rest) <= 2 * k:
            # Not enough history, return all
            return full_messages
        
        # Take last 2*k messages (this gives us k turns of asst+user pairs)
        # But we need to make sure we start with a user message for proper alternation
        last_messages = rest[-(2 * k):]
        
        # If we're starting mid-conversation, we want: user, asst, user, asst, ..., user
        # Check if first message is user or assistant
        if last_messages and last_messages[0].get("role") == "assistant":
            # We started with assistant, need to include one more user before it
            # Or just skip the first assistant to maintain user-first order
            # Actually, let's take 2*k + 1 to ensure we start with user
            last_messages = rest[-(2 * k + 1):] if len(rest) > 2 * k else rest
        
        return normalize_messages([system_msg] + last_messages, field_name="last_k_prompt")

    @vf.stop
    async def game_over(self, state: State) -> bool:
        """Stop when game is over (no moves available)."""
        game: Game2048 = state.get("game")
        if game:
            return game.over
        return False

    @vf.stop
    async def target_reached(self, state: State) -> bool:
        """Stop when target tile is reached."""
        game: Game2048 = state.get("game")
        if game:
            return game.max_tile >= game.target_tile
        return False

    @vf.stop
    async def too_many_invalid_moves(self, state: State) -> bool:
        """Stop when model has made too many consecutive invalid moves."""
        consecutive = state.get("consecutive_invalid_moves", 0)
        return consecutive >= self.max_invalid_moves

    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> Messages:
        """
        Process the LLM's move and return updated game state.
        """
        game: Game2048 = state["game"]
        
        # Extract the last assistant message
        last_assistant_msg = None
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                last_assistant_msg = msg["content"]
                break

        if not last_assistant_msg:
            return [{"role": "user", "content": f"Please make a move. {_format_help(self.output_format)}"}]

        # Parse move from response
        move_str = _extract_move(last_assistant_msg, self.output_format)

        if move_str is None:
            state["invalid_moves"] += 1
            state["consecutive_invalid_moves"] += 1
            return [{
                "role": "user",
                "content": f"""I couldn't find your move. {_format_help(self.output_format)}

Current board:
{game.get_state_text()}

Valid moves: up, down, left, right (or u, d, l, r)
What's your move?""",
            }]

        direction = game.parse_move(move_str)

        if direction is None:
            state["invalid_moves"] += 1
            state["consecutive_invalid_moves"] += 1
            return [{
                "role": "user",
                "content": f"""Invalid move: "{move_str}". Valid moves are: up, down, left, right (or u, d, l, r).

Current board:
{game.get_state_text()}

What's your move?""",
            }]

        # Execute the move
        moved = game.move(direction)

        if not moved:
            state["invalid_moves"] += 1
            state["consecutive_invalid_moves"] += 1
            return [{
                "role": "user",
                "content": f"""That move ({move_str}) didn't change the board. Try a different direction.

Current board:
{game.get_state_text()}

What's your move?""",
            }]

        # Valid move - reset consecutive counter
        state["valid_moves"] += 1
        state["consecutive_invalid_moves"] = 0
        
        # Track move history for summary mode
        move_name = {0: "up", 1: "right", 2: "down", 3: "left"}.get(direction, move_str)
        if "move_history" not in state:
            state["move_history"] = []
        state["move_history"].append(move_name)

        # Check for game end conditions
        if game.won:
            state["final_env_response"] = [{
                "role": "user",
                "content": f"""Congratulations! You reached {game.target_tile}!

Final board:
{game.get_state_text()}

Excellent work!""",
            }]
            return state["final_env_response"]

        if game.over:
            state["final_env_response"] = [{
                "role": "user",
                "content": f"""Game Over! No more moves available.

Final board:
{game.get_state_text()}

You reached a max tile of {game.max_tile}. Better luck next time!""",
            }]
            return state["final_env_response"]

        # Continue playing
        return [{
            "role": "user",
            "content": f"""Move applied: {move_str}

{game.get_state_text()}

What's your next move?""",
        }]


# =============================================================================
# Reward Functions
# =============================================================================

def score_reward(state: State, **kwargs) -> float:
    """
    Reward based on game score, linearly normalized.

    score = sum of all merged-tile values during the game (standard 2048 scoring).
    A typical winning game scores around 20000, so we normalize by 20000 and cap at 1.0.
    """
    game: Game2048 = state.get("game")
    if not game:
        return 0.0
    return min(1.0, game.score / 20000.0)


def max_tile_reward(state: State, **kwargs) -> float:
    """
    Reward based on maximum tile achieved.
    Returns 1.0 for reaching target tile, scaled down for lower tiles.
    """
    game: Game2048 = state.get("game")
    if game:
        import math
        if game.max_tile > 0:
            log_tile = math.log2(game.max_tile)
            log_target = math.log2(game.target_tile)
            return min(1.0, log_tile / log_target)
    return 0.0


def win_reward(state: State, **kwargs) -> float:
    """Binary reward for winning (reaching target tile)."""
    game: Game2048 = state.get("game")
    if game:
        return 1.0 if game.won else 0.0
    return 0.0


def efficiency_reward(state: State, **kwargs) -> float:
    """
    Reward for winning efficiently (fewer moves is better).
    Only applies if the game was won.
    """
    game: Game2048 = state.get("game")
    if game and game.won:
        # Scale expected moves based on grid size and target
        # Larger grids and higher targets need more moves
        import math
        base_moves = 1000
        size_factor = (game.size / 4) ** 2
        target_factor = math.log2(game.target_tile) / math.log2(2048)
        expected_moves = base_moves * size_factor * target_factor
        return max(0.0, 1.0 - game.move_count / expected_moves)
    return 0.0


def valid_moves_ratio(state: State, **kwargs) -> float:
    """Reward for making valid moves (penalize invalid/unparseable moves)."""
    valid = state.get("valid_moves", 0)
    invalid = state.get("invalid_moves", 0)
    total = valid + invalid
    if total > 0:
        return valid / total
    return 0.0


def num_moves_reward(state: State, **kwargs) -> float:
    """
    Survival reward - rewards more moves (staying alive longer).
    
    Formula: log(num_moves + 1) / log(max_moves + 1)
    
    - More moves → higher reward (approaches 1)
    - Fewer moves → lower reward (approaches 0)
    
    Uses total moves (valid + invalid), not just valid moves.
    """
    import math
    valid_moves = state.get("valid_moves", 0)
    invalid_moves = state.get("invalid_moves", 0)
    num_moves = valid_moves + invalid_moves
    
    game: Game2048 = state.get("game")
    max_moves = game.max_moves if game else 500
    
    if max_moves <= 0:
        return 0.0
    
    return math.log(num_moves + 1) / math.log(max_moves + 1)


def max_tile_value(state: State, **kwargs) -> float:
    """Raw max tile value achieved during the game (e.g. 64, 128, 256)."""
    game: Game2048 = state.get("game")
    return float(game.max_tile) if game else 0.0


def game_score(state: State, **kwargs) -> float:
    """Raw game score (sum of merged-tile values)."""
    game: Game2048 = state.get("game")
    return float(game.score) if game else 0.0


def num_turns(state: State, **kwargs) -> float:
    """Total number of turns the model took (valid + invalid)."""
    return float(state.get("valid_moves", 0) + state.get("invalid_moves", 0))


def efficiency_reward(state: State, base_multiplier: float = 0.5, **kwargs) -> float:
    """
    Efficiency reward - rewards reaching high tiles in fewer moves.
    
    Formula: max_tile_reward * (base_multiplier + (1 - num_moves_reward))
    
    - base_multiplier ensures you still get reward even if you use all moves
    - (1 - num_moves_reward) gives bonus for fewer moves
    
    Range: [base_multiplier, base_multiplier + 1] * max_tile_reward
    - All moves used: max_tile_reward * base_multiplier
    - Minimal moves: max_tile_reward * (base_multiplier + 1)
    
    Use in later curriculum stages after model learns to reach high tiles.
    """
    tile_score = max_tile_reward(state)
    moves_score = num_moves_reward(state)
    
    return tile_score * (base_multiplier + (1 - moves_score))


# =============================================================================
# Environment Loader
# =============================================================================

def load_environment(
    num_train_examples: int = 100,
    num_eval_examples: int = 20,
    max_moves: int = 500,
    target_tile: int = 2048,
    grid_size: int = 4,
    context_mode: Literal["full", "markov", "last_k", "summary"] = "full",
    last_k_turns: int = 3,
    summary_interval: int = 5,
    summary_model: Optional[str] = None,
    max_invalid_moves: int = 10,
    system_prompt: Optional[str] = None,
    output_format: Literal["xml", "json"] = "xml",
    seed: int = 42,
    # Reward weights (set to 0 to disable)
    max_tile_weight: float = 0.5,
    valid_moves_weight: float = 0.5,
    score_weight: float = 0.0,
    num_moves_weight: float = 0.0,
    efficiency_weight: float = 0.0,
    efficiency_base_multiplier: float = 0.5,
    **kwargs,
) -> vf.Environment:
    """
    Load the 2048 text environment.
    
    Args:
        num_train_examples: Number of training examples (game instances)
        num_eval_examples: Number of evaluation examples
        max_moves: Maximum moves per game
        target_tile: Target tile to reach (default 2048)
        grid_size: Size of the grid (default 4 for 4x4)
        context_mode: How to manage conversation context
            - "full": Send full conversation history (default)
            - "markov": Send only current state (recommended for 2048)
            - "last_k": Send last K turns of history
            - "summary": LLM-generated summary of game history
        last_k_turns: Number of turns to keep when context_mode="last_k"
        summary_interval: How often to regenerate LLM summary (every N turns)
        summary_model: Model to use for summarization (defaults to same as main model)
        max_invalid_moves: Stop if model makes this many invalid moves (default 10)
        system_prompt: System prompt for the LLM (auto-generated if None)
        output_format: "xml" (default — <move>up</move>) or "json" ({"move": "up"})
        seed: Random seed for reproducibility
    
    Returns:
        Configured Game2048Env instance
    """
    if output_format not in ("xml", "json"):
        raise ValueError(f"Unknown output_format: {output_format!r}. Expected 'xml' or 'json'.")

    # Generate system prompt if not provided
    if system_prompt is None:
        system_prompt = get_system_prompt(
            grid_size=grid_size, target_tile=target_tile, output_format=output_format
        )
    
    # Generate datasets with specified grid size and target
    train_dataset = generate_game_dataset(
        num_train_examples,
        grid_size=grid_size,
        target_tile=target_tile,
        seed=seed,
        output_format=output_format,
    )
    eval_dataset = generate_game_dataset(
        num_eval_examples,
        grid_size=grid_size,
        target_tile=target_tile,
        seed=seed + 10000,
        output_format=output_format,
    )

    # Set up parser for extracting moves
    if output_format == "xml":
        parser = vf.XMLParser(fields=["move", "reasoning"], answer_field="move")
    else:
        # JSON parser using our tolerant extractor
        parser = vf.Parser(extract_fn=lambda text: _extract_move_json(text) or "")

    # Set up rubric with reward functions
    rubric = vf.Rubric(parser=parser)
    
    if max_tile_weight > 0:
        rubric.add_reward_func(max_tile_reward, weight=max_tile_weight)
    if valid_moves_weight > 0:
        rubric.add_reward_func(valid_moves_ratio, weight=valid_moves_weight)
    if score_weight > 0:
        rubric.add_reward_func(score_reward, weight=score_weight)
    if num_moves_weight > 0:
        rubric.add_reward_func(num_moves_reward, weight=num_moves_weight)
    if efficiency_weight > 0:
        rubric.add_reward_func(
            partial(efficiency_reward, base_multiplier=efficiency_base_multiplier),
            weight=efficiency_weight,
            name="efficiency_reward"
        )

    # Metrics (tracked but not used in reward)
    rubric.add_metric(efficiency_reward)
    rubric.add_metric(num_moves_reward)
    rubric.add_metric(max_tile_value)
    rubric.add_metric(game_score)
    rubric.add_metric(num_turns)

    return Game2048Env(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        max_moves=max_moves,
        target_tile=target_tile,
        grid_size=grid_size,
        context_mode=context_mode,
        last_k_turns=last_k_turns,
        summary_interval=summary_interval,
        summary_model=summary_model,
        max_invalid_moves=max_invalid_moves,
        output_format=output_format,
        **kwargs,
    )
