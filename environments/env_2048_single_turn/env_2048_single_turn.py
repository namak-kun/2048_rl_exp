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


def count_valid_moves(game: Game2048) -> int:
    """Count how many directions result in a valid move."""
    count = 0
    for direction in range(4):
        # Create a copy to test without modifying original
        grid_copy = Grid(
            size=game.size,
            cells=[row[:] for row in game.grid.cells]
        )
        test_game = Game2048(
            size=game.size,
            target_tile=game.target_tile,
            grid=grid_copy
        )
        if test_game.move(direction):
            count += 1
    return count


def generate_state_with_valid_move_count(
    target_valid_moves: int,
    size: int = 4,
    target_tile: int = 2048,
    max_attempts: int = 1000,
) -> Game2048 | None:
    """
    Generate a state with exactly target_valid_moves valid directions.
    
    Uses dense random generation with varying fill ratios.
    Returns None if can't find matching state within max_attempts.
    
    Note: 1-valid-move states are extremely rare. Consider using
    target_valid_moves=2 for "hard" states instead.
    """
    for _ in range(max_attempts):
        # Higher fill ratio = likely fewer valid moves
        # Adjust fill ratio based on target
        if target_valid_moves <= 2:
            fill_ratio = random.uniform(0.85, 1.0)
        elif target_valid_moves == 3:
            fill_ratio = random.uniform(0.70, 0.90)
        else:  # 4 valid moves
            fill_ratio = random.uniform(0.3, 0.65)
        
        game = generate_dense_random_state(
            size=size,
            target_tile=target_tile,
            fill_ratio=fill_ratio,
        )
        
        valid_count = count_valid_moves(game)
        if valid_count == target_valid_moves:
            return game
        # For target=1, also accept 2 since 1 is very rare
        if target_valid_moves == 1 and valid_count == 2:
            return game
    
    return None


def generate_single_turn_dataset(
    num_examples: int,
    size: int = 4,
    target_tile: int = 2048,
    min_moves: int = 0,
    max_moves: int = 100,
    seed: int = 42,
    balanced_difficulty: bool = False,
) -> Dataset:
    """
    Generate dataset of random game states for single-turn training.
    
    Args:
        num_examples: Number of examples to generate
        size: Board size
        target_tile: Target tile for win condition
        min_moves: Min random moves for simulation-based generation
        max_moves: Max random moves for simulation-based generation
        seed: Random seed
        balanced_difficulty: If True, sample equally from hard(1-2)/medium(3)/easy(4) states
    """
    random.seed(seed)
    
    examples = []
    
    if balanced_difficulty:
        # Hard-biased distribution: 75% hard, 12.5% medium, 12.5% easy
        # This gives ~50% random baseline (vs 57.5% with 60/20/20)
        hard_count = int(num_examples * 0.75)
        medium_count = int(num_examples * 0.125)
        easy_count = num_examples - hard_count - medium_count
        
        difficulty_targets = [
            (2, hard_count),    # hard: 1-2 valid moves (75%)
            (3, medium_count),  # medium: 3 valid moves (12.5%)
            (4, easy_count),    # easy: 4 valid moves (12.5%)
        ]
        
        for target_valid, count in difficulty_targets:
            generated = 0
            while generated < count:
                game = generate_state_with_valid_move_count(
                    target_valid_moves=target_valid,
                    size=size,
                    target_tile=target_tile,
                    max_attempts=500,
                )
                
                if game is None:
                    # Fallback to simulation-based if can't find target difficulty
                    game = generate_random_game_state(
                        size=size,
                        target_tile=target_tile,
                        min_moves=min_moves,
                        max_moves=max_moves,
                    )
                
                prompt_content = f"""Current board:
{game.get_state_text()}

Score: {game.score}
Max tile: {game.max_tile}

What's your move? Remember to put your move inside <move>...</move> tags."""
                
                examples.append({
                    "prompt": [{"role": "user", "content": prompt_content}],
                    "info": {
                        "grid": [row[:] for row in game.grid.cells],
                        "score": game.score,
                        "size": size,
                        "target_tile": target_tile,
                        "valid_move_count": count_valid_moves(game),
                    },
                })
                generated += 1
        
        # Shuffle to mix difficulties
        random.shuffle(examples)
    else:
        # Original behavior: simulation-based generation
        for i in range(num_examples):
            game = generate_random_game_state(
                size=size, 
                target_tile=target_tile,
                min_moves=min_moves,
                max_moves=max_moves,
            )
            
            prompt_content = f"""Current board:
{game.get_state_text()}

Score: {game.score}
Max tile: {game.max_tile}

What's your move? Remember to put your move inside <move>...</move> tags."""
            
            # Put game state in 'info' dict so verifiers preserves it
            examples.append({
                "prompt": [{"role": "user", "content": prompt_content}],
                "info": {
                    "grid": [row[:] for row in game.grid.cells],
                    "score": game.score,
                    "size": size,
                    "target_tile": target_tile,
                },
            })
    
    return Dataset.from_list(examples)


# =============================================================================
# System Prompt
# =============================================================================

def get_system_prompt(grid_size: int = 4, target_tile: int = 2048) -> str:
    return f"""You are playing the 2048 puzzle game. Your goal is to combine tiles by sliding them in one of four directions (up, down, left, right) to create a tile with the value {target_tile}.

## Rules:
- The board is a {grid_size}x{grid_size} grid
- Tiles slide as far as possible in the chosen direction
- When two tiles with the same value collide, they merge into one tile with double the value
- After each move, a new tile (2 or 4) appears in a random empty cell
- The game ends when no more moves are possible

## How to Play:
Look at the current game state and choose your next move. Respond with your move inside <move>...</move> tags.

Valid moves are: up, down, left, right (or u, d, l, r for short)
up slides all tiles to the top, down slides to the bottom, left slides to the left, and right slides to the right.

Example response:
<reasoning>The highest tile (512) is in the top-left corner. I should keep it there by moving up to merge the two 64 tiles below.</reasoning>
<move>up</move>
"""


# =============================================================================
# Reward Functions
# =============================================================================

def valid_move_reward(state: State, **kwargs) -> float:
    """
    Reward for making a valid move.
    
    Returns:
        1.0 if move was valid (correct format + actually moved tiles)
        0.1 if format was correct but move didn't change board
        0.0 if format was wrong or unparseable
    """
    import re
    import sys
    
    game: Game2048 = state.get("game")
    if game is None:
        print("REWARD DEBUG: game is None in state!", file=sys.stderr, flush=True)
        return 0.0
    
    completion = state.get("completion", [])
    
    # Get completion text
    completion_text = ""
    for msg in completion:
        if hasattr(msg, "content"):
            completion_text = msg.content
        elif isinstance(msg, dict):
            completion_text = msg.get("content", "")
    
    # Parse move from completion
    move_match = re.search(r"<move>\s*(.*?)\s*</move>", completion_text, re.IGNORECASE | re.DOTALL)
    
    if not move_match:
        # No move tags found
        return 0.0
    
    move_str = move_match.group(1).strip()
    direction = Game2048.parse_move(move_str)
    
    if direction is None:
        # Invalid direction
        return 0.0
    
    # Try the move - make sure we're using a fresh copy
    old_cells = [row[:] for row in game.grid.cells]
    game_copy = Game2048(size=game.size, target_tile=game.target_tile, grid=game.grid.copy())
    game_copy.score = game.score
    
    moved = game_copy.move(direction)
    new_cells = game_copy.grid.cells
    
    # Debug logging 
    if not hasattr(valid_move_reward, '_call_count'):
        valid_move_reward._call_count = 0
    valid_move_reward._call_count += 1
    
    reward = 1.0 if moved else 0.1
    
    # Print debug for first 20 calls or any failed moves
    if not moved or valid_move_reward._call_count <= 20:
        print(f"REWARD DEBUG #{valid_move_reward._call_count}: move='{move_str}' dir={direction} moved={moved} reward={reward}", file=sys.stderr, flush=True)
        print(f"  old_cells={old_cells}", file=sys.stderr, flush=True)
        print(f"  new_cells={new_cells}", file=sys.stderr, flush=True)
    
    return reward


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
        super().__init__(**kwargs)
        self.grid_size = grid_size
        self.target_tile = target_tile
    
    async def setup_state(self, state: State) -> State:
        """Initialize game from dataset row."""
        import sys
        
        state = await super().setup_state(state)
        
        # Get info dict which contains our game data
        info = state["input"].get("info", {})
        if isinstance(info, str):
            import json
            info = json.loads(info)
        
        # Reconstruct game from stored grid
        grid_cells = info.get("grid")
        size = info.get("size", self.grid_size)
        target = info.get("target_tile", self.target_tile)
        
        # Debug: Print what's in info
        if not hasattr(self, '_setup_call_count'):
            self._setup_call_count = 0
        self._setup_call_count += 1
        if self._setup_call_count <= 5:
            print(f"SETUP DEBUG #{self._setup_call_count}: info.keys() = {list(info.keys()) if isinstance(info, dict) else 'not a dict'}", file=sys.stderr, flush=True)
            print(f"  grid_cells[:2] = {grid_cells[:2] if grid_cells else None}", file=sys.stderr, flush=True)
        
        if grid_cells is None:
            print(f"SETUP ERROR: grid_cells is None! Falling back to empty grid.", file=sys.stderr, flush=True)
            grid_cells = [[0]*size for _ in range(size)]  # fallback to empty grid
        
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
    **kwargs,
) -> vf.Environment:
    """Load the 2048 single-turn environment."""
    
    system_prompt = get_system_prompt(grid_size=grid_size, target_tile=target_tile)
    
    train_dataset = generate_single_turn_dataset(
        num_train_examples,
        size=grid_size,
        target_tile=target_tile,
        min_moves=min_moves,
        max_moves=max_moves,
        seed=seed,
        balanced_difficulty=balanced_difficulty,
    )
    eval_dataset = generate_single_turn_dataset(
        num_eval_examples,
        size=grid_size,
        target_tile=target_tile,
        min_moves=min_moves,
        max_moves=max_moves,
        seed=seed + 10000,
        balanced_difficulty=balanced_difficulty,
    )
    
    parser = vf.XMLParser(fields=["move"], answer_field="move")
    
    rubric = vf.Rubric(parser=parser)
    rubric.add_reward_func(valid_move_reward, weight=1.0)
    
    return Game2048SingleTurnEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        grid_size=grid_size,
        target_tile=target_tile,
    )
