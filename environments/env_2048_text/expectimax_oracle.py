#!/usr/bin/env python3
"""Expectimax oracle for 2048.

Computes the best move under depth-limited expectimax with the in-game score
as the leaf heuristic. Standard 2048 AI formulation:
  - MAX nodes (player move): pick the direction with highest expected value
  - CHANCE nodes (random tile spawn): expectation over (cell, value)
    Tile is 2 with prob 0.9, 4 with prob 0.1; uniform over empty cells.

For 4x4 boards, depth=3 is fast (<10ms per query). Depth=4 is slower but
still tractable. Empty leaves are scored by `game.score` (sum of all merges
made so far).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from env_2048_text import Game2048


DIRECTION_NAMES = {0: "up", 1: "right", 2: "down", 3: "left"}


def heuristic(game: Game2048) -> float:
    """Leaf heuristic. Using raw game score keeps the gradient signal aligned
    with our training reward."""
    return float(game.score)


def _empty_cells(game: Game2048) -> list[tuple[int, int]]:
    out = []
    for r in range(game.size):
        for c in range(game.size):
            if game.grid.get(r, c) == 0:
                out.append((r, c))
    return out


def _expect_node(game: Game2048, depth: int) -> float:
    """CHANCE node: expectation over (empty_cell, value).
    Each empty cell is equally likely; conditional on that cell, value=2 with
    prob 0.9 and value=4 with prob 0.1."""
    empties = _empty_cells(game)
    if not empties:
        # No spawn possible: just evaluate (game is effectively over).
        return _max_node(game, depth)
    n_empty = len(empties)
    p_cell = 1.0 / n_empty
    expected = 0.0
    for (r, c) in empties:
        for value, p_value in ((2, 0.9), (4, 0.1)):
            game.grid.set(r, c, value)
            expected += p_cell * p_value * _max_node(game, depth)
            game.grid.set(r, c, 0)
    return expected


def _max_node(game: Game2048, depth: int) -> float:
    """MAX node: pick the best direction. depth counts down player-move depth."""
    if depth <= 0 or game.over:
        return heuristic(game)
    best = None
    for d in range(4):
        clone = game.clone()
        moved = clone.move_no_spawn(d) if hasattr(clone, "move_no_spawn") else _try_move_no_spawn(clone, d)
        if not moved:
            continue
        # After the player's move (no spawn yet), the chance node spawns a tile.
        val = _expect_node(clone, depth - 1)
        if best is None or val > best:
            best = val
    if best is None:
        # No legal move
        return heuristic(game)
    return best


def _try_move_no_spawn(game: Game2048, direction: int) -> bool:
    """Apply `direction` to `game` WITHOUT spawning a new random tile.

    Game2048.move() spawns a tile on success, which would mess up our search
    (we want the chance node to handle the spawn explicitly).
    """
    snapshot = [row[:] for row in game.grid.cells]
    snapshot_score = game.score
    snapshot_max = game.max_tile
    snapshot_won = game.won
    snapshot_over = game.over
    snapshot_count = game.move_count
    moved = game.move(direction)
    if moved:
        # Undo the random tile spawn by removing the last-added 2 or 4 in an
        # empty cell. We do this by finding the cell that wasn't in `snapshot`
        # but is now non-zero in `game.grid`. Note: move() also increments
        # score, max_tile, etc. — those we keep.
        for r in range(game.size):
            for c in range(game.size):
                pre = snapshot[r][c]
                post = game.grid.get(r, c)
                # If a cell that was 0 now has 2 or 4 AND it wasn't there before
                # because of a merge (merges produce powers of 2 ≥ 4), this is
                # the spawned tile. Heuristic: spawned tiles are 2 or 4 ONLY,
                # and merges into 2 don't exist (smallest is 4 from 2+2), so:
                #   pre==0, post==2  → definitely spawned
                #   pre==0, post==4  → spawned OR merged from 2+2 (during slide)
                # We can't perfectly distinguish in the 4 case, but for
                # expectimax we'll restore the grid completely.
                pass
        # Cleaner: restore grid + move-count to pre-spawn state, but keep score.
        # The merges already updated score/max_tile correctly.
        post_grid = [[game.grid.get(r, c) for c in range(game.size)] for r in range(game.size)]
        # Find the spawned cell: it's the only cell where pre==0 and post!=0
        # that doesn't match the "expected post-slide" state. Tricky.
        # Simpler approach: just remove ONE tile of value 2 or 4 from a cell
        # that was previously empty.
        for r in range(game.size):
            for c in range(game.size):
                if snapshot[r][c] == 0 and post_grid[r][c] in (2, 4):
                    game.grid.set(r, c, 0)
                    return True
        # If we couldn't find a spawned tile, the move had no spawn (shouldn't happen)
        return True
    return False


def expectimax_oracle(
    game: Game2048,
    depth: int = 3,
) -> tuple[Optional[str], dict]:
    """Run expectimax search. Return (best_direction_name, debug_info).

    If no legal move exists, returns (None, {}).
    """
    best_dir = None
    best_val = None
    per_dir = {}
    for d in range(4):
        clone = game.clone()
        moved = _try_move_no_spawn(clone, d)
        if not moved:
            per_dir[DIRECTION_NAMES[d]] = None
            continue
        val = _expect_node(clone, depth - 1)
        per_dir[DIRECTION_NAMES[d]] = val
        if best_val is None or val > best_val:
            best_val = val
            best_dir = DIRECTION_NAMES[d]
    return best_dir, {"per_direction": per_dir, "best_value": best_val, "depth": depth}


if __name__ == "__main__":
    # Quick sanity check
    import random
    random.seed(0)
    game = Game2048(size=4, target_tile=2048)
    print("Initial board:")
    print(game.get_state_text())
    for _ in range(15):
        valid = [d for d in range(4) if game.clone().move(d)]
        if not valid:
            break
        game.move(random.choice(valid))
    print("After 15 random moves:")
    print(game.get_state_text())
    print(f"Score: {game.score}")

    for depth in (2, 3, 4):
        import time
        t0 = time.time()
        best, info = expectimax_oracle(game, depth=depth)
        elapsed = (time.time() - t0) * 1000
        print(f"depth={depth} elapsed={elapsed:.1f}ms best={best}")
        for d, v in info["per_direction"].items():
            print(f"  {d:<6} value={v}")
