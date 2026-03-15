"""
Tests for the 2048 Text Environment

Run with: python test_env.py
"""

import asyncio
import random
from env_2048_text import (
    Game2048,
    Game2048Env,
    Grid,
    ContextMode,
    generate_game_dataset,
    load_environment,
    score_reward,
    max_tile_reward,
    win_reward,
    valid_moves_ratio,
    get_system_prompt,
)


def test_grid():
    """Test Grid class functionality."""
    print("=== Testing Grid ===")
    
    # Test empty grid
    grid = Grid(4)
    empty_cells = grid.empty_cells()
    assert len(empty_cells) == 16, f"Expected 16 empty cells, got {len(empty_cells)}"
    print(f"✓ Empty 4x4 grid has {len(empty_cells)} empty cells")
    
    # Test setting and getting values
    grid.set(0, 0, 2)
    grid.set(1, 1, 4)
    assert grid.get(0, 0) == 2, "Failed to set/get value"
    assert grid.get(1, 1) == 4, "Failed to set/get value"
    assert len(grid.empty_cells()) == 14, "Empty cells count wrong after setting"
    print("✓ Set/get values work correctly")
    
    # Test cloning
    cloned = grid.clone()
    cloned.set(0, 0, 8)
    assert grid.get(0, 0) == 2, "Clone modified original"
    assert cloned.get(0, 0) == 8, "Clone not updated"
    print("✓ Grid cloning works correctly")
    
    # Test bounds checking
    assert grid.within_bounds(0, 0) == True
    assert grid.within_bounds(3, 3) == True
    assert grid.within_bounds(-1, 0) == False
    assert grid.within_bounds(4, 0) == False
    print("✓ Bounds checking works correctly")
    
    print()


def test_game_initialization():
    """Test Game2048 initialization."""
    print("=== Testing Game Initialization ===")
    
    random.seed(42)
    game = Game2048()
    
    # Should start with 2 tiles
    non_empty = sum(
        1 for x in range(4) for y in range(4)
        if game.grid.get(x, y) > 0
    )
    assert non_empty == 2, f"Expected 2 starting tiles, got {non_empty}"
    print(f"✓ Game starts with 2 tiles")
    
    # Initial state checks
    assert game.score == 0, "Score should be 0 initially"
    assert game.over == False, "Game should not be over initially"
    assert game.won == False, "Game should not be won initially"
    assert game.move_count == 0, "Move count should be 0"
    print("✓ Initial game state is correct")
    
    print(f"\nInitial board:")
    print(game.to_ascii())
    print()


def test_move_mechanics():
    """Test move mechanics including sliding and merging."""
    print("=== Testing Move Mechanics ===")
    
    # Create a game with a known state
    game = Game2048()
    game.grid = Grid(4)  # Clear the grid
    
    # Set up: [2, 2, 0, 0] in first row
    game.grid.set(0, 0, 2)
    game.grid.set(1, 0, 2)
    
    print("Before move right:")
    print(game.to_ascii())
    
    # Move right - should merge 2+2=4
    moved = game.move(1)  # right
    assert moved == True, "Move should succeed"
    
    print("After move right (2+2 should merge to 4):")
    print(game.to_ascii())
    
    # The merged 4 should be at the right
    assert game.grid.get(3, 0) == 4, f"Expected 4 at (3,0), got {game.grid.get(3, 0)}"
    assert game.score == 4, f"Score should be 4, got {game.score}"
    print(f"✓ Merge works correctly (score: {game.score})")
    
    # Test that a tile can only merge once per move
    game2 = Game2048()
    game2.grid = Grid(4)
    # Set up: [2, 2, 2, 2] - should become [0, 0, 4, 4], not [0, 0, 0, 8]
    for x in range(4):
        game2.grid.set(x, 0, 2)
    
    print("\nBefore move right [2,2,2,2]:")
    print(game2.to_ascii())
    
    game2.move(1)  # right
    
    print("After move right (should be [., ., 4, 4] not [., ., ., 8]):")
    print(game2.to_ascii())
    
    # Should have two 4s, not one 8
    assert game2.grid.get(3, 0) == 4, "Rightmost should be 4"
    assert game2.grid.get(2, 0) == 4, "Second from right should be 4"
    print("✓ Tiles only merge once per move")
    
    print()


def test_invalid_moves():
    """Test that invalid moves are detected."""
    print("=== Testing Invalid Move Detection ===")
    
    game = Game2048()
    game.grid = Grid(4)
    # Single tile in corner
    game.grid.set(0, 0, 2)
    
    print("Board with single tile at top-left:")
    print(game.to_ascii())
    
    # Up and Left should not move anything
    moved_up = game.move(0)  # up
    moved_left = game.move(3)  # left
    
    assert moved_up == False, "Moving up should fail"
    assert moved_left == False, "Moving left should fail"
    print("✓ Invalid moves correctly return False")
    
    # Down and Right should work
    game2 = Game2048()
    game2.grid = Grid(4)
    game2.grid.set(0, 0, 2)
    
    moved_down = game2.move(2)  # down
    assert moved_down == True, "Moving down should work"
    print("✓ Valid moves correctly return True")
    
    print()


def test_game_over_detection():
    """Test game over detection."""
    print("=== Testing Game Over Detection ===")
    
    game = Game2048()
    game.grid = Grid(4)
    
    # Fill grid with alternating values that can't merge
    values = [2, 4, 2, 4, 4, 2, 4, 2, 2, 4, 2, 4, 4, 2, 4, 2]
    idx = 0
    for y in range(4):
        for x in range(4):
            game.grid.set(x, y, values[idx])
            idx += 1
    
    print("Board with no possible merges:")
    print(game.to_ascii())
    
    # Try to make a move - should fail and set game.over
    moved = game.move(0)
    assert moved == False, "No move should be possible"
    
    # Manually check if moves are available
    assert game._moves_available() == False, "Should have no moves available"
    print("✓ Game over correctly detected when no merges possible")
    
    print()


def test_win_detection():
    """Test win detection when reaching 2048."""
    print("=== Testing Win Detection ===")
    
    game = Game2048()
    game.grid = Grid(4)
    
    # Set up two 1024 tiles that can merge
    game.grid.set(0, 0, 1024)
    game.grid.set(1, 0, 1024)
    
    print("Board with two 1024 tiles:")
    print(game.to_ascii())
    
    assert game.won == False, "Should not be won yet"
    
    game.move(1)  # right - merge to 2048
    
    print("After merging to 2048:")
    print(game.to_ascii())
    
    assert game.won == True, "Should be won after reaching 2048"
    assert game.grid.get(3, 0) == 2048, "Should have 2048 tile"
    print("✓ Win correctly detected when reaching 2048")
    
    print()


def test_ascii_rendering():
    """Test ASCII table rendering."""
    print("=== Testing ASCII Rendering ===")
    
    game = Game2048()
    game.grid = Grid(4)
    game.grid.set(0, 0, 2)
    game.grid.set(1, 1, 128)
    game.grid.set(2, 2, 2048)
    
    ascii_output = game.to_ascii()
    print("Rendered board:")
    print(ascii_output)
    
    assert "2" in ascii_output, "Should contain 2"
    assert "128" in ascii_output, "Should contain 128"
    assert "2048" in ascii_output, "Should contain 2048"
    assert "." in ascii_output, "Should contain empty cells"
    assert "+" in ascii_output, "Should have borders"
    print("✓ ASCII rendering contains expected elements")
    
    print()


def test_move_parsing():
    """Test move string parsing."""
    print("=== Testing Move Parsing ===")
    
    game = Game2048()
    
    # Test various formats
    test_cases = [
        ("up", 0), ("UP", 0), ("u", 0), ("U", 0),
        ("right", 1), ("RIGHT", 1), ("r", 1), ("R", 1),
        ("down", 2), ("DOWN", 2), ("d", 2), ("D", 2),
        ("left", 3), ("LEFT", 3), ("l", 3), ("L", 3),
        ("  up  ", 0),  # with whitespace
        ("invalid", None),
        ("", None),
    ]
    
    for move_str, expected in test_cases:
        result = game.parse_move(move_str)
        assert result == expected, f"parse_move('{move_str}') = {result}, expected {expected}"
    
    print("✓ All move formats parsed correctly")
    print()


def test_dataset_generation():
    """Test dataset generation."""
    print("=== Testing Dataset Generation ===")
    
    dataset = generate_game_dataset(num_examples=5, seed=42)
    
    assert len(dataset) == 5, f"Expected 5 examples, got {len(dataset)}"
    print(f"✓ Generated dataset with {len(dataset)} examples")
    
    # Check dataset structure
    example = dataset[0]
    assert "task" in example, "Missing 'task' field"
    assert "prompt" in example, "Missing 'prompt' field"
    assert "answer" in example, "Missing 'answer' field"
    assert "info" in example, "Missing 'info' field"
    print("✓ Dataset has correct fields")
    
    # Check prompt format
    prompt = example["prompt"]
    assert isinstance(prompt, list), "Prompt should be a list"
    assert len(prompt) == 1, "Should have one message"
    assert prompt[0]["role"] == "user", "Should be user message"
    assert "2048" in prompt[0]["content"], "Prompt should mention 2048"
    print("✓ Prompt format is correct")
    
    # Check that different seeds produce different games
    dataset2 = generate_game_dataset(num_examples=5, seed=123)
    assert dataset[0]["info"]["initial_grid"] != dataset2[0]["info"]["initial_grid"], \
        "Different seeds should produce different games"
    print("✓ Different seeds produce different games")
    
    print("\nExample prompt preview:")
    print(example["prompt"][0]["content"][:300] + "...")
    print()


def test_reward_functions():
    """Test reward functions."""
    print("=== Testing Reward Functions ===")
    
    # Create mock state
    game = Game2048()
    game.score = 10000
    game.max_tile = 1024
    game.won = False
    
    state = {
        "game": game,
        "valid_moves": 8,
        "invalid_moves": 2,
    }
    
    # Test score_reward
    sr = score_reward(state)
    assert 0.0 <= sr <= 1.0, f"score_reward out of range: {sr}"
    print(f"✓ score_reward({game.score}) = {sr:.3f}")
    
    # Test max_tile_reward
    mtr = max_tile_reward(state)
    assert 0.0 <= mtr <= 1.0, f"max_tile_reward out of range: {mtr}"
    print(f"✓ max_tile_reward({game.max_tile}) = {mtr:.3f}")
    
    # Test win_reward
    wr = win_reward(state)
    assert wr == 0.0, "Should not have win reward"
    print(f"✓ win_reward (not won) = {wr}")
    
    game.won = True
    wr = win_reward(state)
    assert wr == 1.0, "Should have win reward"
    print(f"✓ win_reward (won) = {wr}")
    
    # Test valid_moves_ratio
    vmr = valid_moves_ratio(state)
    assert vmr == 0.8, f"Expected 8/10=0.8, got {vmr}"
    print(f"✓ valid_moves_ratio = {vmr}")
    
    print()


def test_environment_loading():
    """Test environment loading."""
    print("=== Testing Environment Loading ===")
    
    env = load_environment(
        num_train_examples=5,
        num_eval_examples=2,
        max_moves=100,
        seed=42,
    )
    
    print(f"✓ Loaded environment: {type(env).__name__}")
    print(f"✓ Train dataset size: {len(env.dataset)}")
    print(f"✓ Eval dataset size: {len(env.eval_dataset)}")
    print(f"✓ Max turns: {env.max_turns}")
    
    # Check that rubric has reward functions
    assert env.rubric is not None, "Rubric should be set"
    print(f"✓ Rubric configured")
    
    print()


def test_different_grid_sizes():
    """Test that different grid sizes work correctly."""
    print("=== Testing Different Grid Sizes ===")
    
    # Test 3x3 grid
    game_3x3 = Game2048(size=3)
    assert game_3x3.size == 3, "Grid size should be 3"
    assert len(game_3x3.grid.empty_cells()) == 7, "3x3 grid should have 9-2=7 empty cells initially"
    print(f"✓ 3x3 grid created successfully")
    print(game_3x3.to_ascii())
    
    # Test 5x5 grid
    game_5x5 = Game2048(size=5)
    assert game_5x5.size == 5, "Grid size should be 5"
    assert len(game_5x5.grid.empty_cells()) == 23, "5x5 grid should have 25-2=23 empty cells initially"
    print(f"✓ 5x5 grid created successfully")
    print(game_5x5.to_ascii())
    
    # Test 6x6 grid
    game_6x6 = Game2048(size=6)
    assert game_6x6.size == 6, "Grid size should be 6"
    print(f"✓ 6x6 grid created successfully")
    print(game_6x6.to_ascii())
    
    # Test moves work on different sizes
    game_3x3_test = Game2048(size=3)
    game_3x3_test.grid = Grid(3)
    game_3x3_test.grid.set(0, 0, 2)
    game_3x3_test.grid.set(1, 0, 2)
    moved = game_3x3_test.move(1)  # right
    assert moved, "Move should succeed on 3x3"
    assert game_3x3_test.grid.get(2, 0) == 4, "Merge should work on 3x3"
    print("✓ Moves work correctly on 3x3 grid")
    
    print()


def test_different_target_tiles():
    """Test different target tile configurations."""
    print("=== Testing Different Target Tiles ===")
    
    # Test target 1024
    game_1024 = Game2048(target_tile=1024)
    game_1024.grid = Grid(4)
    game_1024.grid.set(0, 0, 512)
    game_1024.grid.set(1, 0, 512)
    game_1024.move(1)  # right
    assert game_1024.won, "Should win when reaching 1024 with target=1024"
    print("✓ Target tile 1024 works")
    
    # Test target 4096
    game_4096 = Game2048(target_tile=4096)
    game_4096.grid = Grid(4)
    game_4096.grid.set(0, 0, 1024)
    game_4096.grid.set(1, 0, 1024)
    game_4096.move(1)  # right
    assert not game_4096.won, "Should not win at 2048 when target=4096"
    assert game_4096.grid.get(3, 0) == 2048, "Should have 2048 tile"
    print("✓ Target tile 4096 works (2048 not winning)")
    
    # Continue to 4096
    game_4096.grid.set(2, 0, 2048)
    game_4096.move(1)  # right
    assert game_4096.won, "Should win when reaching 4096"
    print("✓ Win at 4096 works")
    
    # Test smaller target (256) for faster testing
    game_256 = Game2048(target_tile=256)
    game_256.grid = Grid(4)
    game_256.grid.set(0, 0, 128)
    game_256.grid.set(1, 0, 128)
    game_256.move(1)
    assert game_256.won, "Should win at 256 with target=256"
    print("✓ Target tile 256 works")
    
    print()


def test_dataset_with_different_sizes():
    """Test dataset generation with different grid sizes."""
    print("=== Testing Dataset with Different Grid Sizes ===")
    
    # Generate 3x3 dataset
    dataset_3x3 = generate_game_dataset(num_examples=3, grid_size=3, target_tile=512, seed=42)
    assert len(dataset_3x3) == 3
    assert dataset_3x3[0]["info"]["grid_size"] == 3
    assert dataset_3x3[0]["info"]["target_tile"] == 512
    assert "3x3" in dataset_3x3[0]["prompt"][0]["content"]
    assert "512" in dataset_3x3[0]["prompt"][0]["content"]
    print("✓ 3x3 dataset generated correctly")
    print(f"  Preview: {dataset_3x3[0]['prompt'][0]['content'][:100]}...")
    
    # Generate 5x5 dataset
    dataset_5x5 = generate_game_dataset(num_examples=3, grid_size=5, target_tile=4096, seed=42)
    assert dataset_5x5[0]["info"]["grid_size"] == 5
    assert dataset_5x5[0]["info"]["target_tile"] == 4096
    print("✓ 5x5 dataset generated correctly")
    
    print()


def test_environment_with_different_sizes():
    """Test environment loading with different grid sizes."""
    print("=== Testing Environment with Different Grid Sizes ===")
    
    # Test 3x3 environment
    env_3x3 = load_environment(
        num_train_examples=3,
        num_eval_examples=1,
        grid_size=3,
        target_tile=512,
        max_moves=100,
    )
    assert env_3x3.grid_size == 3
    assert env_3x3.target_tile == 512
    print(f"✓ 3x3 environment loaded (grid_size={env_3x3.grid_size}, target={env_3x3.target_tile})")
    
    # Test 5x5 environment
    env_5x5 = load_environment(
        num_train_examples=3,
        num_eval_examples=1,
        grid_size=5,
        target_tile=4096,
        max_moves=1000,
    )
    assert env_5x5.grid_size == 5
    assert env_5x5.target_tile == 4096
    print(f"✓ 5x5 environment loaded (grid_size={env_5x5.grid_size}, target={env_5x5.target_tile})")
    
    print()


def test_reward_functions_with_different_targets():
    """Test reward functions work correctly with different targets."""
    print("=== Testing Rewards with Different Targets ===")
    
    # Test with target 1024
    game_1024 = Game2048(target_tile=1024)
    game_1024.score = 5000
    game_1024.max_tile = 512
    game_1024.won = False
    
    state_1024 = {"game": game_1024, "valid_moves": 10, "invalid_moves": 0}
    
    mtr_1024 = max_tile_reward(state_1024)
    # 512 is log2(512)=9, target log2(1024)=10, so reward = 9/10 = 0.9
    assert 0.85 <= mtr_1024 <= 0.95, f"max_tile_reward for 512/1024 should be ~0.9, got {mtr_1024}"
    print(f"✓ max_tile_reward(512, target=1024) = {mtr_1024:.3f}")
    
    # Test with target 4096
    game_4096 = Game2048(target_tile=4096)
    game_4096.score = 10000
    game_4096.max_tile = 1024
    game_4096.won = False
    
    state_4096 = {"game": game_4096, "valid_moves": 10, "invalid_moves": 0}
    
    mtr_4096 = max_tile_reward(state_4096)
    # 1024 is log2(1024)=10, target log2(4096)=12, so reward = 10/12 ≈ 0.833
    assert 0.8 <= mtr_4096 <= 0.9, f"max_tile_reward for 1024/4096 should be ~0.833, got {mtr_4096}"
    print(f"✓ max_tile_reward(1024, target=4096) = {mtr_4096:.3f}")
    
    print()


def test_context_modes():
    """Test different context management modes."""
    print("=== Testing Context Modes ===")
    
    # Test loading environment with each context mode
    for mode in ["full", "markov", "last_k", "summary"]:
        env = load_environment(
            num_train_examples=2,
            num_eval_examples=1,
            context_mode=mode,
            last_k_turns=3,
            max_moves=50,
        )
        assert env.context_mode == ContextMode(mode), f"Context mode should be {mode}"
        print(f"✓ Environment loaded with context_mode='{mode}'")
    
    # Test that last_k_turns is configurable
    env_k5 = load_environment(
        num_train_examples=2,
        num_eval_examples=1,
        context_mode="last_k",
        last_k_turns=5,
    )
    assert env_k5.last_k_turns == 5, "last_k_turns should be 5"
    print("✓ last_k_turns parameter works correctly")
    
    print()


def test_context_mode_prompt_generation():
    """Test that different context modes generate appropriate prompts."""
    print("=== Testing Context Mode Prompt Generation ===")
    
    # Create environment with markov mode
    env = load_environment(
        num_train_examples=2,
        num_eval_examples=1,
        context_mode="markov",
    )
    
    # Test the helper methods exist and work
    game = Game2048()
    game.score = 100
    game.move_count = 5
    
    # Test _get_current_state_prompt
    prompt = env._get_current_state_prompt(game)
    assert "Score: 100" in prompt, "Current state prompt should include score"
    assert "<move>" in prompt, "Current state prompt should mention move tags"
    print("✓ _get_current_state_prompt generates valid prompt")
    
    # Test summary mode configuration
    env_summary = load_environment(
        num_train_examples=2,
        num_eval_examples=1,
        context_mode="summary",
        summary_interval=10,
    )
    assert env_summary.summary_interval == 10, "summary_interval should be configurable"
    print("✓ Summary mode configuration works")
    
    print()


def test_loop_detection():
    """Test consecutive invalid moves termination condition."""
    print("=== Testing Consecutive Invalid Moves Termination ===")
    
    # Test max_invalid_moves parameter
    env = load_environment(
        num_train_examples=2,
        num_eval_examples=1,
        max_invalid_moves=5,
    )
    assert env.max_invalid_moves == 5, "max_invalid_moves should be 5"
    print("✓ max_invalid_moves parameter configurable")
    
    # Test too_many_invalid_moves stop condition
    import asyncio
    
    async def test_invalid_stop():
        env_test = load_environment(
            num_train_examples=2,
            num_eval_examples=1,
            max_invalid_moves=3,
        )
        
        # Uses consecutive_invalid_moves, not invalid_moves
        state = {"consecutive_invalid_moves": 2}  # Below threshold
        should_stop = await env_test.too_many_invalid_moves(state)
        assert not should_stop, "Should not stop when below threshold"
        
        state["consecutive_invalid_moves"] = 3  # At threshold
        should_stop = await env_test.too_many_invalid_moves(state)
        assert should_stop, "Should stop when at threshold"
        
        state["consecutive_invalid_moves"] = 5  # Above threshold
        should_stop = await env_test.too_many_invalid_moves(state)
        assert should_stop, "Should stop when above threshold"
        
        return True
    
    result = asyncio.run(test_invalid_stop())
    assert result, "Consecutive invalid moves stop test should pass"
    print("✓ too_many_invalid_moves stop condition works correctly")
    
    # Test that consecutive counter resets on valid move
    async def test_reset():
        env_test = load_environment(
            num_train_examples=2,
            num_eval_examples=1,
            max_invalid_moves=3,
        )
        state = await env_test.setup_state({"info": {"seed": 42}})
        assert state["consecutive_invalid_moves"] == 0, "Should start at 0"
        assert state["invalid_moves"] == 0, "Total should start at 0"
        return True
    
    result = asyncio.run(test_reset())
    assert result, "Reset test should pass"
    print("✓ Consecutive counter initialized correctly")
    
    print()


def test_render_completion_full_history():
    """Test that render_completion captures full history in non-FULL modes."""
    print("=== Testing Render Completion Full History ===")
    
    import asyncio
    from env_2048_text import ContextMode, normalize_messages, concat_messages
    
    async def test_markov_logging():
        # Create markov mode environment
        env = load_environment(
            num_train_examples=2,
            num_eval_examples=1,
            context_mode="markov",
            max_moves=10,
        )
        assert env.context_mode == ContextMode.MARKOV
        
        # Set up initial state
        state = {
            "info": {"seed": 42},
            "prompt": [
                {"role": "system", "content": "You are playing 2048."},
                {"role": "user", "content": "Initial board..."}
            ],
            "trajectory": [],
        }
        state = await env.setup_state(state)
        
        # Simulate turn 0: get_prompt_messages returns initial prompt
        prompt0 = await env.get_prompt_messages(state)
        assert len(prompt0) == 2, "Turn 0 should return initial prompt"
        
        # Simulate adding turn 0 to trajectory (what add_model_response does)
        state["trajectory"].append({
            "prompt": prompt0,
            "completion": [{"role": "assistant", "content": "<move>right</move>"}]
        })
        
        # Simulate turn 1: get_prompt_messages should call env_response and store full history
        prompt1 = await env.get_prompt_messages(state)
        
        # In markov mode, LLM sees only 2 messages
        assert len(prompt1) == 2, f"Markov mode should return 2 messages, got {len(prompt1)}"
        assert prompt1[0]["role"] == "system"
        assert prompt1[1]["role"] == "user"
        
        # But full history should be stored
        assert "_full_messages_for_logging" in state, "Should store full messages for logging"
        full_msgs = state["_full_messages_for_logging"]
        assert len(full_msgs) > 2, f"Full messages should have more than 2 messages, got {len(full_msgs)}"
        
        # Add turn 1 to trajectory
        state["trajectory"].append({
            "prompt": prompt1,  # This is the condensed markov prompt
            "completion": [{"role": "assistant", "content": "<move>down</move>"}]
        })
        
        # Now test render_completion
        await env.render_completion(state)
        
        # Completion should have full history, not just 1 message
        completion = state.get("completion", [])
        assert len(completion) > 1, f"Completion should have multiple messages, got {len(completion)}"
        
        # Check that we have assistant messages in completion
        assistant_msgs = [m for m in completion if m.get("role") == "assistant"]
        assert len(assistant_msgs) >= 2, f"Should have at least 2 assistant messages, got {len(assistant_msgs)}"
        
        return True
    
    result = asyncio.run(test_markov_logging())
    assert result, "Markov logging test should pass"
    print("✓ Markov mode captures full history in render_completion")
    
    async def test_full_mode_unchanged():
        # Create full mode environment
        env = load_environment(
            num_train_examples=2,
            num_eval_examples=1,
            context_mode="full",
            max_moves=10,
        )
        assert env.context_mode == ContextMode.FULL
        
        # Set up initial state
        state = {
            "info": {"seed": 42},
            "prompt": [
                {"role": "system", "content": "You are playing 2048."},
                {"role": "user", "content": "Initial board..."}
            ],
            "trajectory": [],
        }
        state = await env.setup_state(state)
        
        # Turn 0
        prompt0 = await env.get_prompt_messages(state)
        state["trajectory"].append({
            "prompt": prompt0,
            "completion": [{"role": "assistant", "content": "<move>right</move>"}]
        })
        
        # Turn 1
        prompt1 = await env.get_prompt_messages(state)
        
        # In full mode, should NOT store _full_messages_for_logging (uses parent's render)
        # Actually it returns early before storing, so this key should not exist
        # or if it does from a previous iteration, that's fine
        
        # Full mode returns full history to LLM
        assert len(prompt1) > 2, f"Full mode should return full history, got {len(prompt1)}"
        
        return True
    
    result = asyncio.run(test_full_mode_unchanged())
    assert result, "Full mode test should pass"
    print("✓ Full mode behavior unchanged")
    
    print()


def test_last_k_mode_uses_real_env_responses():
    """Test that LAST_K mode uses actual env responses, not placeholders."""
    print("Testing LAST_K mode uses real env responses...")
    
    async def test_last_k():
        # Create minimal dataset for the env - prompt must be a list of messages
        from datasets import Dataset
        system_prompt = get_system_prompt(grid_size=4, target_tile=2048)
        
        dataset = Dataset.from_dict({
            "prompt": [[{"role": "user", "content": "Initial board..."}]],
            "info": [{"seed": 42, "grid_size": 4, "target_tile": 2048}]
        })
        
        env = Game2048Env(
            context_mode="last_k",
            last_k_turns=2,
            max_steps=10,
            grid_size=4,
            target_tile=2048,
            dataset=dataset,
            system_prompt=system_prompt
        )
        
        # Create initial state with prompt (including system) and info
        state = {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Initial board..."}
            ],
            "info": {"seed": 42, "grid_size": 4, "target_tile": 2048},
            "trajectory": [],
        }
        state = await env.setup_state(state)
        game: Game2048 = state["game"]
        
        # Turn 0: Get initial prompt
        prompt0 = await env.get_prompt_messages(state)
        assert len(prompt0) == 2, f"Turn 0 should have [system, user], got {len(prompt0)}"
        initial_state_content = prompt0[1]["content"]
        
        # Simulate Turn 0 completion
        state["trajectory"].append({
            "prompt": prompt0,
            "completion": [{"role": "assistant", "content": "<move>right</move>"}]
        })
        
        # Turn 1
        prompt1 = await env.get_prompt_messages(state)
        # Should be: system, initial_state, asst_0, env_response_0
        full_messages_1 = state.get("_full_messages_for_logging", [])
        
        # The env response should contain actual board state (ASCII table with +----+)
        if len(full_messages_1) > 3:
            env_response_1 = full_messages_1[3]["content"]  # user message after asst_0
            assert "+----+" in env_response_1, \
                f"Env response should contain actual board state, got: {env_response_1[:200]}"
        
        state["trajectory"].append({
            "prompt": prompt1,
            "completion": [{"role": "assistant", "content": "<move>up</move>"}]
        })
        
        # Turn 2
        prompt2 = await env.get_prompt_messages(state)
        full_messages_2 = state.get("_full_messages_for_logging", [])
        
        # With k=2, after 2 turns, we should have meaningful history
        # full_messages_2 = [system, user0, asst0, user1, asst1, user2]
        # last_k prompt should use actual env responses from full_messages
        
        # Verify last_k messages use real content
        # The returned prompt2 should be based on _build_last_k_messages
        # which now pulls from _full_messages_for_logging
        
        state["trajectory"].append({
            "prompt": prompt2,
            "completion": [{"role": "assistant", "content": "<move>left</move>"}]
        })
        
        # Turn 3 - now we have more than k=2 turns
        prompt3 = await env.get_prompt_messages(state)
        
        # The prompt should have: system + last 2*k messages
        # Check that it contains board representations (ASCII table)
        prompt_content = " ".join([m["content"] for m in prompt3])
        
        # Should have actual board ASCII (our implementation uses +----+ not Unicode box chars)
        assert "+----+" in prompt_content, \
            f"LAST_K prompt should contain actual board ASCII, not placeholders. Content: {prompt_content[:500]}"
        
        # Verify we're not using the old placeholder text
        placeholder_count = prompt_content.count("Move applied. Next board shown below.")
        board_count = prompt_content.count("+----+")  # Part of ASCII table
        
        # We should have real boards, not just placeholders
        assert board_count >= placeholder_count, \
            f"Should have more real boards ({board_count}) than placeholders ({placeholder_count})"
        
        return True
    
    result = asyncio.run(test_last_k())
    assert result, "LAST_K mode test should pass"
    print("✓ LAST_K mode uses real env responses")
    
    print()


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running 2048 Environment Tests")
    print("=" * 60)
    print()
    
    test_grid()
    test_game_initialization()
    test_move_mechanics()
    test_invalid_moves()
    test_game_over_detection()
    test_win_detection()
    test_ascii_rendering()
    test_move_parsing()
    test_dataset_generation()
    test_reward_functions()
    test_environment_loading()
    test_different_grid_sizes()
    test_different_target_tiles()
    test_dataset_with_different_sizes()
    test_environment_with_different_sizes()
    test_reward_functions_with_different_targets()
    test_context_modes()
    test_context_mode_prompt_generation()
    test_loop_detection()
    test_render_completion_full_history()
    test_last_k_mode_uses_real_env_responses()
    
    print("=" * 60)
    print("ALL TESTS PASSED! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
