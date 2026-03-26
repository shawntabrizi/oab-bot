"""Tests for OABEnv (requires oab_py native module).

These tests are skipped if oab_py is not available (not built via maturin).
"""

import numpy as np
import pytest

try:
    import oab_py
    HAS_OAB_PY = True
except ImportError:
    HAS_OAB_PY = False

pytestmark = pytest.mark.skipif(not HAS_OAB_PY, reason="oab_py not built")

from oab_shared import OBS_DIM, NUM_ACTIONS


@pytest.fixture
def env():
    from env import OABEnv, BoardPool
    pool = BoardPool(max_size=50)
    e = OABEnv(set_id=0, board_pool=pool)
    yield e


def test_reset_returns_correct_shape(env):
    obs, info = env.reset()
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert "round" in info
    assert "lives" in info
    assert "wins" in info


def test_obs_values_in_range(env):
    obs, _ = env.reset()
    assert np.all(obs >= 0.0)
    assert np.all(obs <= 1.0)


def test_action_masks_shape(env):
    env.reset()
    mask = env.action_masks()
    assert mask.shape == (NUM_ACTIONS,)
    assert mask.dtype == bool
    # EndTurn should always be valid
    assert mask[0] == True


def test_step_endturn_triggers_battle(env):
    obs, info = env.reset()
    # Action 0 = EndTurn
    obs2, reward, terminated, truncated, info2 = env.step(0)
    assert obs2.shape == (OBS_DIM,)
    # Reward should be -1, 0, or 1 (battle outcome)
    assert reward in (-1.0, 0.0, 1.0)
    assert "battle_result" in info2


def test_full_game_terminates(env):
    """Play a full game — should terminate within reasonable steps."""
    env.reset()
    total_steps = 0
    max_steps = 5000

    while total_steps < max_steps:
        mask = env.action_masks()
        # Pick first valid action
        valid_actions = np.where(mask)[0]
        action = valid_actions[0]  # EndTurn is always valid at index 0
        _, _, terminated, truncated, info = env.step(action)
        total_steps += 1
        if terminated or truncated:
            break

    assert terminated or truncated, f"Game did not terminate within {max_steps} steps"
    assert "game_result" in info


def test_board_pool_populated(env):
    """After playing some turns, the board pool should have entries."""
    from env import BoardPool
    pool = env.board_pool
    env.reset()
    # Play a few turns (EndTurn each time)
    for _ in range(5):
        _, _, terminated, _, _ = env.step(0)
        if terminated:
            env.reset()
    assert len(pool) > 0
