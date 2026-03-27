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


@pytest.fixture
def env():
    from env import OABEnv
    from oab_shared import MatchedPool
    pool = MatchedPool()
    e = OABEnv(set_id=0, board_pool=pool)
    yield e


def test_obs_dim_from_rust():
    session = oab_py.GameSession(0, 0)
    # Dynamic — depends on card set, just verify it's positive
    assert session.obs_dim() > 0
    assert oab_py.GameSession.num_actions() == 66


def test_reset_returns_correct_shape(env):
    obs, info = env.reset()
    session = oab_py.GameSession(0, env.set_id)
    assert obs.shape == (session.obs_dim(),)
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
    assert mask.shape == (66,)
    assert mask.dtype == bool
    assert mask[0] == True  # EndTurn always valid


def test_step_endturn_triggers_battle(env):
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    session = oab_py.GameSession(0, env.set_id)
    assert obs.shape == (session.obs_dim(),)
    assert isinstance(reward, float)
    assert "battle_result" in info


def test_full_game_terminates(env):
    env.reset()
    total_steps = 0
    max_steps = 5000

    while total_steps < max_steps:
        mask = env.action_masks()
        valid_actions = np.where(mask)[0]
        action = valid_actions[0]
        _, _, terminated, truncated, info = env.step(action)
        total_steps += 1
        if terminated or truncated:
            break

    assert terminated or truncated
    assert "game_result" in info


def test_board_pool_populated(env):
    pool = env.board_pool
    env.reset()
    for _ in range(5):
        _, _, terminated, _, _ = env.step(0)
        if terminated:
            env.reset()
    assert len(pool) > 0


def test_get_hand_names(env):
    env.reset()
    names = env.get_hand_names()
    assert isinstance(names, list)
    assert any(n is not None for n in names)


def test_phase_decomposition_adds_phase_features():
    from env import OABEnv
    from oab_shared import MatchedPool

    env = OABEnv(set_id=0, board_pool=MatchedPool(), phase_decomposition=True)
    obs, info = env.reset()
    session = oab_py.GameSession(0, env.set_id)

    assert obs.shape == (session.obs_dim() + 2,)
    assert info["phase"] == "shop"
    assert np.array_equal(obs[-2:], np.array([1.0, 0.0], dtype=np.float32))


def test_phase_decomposition_done_shopping_transitions_phase():
    from env import OABEnv
    from oab_shared import MatchedPool

    env = OABEnv(set_id=0, board_pool=MatchedPool(), phase_decomposition=True)
    env.reset(seed=0)

    obs, reward, terminated, truncated, info = env.step(0)

    assert reward == 0.0
    assert not terminated
    assert not truncated
    assert info["phase"] == "position"
    assert env.describe_action(0) == ("EndTurn", {})
    assert np.array_equal(obs[-2:], np.array([0.0, 1.0], dtype=np.float32))


def test_round_cap_forces_defeat():
    from env import OABEnv
    from oab_shared import MatchedPool

    env = OABEnv(set_id=0, board_pool=MatchedPool(), max_rounds=1)
    env.reset(seed=0)

    _, reward, terminated, truncated, info = env.step(0)

    assert terminated
    assert not truncated
    assert reward == -1.0
    assert info["game_result"] == "defeat"
    assert info["termination_reason"] == "max_rounds"


def test_end_turn_reports_board_reward_components():
    from env import OABEnv
    from oab_shared import MatchedPool

    env = OABEnv(
        set_id=0,
        board_pool=MatchedPool(),
        board_unit_reward=0.0,
        empty_board_penalty=-0.25,
    )
    env.reset(seed=0)

    _, reward, terminated, truncated, info = env.step(0)

    assert not truncated
    assert isinstance(reward, float)
    assert info["board_units"] == 0
    assert info["end_turn_reward"] == -0.25
