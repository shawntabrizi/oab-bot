"""Tests for oab_shared module — MatchedPool and action table."""

import pytest

from oab_shared import (
    NUM_ACTIONS,
    ACTION_TABLE,
    MatchedPool,
    PhaseController,
    POSITION_PHASE,
)


# ── Action Table ──

def test_action_count():
    assert NUM_ACTIONS == 66


def test_action_type_counts():
    counts = {}
    for action_type, _ in ACTION_TABLE:
        counts[action_type] = counts.get(action_type, 0) + 1
    assert counts["EndTurn"] == 1
    assert counts["BurnFromHand"] == 5
    assert counts["PlayFromHand"] == 25
    assert counts["BurnFromBoard"] == 5
    assert counts["SwapBoard"] == 10
    assert counts["MoveBoard"] == 20


def test_endturn_is_action_zero():
    assert ACTION_TABLE[0] == ("EndTurn", {})


# ── MatchedPool ──

def test_matched_pool_exact_match():
    pool = MatchedPool(max_per_bucket=10)
    board_a = [{"card_id": 1, "slot": 0}]
    board_b = [{"card_id": 2, "slot": 0}]
    pool.add(3, 2, 3, board_a)
    pool.add(3, 2, 3, board_b)
    pool.add(5, 0, 1, [{"card_id": 99, "slot": 0}])
    for _ in range(20):
        result = pool.sample(3, 2, 3)
        assert result in (board_a, board_b)


def test_matched_pool_fallback_close():
    pool = MatchedPool(max_per_bucket=10)
    board = [{"card_id": 5, "slot": 0}]
    pool.add(3, 4, 2, board)
    result = pool.sample(3, 3, 3)
    assert result == board


def test_matched_pool_fallback_same_round():
    pool = MatchedPool(max_per_bucket=10)
    board = [{"card_id": 5, "slot": 0}]
    pool.add(3, 8, 1, board)
    result = pool.sample(3, 0, 3)
    assert result == board


def test_matched_pool_fallback_any():
    pool = MatchedPool(max_per_bucket=10)
    board = [{"card_id": 5, "slot": 0}]
    pool.add(7, 5, 2, board)
    result = pool.sample(1, 0, 3)
    assert result == board


def test_matched_pool_empty_returns_empty():
    pool = MatchedPool()
    assert pool.sample(1, 0, 3) == []


def test_matched_pool_bucket_eviction():
    pool = MatchedPool(max_per_bucket=4)
    for i in range(10):
        pool.add(1, 0, 3, [{"card_id": i, "slot": 0}])
    assert len(pool) <= 4


def test_matched_pool_bucket_keeps_strong_board_archive():
    pool = MatchedPool(max_per_bucket=2)
    weak = [{"card_id": 1, "slot": 0, "attack": 1, "health": 1}]
    strong = [{"card_id": 2, "slot": 0, "attack": 12, "health": 12}]
    recent = [{"card_id": 3, "slot": 0, "attack": 4, "health": 4}]
    pool.add(3, 2, 3, weak)
    pool.add(3, 2, 3, strong)
    pool.add(3, 2, 3, recent)

    for _ in range(20):
        result = pool.sample(3, 2, 3)
        assert result in (strong, recent)


def test_matched_pool_deduplicates_identical_boards():
    pool = MatchedPool(max_per_bucket=10)
    board = [{"card_id": 7, "slot": 0}]
    pool.add(3, 2, 3, board)
    pool.add(3, 2, 3, [{"card_id": 7, "slot": 0}])
    assert len(pool) == 1


def test_matched_pool_excludes_identical_board_when_possible():
    pool = MatchedPool(max_per_bucket=10)
    board_a = [{"card_id": 1, "slot": 0}]
    board_b = [{"card_id": 2, "slot": 0}]
    pool.add(3, 2, 3, board_a)
    pool.add(3, 2, 3, board_b)

    for _ in range(20):
        result = pool.sample(3, 2, 3, exclude_board=board_a)
        assert result == board_b


def test_matched_pool_prefers_stronger_same_round():
    pool = MatchedPool(max_per_bucket=10)
    weaker = [{"card_id": 1, "slot": 0}]
    stronger = [{"card_id": 2, "slot": 0}]
    pool.add(4, 1, 2, weaker)
    pool.add(4, 4, 3, stronger)

    result = pool.sample(4, 2, 2)
    assert result == stronger


def test_matched_pool_challenge_probability_can_override_exact_match():
    pool = MatchedPool(max_per_bucket=10, challenge_probability=1.0)
    exact = [{"card_id": 1, "slot": 0}]
    stronger = [{"card_id": 2, "slot": 0}]
    pool.add(4, 2, 2, exact)
    pool.add(4, 4, 3, stronger)

    for _ in range(10):
        result = pool.sample(4, 2, 2)
        assert result == stronger


# ── Config ──

def test_config_defaults():
    from config import TrainConfig
    c = TrainConfig()
    assert c.learning_rate == 3e-4
    assert c.timesteps == 500_000
    assert c.lobby_size == 10
    assert c.play_reward == 0.05
    assert c.reorder_penalty == -0.05
    assert c.board_unit_reward == 0.03
    assert c.empty_board_penalty == -0.2
    assert c.challenge_probability == 0.35
    assert c.seed_games_per_model == 8
    assert c.seed_opponent_models == []
    assert c.phase_decomposition is False
    assert c.shop_action_limit == 10
    assert c.position_action_limit == 5
    assert c.max_rounds == 20


def test_config_load_save(tmp_path):
    from config import TrainConfig, load_config, save_config
    config = TrainConfig(
        learning_rate=1e-3,
        timesteps=100,
        play_reward=0.07,
        reorder_penalty=-0.03,
        board_unit_reward=0.04,
        empty_board_penalty=-0.25,
        challenge_probability=0.6,
        seed_games_per_model=12,
        seed_opponent_models=["models/a", "models/b"],
        phase_decomposition=True,
        shop_action_limit=7,
        position_action_limit=3,
        max_rounds=12,
    )
    path = tmp_path / "test_config.json"
    save_config(config, str(path))
    loaded = load_config(str(path))
    assert loaded.learning_rate == 1e-3
    assert loaded.timesteps == 100
    assert loaded.batch_size == 64
    assert loaded.play_reward == 0.07
    assert loaded.reorder_penalty == -0.03
    assert loaded.board_unit_reward == 0.04
    assert loaded.empty_board_penalty == -0.25
    assert loaded.challenge_probability == 0.6
    assert loaded.seed_games_per_model == 12
    assert loaded.seed_opponent_models == ["models/a", "models/b"]
    assert loaded.phase_decomposition is True
    assert loaded.shop_action_limit == 7
    assert loaded.position_action_limit == 3
    assert loaded.max_rounds == 12


def test_config_partial_json(tmp_path):
    import json
    path = tmp_path / "partial.json"
    path.write_text(json.dumps({"learning_rate": 0.001}))
    from config import load_config
    config = load_config(str(path))
    assert config.learning_rate == 0.001
    assert config.batch_size == 64


def test_phase_controller_masks_shop_vs_position():
    controller = PhaseController(enabled=True, shop_action_limit=10, position_action_limit=5)
    base_mask = [True] * NUM_ACTIONS

    shop_mask = controller.apply_mask(base_mask)
    assert all(shop_mask[:36])
    assert not any(shop_mask[36:])

    controller.advance_to_position()
    assert controller.phase == POSITION_PHASE

    position_mask = controller.apply_mask(base_mask)
    assert position_mask[0]
    assert not any(position_mask[1:36])
    assert all(position_mask[36:])


def test_phase_controller_done_shopping_semantics():
    controller = PhaseController(enabled=True)
    assert controller.describe_action(0) == ("DoneShopping", {})
    controller.advance_to_position()
    assert controller.describe_action(0) == ACTION_TABLE[0]
