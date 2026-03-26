"""Tests for oab_shared module — MatchedPool and action table."""

import pytest

from oab_shared import (
    NUM_ACTIONS,
    ACTION_TABLE,
    MatchedPool,
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


# ── Config ──

def test_config_defaults():
    from config import TrainConfig
    c = TrainConfig()
    assert c.learning_rate == 3e-4
    assert c.timesteps == 500_000
    assert c.lobby_size == 10


def test_config_load_save(tmp_path):
    from config import TrainConfig, load_config, save_config
    config = TrainConfig(learning_rate=1e-3, timesteps=100)
    path = tmp_path / "test_config.json"
    save_config(config, str(path))
    loaded = load_config(str(path))
    assert loaded.learning_rate == 1e-3
    assert loaded.timesteps == 100
    assert loaded.batch_size == 64


def test_config_partial_json(tmp_path):
    import json
    path = tmp_path / "partial.json"
    path.write_text(json.dumps({"learning_rate": 0.001}))
    from config import load_config
    config = load_config(str(path))
    assert config.learning_rate == 0.001
    assert config.batch_size == 64
