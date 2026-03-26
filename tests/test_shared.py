"""Tests for oab_shared module — action table, normalization, obs encoding, masking."""

import numpy as np
import pytest

from oab_shared import (
    HAND_SIZE,
    BOARD_SIZE,
    OBS_DIM,
    NUM_ACTIONS,
    ACTION_TABLE,
    DEFAULT_MAX_ACTIONS_PER_TURN,
    GameStateTracker,
    norm,
    norm_id,
    extract_card_id,
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
    assert counts["PlayFromHand"] == 25  # 5 hand * 5 board
    assert counts["BurnFromBoard"] == 5
    assert counts["SwapBoard"] == 10  # C(5,2)
    assert counts["MoveBoard"] == 20  # 5*4 ordered pairs


def test_endturn_is_action_zero():
    assert ACTION_TABLE[0] == ("EndTurn", {})


# ── Normalization Helpers ──

def test_norm_basic():
    assert norm(5, 10) == 0.5
    assert norm(0, 10) == 0.0
    assert norm(10, 10) == 1.0


def test_norm_clamps():
    assert norm(15, 10) == 1.0
    assert norm(100, 10) == 1.0


def test_norm_zero_max():
    assert norm(5, 0) == 0.0


def test_norm_id_int():
    assert norm_id(50) == pytest.approx(0.5)
    assert norm_id(0) == 0.0


def test_norm_id_dict():
    assert norm_id({"SomeVariant": 50}) == pytest.approx(0.5)


def test_norm_id_empty_dict():
    assert norm_id({}) == 0.0


def test_extract_card_id_int():
    assert extract_card_id({"id": 42}) == 42


def test_extract_card_id_dict():
    assert extract_card_id({"id": {"Variant": 42}}) == 42


def test_extract_card_id_missing():
    assert extract_card_id({}) == 0


# ── Mock State Helper ──

def _make_state(hand=None, board=None, mana=5, mana_limit=10, round_num=1,
                lives=3, wins=0, bag_count=45):
    """Build a minimal game state dict for testing."""
    if hand is None:
        hand = [None] * HAND_SIZE
    if board is None:
        board = [None] * BOARD_SIZE
    return {
        "hand": hand,
        "board": board,
        "mana": mana,
        "mana_limit": mana_limit,
        "round": round_num,
        "lives": lives,
        "wins": wins,
        "bag_count": bag_count,
    }


def _make_card(card_id=1, attack=3, health=4, play_cost=2, burn_value=1):
    """Build a minimal card dict."""
    return {
        "id": card_id,
        "name": f"Card{card_id}",
        "attack": attack,
        "health": health,
        "play_cost": play_cost,
        "burn_value": burn_value,
        "battle_abilities": [],
        "shop_abilities": [],
    }


# ── Observation Encoding ──

def test_encode_obs_shape():
    tracker = GameStateTracker()
    tracker.sync(_make_state())
    obs = tracker.encode_obs()
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32


def test_encode_obs_empty_state_all_zeros_or_valid():
    tracker = GameStateTracker()
    tracker.sync(_make_state(mana=0, mana_limit=0))
    obs = tracker.encode_obs()
    assert np.all(obs >= 0.0)
    assert np.all(obs <= 1.0)


def test_encode_obs_with_cards():
    card = _make_card(card_id=10, attack=5, health=10, play_cost=3, burn_value=2)
    hand = [card] + [None] * (HAND_SIZE - 1)
    board = [card.copy()] + [None] * (BOARD_SIZE - 1)
    tracker = GameStateTracker()
    tracker.sync(_make_state(hand=hand, board=board, mana=5))
    obs = tracker.encode_obs()
    assert obs.shape == (OBS_DIM,)
    assert np.all(obs >= 0.0)
    assert np.all(obs <= 1.0)
    # First hand slot should have presence=1.0
    assert obs[0] == 1.0
    # Second hand slot should have presence=0.0
    assert obs[9] == 0.0  # HAND_FEATURES = 9


# ── Action Masking ──

def test_mask_empty_state():
    """Empty hand and board: only EndTurn should be valid."""
    tracker = GameStateTracker()
    tracker.sync(_make_state())
    mask = tracker.action_masks()
    assert mask[0] == True  # EndTurn
    # All BurnFromHand should be masked (no cards in hand)
    for i in range(1, 6):
        assert mask[i] == False
    # All PlayFromHand should be masked
    for i in range(6, 31):
        assert mask[i] == False
    # All BurnFromBoard should be masked
    for i in range(31, 36):
        assert mask[i] == False
    # All SwapBoard should be masked (need 2 units)
    for i in range(36, 46):
        assert mask[i] == False
    # All MoveBoard should be masked (need units at both slots)
    for i in range(46, 66):
        assert mask[i] == False


def test_mask_with_hand_card():
    """Card in hand[0] with enough mana: BurnFromHand[0] and PlayFromHand[0,*] valid."""
    card = _make_card(play_cost=2)
    hand = [card] + [None] * (HAND_SIZE - 1)
    tracker = GameStateTracker()
    tracker.sync(_make_state(hand=hand, mana=5))
    mask = tracker.action_masks()
    # BurnFromHand(0) = action 1
    assert mask[1] == True
    # BurnFromHand(1) = action 2 — no card
    assert mask[2] == False
    # PlayFromHand(0, 0) = action 6 — card exists, slot empty, mana sufficient
    assert mask[6] == True


def test_mask_insufficient_mana():
    """Card costs more than available mana: PlayFromHand masked."""
    card = _make_card(play_cost=8)
    hand = [card] + [None] * (HAND_SIZE - 1)
    tracker = GameStateTracker()
    tracker.sync(_make_state(hand=hand, mana=3))
    mask = tracker.action_masks()
    # BurnFromHand still valid (doesn't cost mana)
    assert mask[1] == True
    # PlayFromHand(0, 0) should be masked — not enough mana
    assert mask[6] == False


def test_mask_occupied_board_slot():
    """Board slot occupied: PlayFromHand to that slot masked."""
    card = _make_card(play_cost=2)
    board_unit = _make_card(card_id=2)
    hand = [card] + [None] * (HAND_SIZE - 1)
    board = [board_unit] + [None] * (BOARD_SIZE - 1)
    tracker = GameStateTracker()
    tracker.sync(_make_state(hand=hand, board=board, mana=5))
    mask = tracker.action_masks()
    # PlayFromHand(0, 0) = action 6 — slot 0 occupied
    assert mask[6] == False
    # PlayFromHand(0, 1) = action 7 — slot 1 empty
    assert mask[7] == True


def test_mask_action_limit_forces_endturn():
    """After max actions, only EndTurn is valid."""
    card = _make_card()
    hand = [card] + [None] * (HAND_SIZE - 1)
    tracker = GameStateTracker()
    tracker.sync(_make_state(hand=hand, mana=5))
    # Fill pending_actions to the limit
    tracker.pending_actions = [("BurnFromHand", {"hand_index": 0})] * DEFAULT_MAX_ACTIONS_PER_TURN
    mask = tracker.action_masks()
    assert mask[0] == True  # EndTurn
    assert np.sum(mask) == 1  # Only EndTurn


def test_mask_swap_needs_two_units():
    """SwapBoard needs both slots occupied."""
    unit = _make_card()
    board = [unit, unit.copy()] + [None] * (BOARD_SIZE - 2)
    tracker = GameStateTracker()
    tracker.sync(_make_state(board=board))
    mask = tracker.action_masks()
    # SwapBoard(0,1) = action 36 — both occupied
    assert mask[36] == True
    # SwapBoard(0,2) = action 37 — slot 2 empty
    assert mask[37] == False


# ── Local State Application ──

def test_burn_from_hand():
    card = _make_card(burn_value=2)
    hand = [card] + [None] * (HAND_SIZE - 1)
    tracker = GameStateTracker()
    tracker.sync(_make_state(hand=hand, mana=3, mana_limit=10))
    tracker.apply_action("BurnFromHand", {"hand_index": 0})
    assert tracker.hand[0] is None
    assert tracker.mana == 5  # 3 + 2


def test_burn_from_hand_caps_at_mana_limit():
    card = _make_card(burn_value=5)
    hand = [card] + [None] * (HAND_SIZE - 1)
    tracker = GameStateTracker()
    tracker.sync(_make_state(hand=hand, mana=8, mana_limit=10))
    tracker.apply_action("BurnFromHand", {"hand_index": 0})
    assert tracker.mana == 10  # capped at limit


def test_play_from_hand():
    card = _make_card(play_cost=3)
    hand = [card] + [None] * (HAND_SIZE - 1)
    tracker = GameStateTracker()
    tracker.sync(_make_state(hand=hand, mana=5))
    tracker.apply_action("PlayFromHand", {"hand_index": 0, "board_slot": 2})
    assert tracker.hand[0] is None
    assert tracker.board[2] is not None
    assert tracker.board[2]["name"] == card["name"]
    assert tracker.mana == 2  # 5 - 3


def test_burn_from_board():
    unit = _make_card(burn_value=3)
    board = [unit] + [None] * (BOARD_SIZE - 1)
    tracker = GameStateTracker()
    tracker.sync(_make_state(board=board, mana=2, mana_limit=10))
    tracker.apply_action("BurnFromBoard", {"board_slot": 0})
    assert tracker.board[0] is None
    assert tracker.mana == 5  # 2 + 3


def test_swap_board():
    unit_a = _make_card(card_id=1)
    unit_b = _make_card(card_id=2)
    board = [unit_a, unit_b] + [None] * (BOARD_SIZE - 2)
    tracker = GameStateTracker()
    tracker.sync(_make_state(board=board))
    tracker.apply_action("SwapBoard", {"slot_a": 0, "slot_b": 1})
    assert tracker.board[0]["id"] == 2
    assert tracker.board[1]["id"] == 1


def test_move_board_forward():
    """Move slot 0 to slot 2: shifts 1,2 left."""
    units = [_make_card(card_id=i) for i in range(3)]
    board = units + [None] * (BOARD_SIZE - 3)
    tracker = GameStateTracker()
    tracker.sync(_make_state(board=board))
    tracker.apply_action("MoveBoard", {"from_slot": 0, "to_slot": 2})
    assert tracker.board[0]["id"] == 1
    assert tracker.board[1]["id"] == 2
    assert tracker.board[2]["id"] == 0


def test_move_board_backward():
    """Move slot 2 to slot 0: shifts 0,1 right."""
    units = [_make_card(card_id=i) for i in range(3)]
    board = units + [None] * (BOARD_SIZE - 3)
    tracker = GameStateTracker()
    tracker.sync(_make_state(board=board))
    tracker.apply_action("MoveBoard", {"from_slot": 2, "to_slot": 0})
    assert tracker.board[0]["id"] == 2
    assert tracker.board[1]["id"] == 0
    assert tracker.board[2]["id"] == 1


# ── Board as Opponent ──

def test_get_board_as_opponent():
    unit = _make_card(card_id=7)
    board = [None, unit, None, None, None]
    tracker = GameStateTracker()
    tracker.sync(_make_state(board=board))
    opp = tracker.get_board_as_opponent()
    assert len(opp) == 1
    assert opp[0] == {"card_id": 7, "slot": 1}


def test_get_board_as_opponent_empty():
    tracker = GameStateTracker()
    tracker.sync(_make_state())
    assert tracker.get_board_as_opponent() == []


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
    # Unset fields should have defaults
    assert loaded.batch_size == 64


def test_config_partial_json(tmp_path):
    """Loading a JSON with only some fields merges over defaults."""
    import json
    path = tmp_path / "partial.json"
    path.write_text(json.dumps({"learning_rate": 0.001}))
    from config import load_config
    config = load_config(str(path))
    assert config.learning_rate == 0.001
    assert config.batch_size == 64  # default
