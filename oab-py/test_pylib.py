#!/usr/bin/env python3
"""End-to-end tests for the oab_py native bindings."""

import json
import oab_py


def test_create_session():
    session = oab_py.GameSession(42, 0)
    state = json.loads(session.get_state())
    assert state["round"] == 1
    assert state["phase"] == "shop"
    assert state["lives"] == 3
    assert state["wins"] == 0
    assert state["mana_limit"] == 3
    assert len(state["hand"]) <= 5
    assert len(state["board"]) == 5
    assert state["bag_count"] > 0
    assert len(state["bag"]) > 0
    print("  PASS: create_session")


def test_deterministic_seeds():
    s1 = json.loads(oab_py.GameSession(123, 0).get_state())
    s2 = json.loads(oab_py.GameSession(123, 0).get_state())
    assert s1["hand"] == s2["hand"]
    assert s1["bag_count"] == s2["bag_count"]
    assert s1["mana"] == s2["mana"]

    s3 = json.loads(oab_py.GameSession(999, 0).get_state())
    # Different seed should (almost certainly) produce different hand
    assert s1["hand"] != s3["hand"] or s1["bag"] != s3["bag"]
    print("  PASS: deterministic_seeds")


def test_shop_empty_actions():
    session = oab_py.GameSession(42, 0)
    state = json.loads(session.shop("[]"))
    assert state["phase"] == "battle"
    assert state["round"] == 1
    print("  PASS: shop_empty_actions")


def test_shop_then_battle():
    session = oab_py.GameSession(42, 0)
    session.shop("[]")
    result = json.loads(session.battle("[]"))
    assert result["completed_round"] == 1
    assert result["battle_result"] in ("Victory", "Defeat", "Draw")
    assert result["reward"] in (1, -1, 0)
    assert result["state"]["round"] == 2 or result["game_over"]
    print("  PASS: shop_then_battle")


def test_battle_before_shop_fails():
    session = oab_py.GameSession(42, 0)
    try:
        session.battle("[]")
        assert False, "Should have raised"
    except RuntimeError as e:
        assert "Wrong phase" in str(e)
    print("  PASS: battle_before_shop_fails")


def test_shop_twice_fails():
    session = oab_py.GameSession(42, 0)
    session.shop("[]")
    try:
        session.shop("[]")
        assert False, "Should have raised"
    except RuntimeError as e:
        assert "Wrong phase" in str(e)
    print("  PASS: shop_twice_fails")


def test_reset():
    session = oab_py.GameSession(42, 0)
    session.shop("[]")
    session.battle("[]")

    state = json.loads(session.reset(99, None))
    assert state["round"] == 1
    assert state["phase"] == "shop"
    assert state["wins"] == 0
    print("  PASS: reset")


def test_reset_with_set_id():
    session = oab_py.GameSession(42, 0)
    state = json.loads(session.reset(99, 0))
    assert state["round"] == 1
    print("  PASS: reset_with_set_id")


def test_play_full_game():
    """Play a full game: burn for mana, then play cards."""
    session = oab_py.GameSession(42, 0)
    rounds_played = 0

    while True:
        state = json.loads(session.get_state())

        # Strategy: burn worst cards for mana, play best with earned mana
        actions = []
        mana = state["mana"]
        hand = [(i, c) for i, c in enumerate(state["hand"]) if c]
        empty_slots = [i for i, s in enumerate(state["board"]) if s is None]

        if hand:
            # Sort by cost (cheap = burn candidates, expensive = play candidates)
            hand.sort(key=lambda x: x[1]["play_cost"])

            # Burn the cheapest cards first
            to_burn = hand[: len(hand) // 2 + 1]
            to_play = hand[len(hand) // 2 + 1 :]

            for i, card in to_burn:
                actions.append({"type": "BurnFromHand", "hand_index": i})
                mana = min(mana + card["burn_value"], state["mana_limit"])

            for i, card in to_play:
                if card["play_cost"] <= mana and empty_slots:
                    slot = empty_slots.pop(0)
                    actions.append({"type": "PlayFromHand", "hand_index": i, "board_slot": slot})
                    mana -= card["play_cost"]

        session.shop(json.dumps(actions))
        result = json.loads(session.battle("[]"))
        rounds_played += 1

        if result["game_over"]:
            break

        assert rounds_played < 50, "Game should end within 50 rounds"

    assert rounds_played > 0
    assert result["game_result"] in ("victory", "defeat")
    print(f"  PASS: play_full_game ({rounds_played} rounds, {result['game_result']})")


def test_play_with_actions():
    """Play a round: burn a card for mana, then play one."""
    session = oab_py.GameSession(42, 0)
    state = json.loads(session.get_state())

    actions = []
    mana = state["mana"]
    hand = [(i, c) for i, c in enumerate(state["hand"]) if c]
    empty_slots = [i for i, s in enumerate(state["board"]) if s is None]

    # Burn first card for mana
    if len(hand) >= 2:
        burn_idx, burn_card = hand[0]
        actions.append({"type": "BurnFromHand", "hand_index": burn_idx})
        mana = min(mana + burn_card["burn_value"], state["mana_limit"])

        # Play second card if affordable
        play_idx, play_card = hand[1]
        if play_card["play_cost"] <= mana and empty_slots:
            actions.append({
                "type": "PlayFromHand",
                "hand_index": play_idx,
                "board_slot": empty_slots[0],
            })

    shop_state = json.loads(session.shop(json.dumps(actions)))
    assert shop_state["phase"] == "battle"

    result = json.loads(session.battle("[]"))
    assert result["completed_round"] == 1
    print(f"  PASS: play_with_actions ({len(actions)} actions)")


def test_battle_with_opponent():
    """Battle against a custom opponent board."""
    session = oab_py.GameSession(42, 0)
    state = json.loads(session.get_state())

    # Get a valid card_id from the hand
    card_id = None
    for card in state["hand"]:
        if card:
            card_id = card["id"]
            break

    assert card_id is not None, "Should have at least one card in hand"

    session.shop("[]")
    opponent = json.dumps([{"card_id": card_id, "slot": 0}])
    result = json.loads(session.battle(opponent))
    assert result["battle_result"] in ("Victory", "Defeat", "Draw")
    print(f"  PASS: battle_with_opponent (result: {result['battle_result']})")


def test_bag_decreases():
    """Bag should shrink as rounds progress."""
    session = oab_py.GameSession(42, 0)
    state = json.loads(session.get_state())
    initial_bag = state["bag_count"]

    session.shop("[]")
    result = json.loads(session.battle("[]"))

    if not result["game_over"]:
        next_bag = result["state"]["bag_count"]
        # Bag should be smaller (cards drawn for hand) or same (if hand returned)
        assert next_bag <= initial_bag
    print("  PASS: bag_decreases")


def test_get_cards():
    session = oab_py.GameSession(42, 0)
    cards = json.loads(session.get_cards())
    assert len(cards) > 0
    card = cards[0]
    assert "id" in card
    assert "name" in card
    assert "attack" in card
    assert "health" in card
    print(f"  PASS: get_cards ({len(cards)} cards)")


if __name__ == "__main__":
    print("Running oab_py end-to-end tests...\n")

    test_create_session()
    test_deterministic_seeds()
    test_shop_empty_actions()
    test_shop_then_battle()
    test_battle_before_shop_fails()
    test_shop_twice_fails()
    test_reset()
    test_reset_with_set_id()
    test_play_full_game()
    test_play_with_actions()
    test_battle_with_opponent()
    test_bag_decreases()
    test_get_cards()

    print("\nAll tests passed!")
