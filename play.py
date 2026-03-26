#!/usr/bin/env python3 -u
"""Play Open Auto Battler on a live server using a trained RL model.

Connects to the OAB HTTP server and plays games using the trained
MaskablePPO agent. Supports both chain mode (on-chain games via
POST /submit) and local mode (POST /shop + POST /battle).

Usage:
    # Chain mode (on-chain):
    python play.py --url http://localhost:3000

    # Local mode (testing):
    python play.py --url http://localhost:3000 --local

    # Custom model and number of games:
    python play.py --url http://localhost:3000 --model models/oab_agent_5m --games 10
"""

import argparse
import time

import numpy as np
import requests
from sb3_contrib import MaskablePPO

from oab_shared import (
    HAND_SIZE,
    BOARD_SIZE,
    ACTION_TABLE,
    GameStateTracker,
    extract_card_id,
)


# ── Server Client ──

class OABClient:
    """HTTP client for the OAB game server."""

    def __init__(self, base_url, local_mode=False, agent_id="bot"):
        self.base_url = base_url.rstrip("/")
        self.local_mode = local_mode
        self.agent_id = agent_id

    def reset(self, seed=None, set_id=0):
        body = {"set_id": set_id}
        if self.local_mode:
            body["agent_id"] = self.agent_id
        if seed is not None:
            body["seed"] = seed
        resp = requests.post(f"{self.base_url}/reset", json=body)
        resp.raise_for_status()
        return resp.json()

    def submit(self, actions):
        """Chain mode: submit all actions for the turn."""
        resp = requests.post(f"{self.base_url}/submit", json={"actions": actions})
        resp.raise_for_status()
        return resp.json()

    def shop(self, actions):
        """Local mode: apply shop actions."""
        body = {"agent_id": self.agent_id, "actions": actions}
        resp = requests.post(f"{self.base_url}/shop", json=body)
        resp.raise_for_status()
        return resp.json()

    def battle(self, opponent=None):
        """Local mode: run battle."""
        body = {"agent_id": self.agent_id, "opponent": opponent or []}
        resp = requests.post(f"{self.base_url}/battle", json=body)
        resp.raise_for_status()
        return resp.json()

    def get_state(self):
        params = {}
        if self.local_mode:
            params["agent_id"] = self.agent_id
        resp = requests.get(f"{self.base_url}/state", params=params)
        resp.raise_for_status()
        return resp.json()


# ── Display Helpers ──

def _card_name(card):
    """Get a display name for a card/unit dict."""
    if card is None:
        return "---"
    return card.get("name", f"#{extract_card_id(card)}")


def _card_line(slot, card):
    """Format a card for a table-style display: [slot] Name  ATK/HP  cost burn"""
    name = _card_name(card)
    atk = card.get("attack", "?")
    hp = card.get("health", "?")
    cost = card.get("play_cost", "?")
    burn = card.get("burn_value", "?")
    return f"    [{slot}] {name:<22s} {atk:>2}/{hp:<2}  cost:{cost}  burn:{burn}"


def _format_action(action_type, params, tracker):
    """Format a single action with mana tracking."""
    mana_before = tracker.mana
    if action_type == "BurnFromHand":
        i = params["hand_index"]
        card = tracker.hand[i]
        name = _card_name(card)
        burn_val = card["burn_value"] if card else 0
        mana_after = min(mana_before + burn_val, tracker.mana_limit)
        return f"BURN  hand[{i}] {name}  (+{burn_val} mana -> {mana_after})"
    elif action_type == "PlayFromHand":
        i, j = params["hand_index"], params["board_slot"]
        card = tracker.hand[i]
        name = _card_name(card)
        cost = card["play_cost"] if card else 0
        atk = card.get("attack", "?") if card else "?"
        hp = card.get("health", "?") if card else "?"
        mana_after = mana_before - cost
        return f"PLAY  hand[{i}] {name} ({atk}/{hp}) -> board[{j}]  (-{cost} mana -> {mana_after})"
    elif action_type == "BurnFromBoard":
        j = params["board_slot"]
        unit = tracker.board[j]
        name = _card_name(unit)
        burn_val = unit["burn_value"] if unit else 0
        mana_after = min(mana_before + burn_val, tracker.mana_limit)
        return f"SELL  board[{j}] {name}  (+{burn_val} mana -> {mana_after})"
    elif action_type == "SwapBoard":
        a, b = params["slot_a"], params["slot_b"]
        return f"SWAP  board[{a}] {_card_name(tracker.board[a])} <-> board[{b}] {_card_name(tracker.board[b])}"
    elif action_type == "MoveBoard":
        f, t = params["from_slot"], params["to_slot"]
        return f"MOVE  board[{f}] {_card_name(tracker.board[f])} -> board[{t}]"
    return f"{action_type} {params}"


def _board_str(board):
    """One-line board summary for battle display."""
    units = []
    for i in range(BOARD_SIZE):
        unit = board[i] if i < len(board) else None
        if unit is not None:
            name = _card_name(unit)
            atk = unit.get("attack", "?")
            hp = unit.get("health", "?")
            units.append(f"[{i}]{name}({atk}/{hp})")
    return " | ".join(units) if units else "(empty)"


# ── Bot ──

def play_game(model, client, tracker, game_num, verbose=True):
    """Play a single game, returning the result dict."""
    state = client.reset(set_id=0)
    tracker.sync(state)
    action_log = []
    # Snapshot the starting state for display
    turn_start_hand = [c.copy() if c else None for c in tracker.hand]
    turn_start_board = [u.copy() if u else None for u in tracker.board]
    turn_start_mana = tracker.mana

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  GAME {game_num}")
        print(f"{'=' * 60}")

    while True:
        obs = tracker.encode_obs()
        masks = tracker.action_masks()
        action, _ = model.predict(obs, action_masks=masks, deterministic=True)
        action = int(action)
        action_type, params = ACTION_TABLE[action]

        if action_type == "EndTurn":
            if verbose:
                rnd = tracker.state["round"]
                mana_limit = tracker.state["mana_limit"]
                lives = tracker.state["lives"]
                wins = tracker.state["wins"]
                bag = tracker.state["bag_count"]

                print(f"\n  ---- Round {rnd} ---- lives: {lives}  wins: {wins}  bag: {bag}")

                # Hand dealt this round
                print(f"  Hand (mana: {turn_start_mana}/{mana_limit}):")
                any_card = False
                for i in range(HAND_SIZE):
                    card = turn_start_hand[i]
                    if card is not None:
                        print(_card_line(i, card))
                        any_card = True
                if not any_card:
                    print(f"    (empty)")

                # Board at start of round
                if any(u is not None for u in turn_start_board):
                    print(f"  Board: {_board_str(turn_start_board)}")

                # Decisions
                print(f"  Decisions:")
                if action_log:
                    for j, desc in enumerate(action_log, 1):
                        print(f"    {j}. {desc}")
                else:
                    print(f"    -> End turn (no shop actions)")

                # Board going into battle
                print(f"  Into battle: {_board_str(tracker.board)}")

            # Build server action list
            server_actions = []
            for at, ap in tracker.pending_actions:
                a = {"type": at}
                a.update(ap)
                server_actions.append(a)

            if client.local_mode:
                try:
                    client.shop(server_actions)
                    result = client.battle()
                except requests.HTTPError as e:
                    print(f"  Error: {e.response.text}")
                    return {"result": "error", "wins": 0, "reward": 0}
            else:
                try:
                    result = client.submit(server_actions)
                except requests.HTTPError as e:
                    print(f"  Error: {e.response.text}")
                    return {"result": "error", "wins": 0, "reward": 0}

            action_log = []

            battle_result = result["battle_result"]
            game_over = result["game_over"]
            reward = result["reward"]
            next_state = result["state"]

            if verbose:
                survived = result["battle_report"]["player_units_survived"]
                enemies = result["battle_report"]["enemy_units_faced"]
                icon = {"Victory": "WIN", "Defeat": "LOSS", "Draw": "DRAW"}.get(battle_result, "?")
                print(f"  Result: ** {icon} **  (vs {enemies} enemies, {survived} survived)")

            if game_over:
                game_result = result.get("game_result", "unknown")
                wins = next_state["wins"]
                lives = next_state["lives"]
                if verbose:
                    print(f"\n  {'=' * 40}")
                    if game_result == "victory":
                        print(f"  GAME OVER: VICTORY! (wins={wins})")
                    else:
                        print(f"  GAME OVER: DEFEAT (wins={wins}, lives={lives})")
                    print(f"  {'=' * 40}")
                return {
                    "result": game_result,
                    "wins": wins,
                    "reward": reward,
                }

            tracker.sync(next_state)
            # Snapshot for next turn's display
            turn_start_hand = [c.copy() if c else None for c in tracker.hand]
            turn_start_board = [u.copy() if u else None for u in tracker.board]
            turn_start_mana = tracker.mana
        else:
            # Log the action before applying it (so we see the card names)
            if verbose:
                action_log.append(_format_action(action_type, params, tracker))
            tracker.pending_actions.append((action_type, params))
            tracker.apply_action(action_type, params)


def main():
    parser = argparse.ArgumentParser(
        description="Play OAB on a server using a trained RL model"
    )
    parser.add_argument(
        "--url", default="http://localhost:3000",
        help="OAB server URL (default: http://localhost:3000)"
    )
    parser.add_argument(
        "--model", default="models/oab_agent",
        help="Path to trained model (default: models/oab_agent)"
    )
    parser.add_argument(
        "--games", type=int, default=1,
        help="Number of games to play (default: 1)"
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Use local mode (POST /shop + /battle) instead of chain mode (POST /submit)"
    )
    parser.add_argument(
        "--agent-id", default="bot",
        help="Agent ID for local mode sessions (default: bot)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Only print game results, not per-round details"
    )
    parser.add_argument(
        "--delay", type=float, default=0,
        help="Delay in seconds between turns (useful for chain mode)"
    )
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    model = MaskablePPO.load(args.model)

    mode = "local" if args.local else "chain"
    print(f"Connecting to {args.url} ({mode} mode)...")
    client = OABClient(args.url, local_mode=args.local, agent_id=args.agent_id)
    tracker = GameStateTracker()

    victories = 0
    defeats = 0
    errors = 0

    for game_num in range(1, args.games + 1):
        result = play_game(model, client, tracker, game_num, verbose=not args.quiet)

        if result["result"] == "victory":
            victories += 1
        elif result["result"] == "defeat":
            defeats += 1
        else:
            errors += 1

        if args.delay > 0 and game_num < args.games:
            time.sleep(args.delay)

    print(f"\n{'=' * 40}")
    print(f"  Results ({args.games} games, {mode} mode)")
    print(f"{'=' * 40}")
    print(f"  Victories: {victories} ({100*victories/args.games:.1f}%)")
    print(f"  Defeats:   {defeats} ({100*defeats/args.games:.1f}%)")
    if errors:
        print(f"  Errors:    {errors}")


if __name__ == "__main__":
    main()
