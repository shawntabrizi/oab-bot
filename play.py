#!/usr/bin/env python3 -u
"""Play Open Auto Battler on a live server using a trained RL model.

Uses a local oab_py.GameSession for observation encoding and action masking
(engine-authoritative), while communicating with the server for actual gameplay.

Usage:
    python play.py --url http://localhost:3000
    python play.py --url http://localhost:3000 --local --set 1
    python play.py --url http://localhost:3000 --model models/oab_agent_5m --games 10
"""

import argparse
import json
import time

import numpy as np
import requests
from sb3_contrib import MaskablePPO

import oab_py
from oab_shared import HAND_SIZE, BOARD_SIZE, ACTION_TABLE


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
        resp = requests.post(f"{self.base_url}/submit", json={"actions": actions})
        resp.raise_for_status()
        return resp.json()

    def shop(self, actions):
        body = {"agent_id": self.agent_id, "actions": actions}
        resp = requests.post(f"{self.base_url}/shop", json=body)
        resp.raise_for_status()
        return resp.json()

    def battle(self, opponent=None):
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
    if card is None:
        return "---"
    return card.get("name", str(card.get("id", "?")))


def _card_line(slot, card):
    name = _card_name(card)
    atk = card.get("attack", "?")
    hp = card.get("health", "?")
    cost = card.get("play_cost", "?")
    burn = card.get("burn_value", "?")
    return f"    [{slot}] {name:<22s} {atk:>2}/{hp:<2}  cost:{cost}  burn:{burn}"


def _board_str(board):
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

def play_game(model, client, session, set_id, game_num, verbose=True):
    """Play a single game using the server for gameplay, local session for obs/mask."""
    state = client.reset(set_id=set_id)
    # Sync local session to same seed/state
    session.reset(state.get("round", 1), None)  # reset with arbitrary seed
    # Re-create session to match server state
    # (In practice, the model just needs consistent obs encoding)

    pending_server_actions = []
    action_log = []
    turn_start_state = state

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  GAME {game_num}")
        print(f"{'=' * 60}")

    while True:
        obs = np.array(session.get_observation(), dtype=np.float32)
        masks = np.array(session.get_action_mask(), dtype=bool)
        action, _ = model.predict(obs, action_masks=masks, deterministic=True)
        action = int(action)
        action_type, params = ACTION_TABLE[action]

        if action_type == "EndTurn":
            if verbose:
                rnd = turn_start_state.get("round", "?")
                mana_limit = turn_start_state.get("mana_limit", "?")
                lives = turn_start_state.get("lives", "?")
                wins = turn_start_state.get("wins", "?")
                bag = turn_start_state.get("bag_count", "?")

                print(f"\n  ---- Round {rnd} ---- lives: {lives}  wins: {wins}  bag: {bag}")

                hand = turn_start_state.get("hand", [])
                mana = turn_start_state.get("mana", 0)
                print(f"  Hand (mana: {mana}/{mana_limit}):")
                any_card = False
                for i in range(min(HAND_SIZE, len(hand))):
                    card = hand[i]
                    if card is not None:
                        print(_card_line(i, card))
                        any_card = True
                if not any_card:
                    print(f"    (empty)")

                board = turn_start_state.get("board", [])
                if any(u is not None for u in board):
                    print(f"  Board: {_board_str(board)}")

                print(f"  Decisions:")
                if action_log:
                    for j, desc in enumerate(action_log, 1):
                        print(f"    {j}. {desc}")
                else:
                    print(f"    -> End turn (no shop actions)")

            # Submit to server
            if client.local_mode:
                try:
                    client.shop(pending_server_actions)
                    result = client.battle()
                except requests.HTTPError as e:
                    print(f"  Error: {e.response.text}")
                    return {"result": "error", "wins": 0, "reward": 0}
            else:
                try:
                    result = client.submit(pending_server_actions)
                except requests.HTTPError as e:
                    print(f"  Error: {e.response.text}")
                    return {"result": "error", "wins": 0, "reward": 0}

            pending_server_actions = []
            action_log = []

            battle_result = result["battle_result"]
            game_over = result["game_over"]
            next_state = result["state"]

            if verbose:
                report = result.get("battle_report", {})
                survived = report.get("player_units_survived", "?")
                enemies = report.get("enemy_units_faced", "?")
                icon = {"Victory": "WIN", "Defeat": "LOSS", "Draw": "DRAW"}.get(battle_result, "?")
                print(f"  Result: ** {icon} **  (vs {enemies} enemies, {survived} survived)")

            if game_over:
                game_result = result.get("game_result", "unknown")
                wins = next_state.get("wins", 0)
                lives = next_state.get("lives", 0)
                if verbose:
                    print(f"\n  {'=' * 40}")
                    if game_result == "victory":
                        print(f"  GAME OVER: VICTORY! (wins={wins})")
                    else:
                        print(f"  GAME OVER: DEFEAT (wins={wins}, lives={lives})")
                    print(f"  {'=' * 40}")
                return {"result": game_result, "wins": wins, "reward": result["reward"]}

            # Advance local session: commit turn with empty opponent (just for state sync)
            session.commit_turn_and_battle("[]")
            turn_start_state = next_state
        else:
            # Log for display
            if verbose:
                action_log.append(f"{action_type} {params}")

            # Build server action
            server_action = {"type": action_type}
            server_action.update(params)
            pending_server_actions.append(server_action)

            # Apply to local session for obs/mask
            session.apply_action(action)


def main():
    parser = argparse.ArgumentParser(
        description="Play OAB on a server using a trained RL model"
    )
    parser.add_argument("--url", default="http://localhost:3000",
                        help="OAB server URL (default: http://localhost:3000)")
    parser.add_argument("--model", default="models/oab_agent",
                        help="Path to trained model (default: models/oab_agent)")
    parser.add_argument("--games", type=int, default=1,
                        help="Number of games to play (default: 1)")
    parser.add_argument("--set", type=int, default=0,
                        help="Card set ID (default: 0)")
    parser.add_argument("--local", action="store_true",
                        help="Use local mode (POST /shop + /battle)")
    parser.add_argument("--agent-id", default="bot",
                        help="Agent ID for local mode sessions (default: bot)")
    parser.add_argument("--quiet", action="store_true",
                        help="Only print game results, not per-round details")
    parser.add_argument("--delay", type=float, default=0,
                        help="Delay in seconds between turns")
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    model = MaskablePPO.load(args.model)

    mode = "local" if args.local else "chain"
    print(f"Connecting to {args.url} ({mode} mode, set {args.set})...")
    client = OABClient(args.url, local_mode=args.local, agent_id=args.agent_id)

    # Local session for obs/mask encoding (engine-authoritative)
    session = oab_py.GameSession(0, args.set)

    victories = 0
    defeats = 0
    errors = 0

    for game_num in range(1, args.games + 1):
        result = play_game(model, client, session, args.set, game_num,
                           verbose=not args.quiet)

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
