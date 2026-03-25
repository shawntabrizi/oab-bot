#!/usr/bin/env python3
"""Gymnasium environment for Open Auto Battler with self-play support.

Uses the native PyO3 bindings (oab_py) for direct game engine access,
bypassing the HTTP server entirely.

Each gym step is one action within a shop turn. The 'EndTurn' action
submits shop actions, picks an opponent from the shared board pool,
runs the battle, and returns the result.
"""

import json
import random
import threading
from itertools import combinations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import oab_py

# ── Game Constants ──

HAND_SIZE = 5
BOARD_SIZE = 5

# ── Normalization Constants ──

MAX_ATTACK = 15.0
MAX_HEALTH = 20.0
MAX_COST = 10.0
MAX_BURN = 5.0
MAX_MANA = 10.0
MAX_ROUND = 20.0
MAX_LIVES = 3.0
MAX_WINS = 10.0
MAX_BAG = 50.0
MAX_CARD_ID = 100.0
MAX_ABILITIES = 5.0

# ── Observation Layout ──

HAND_FEATURES = 9
BOARD_FEATURES = 8
SCALAR_FEATURES = 6

OBS_DIM = HAND_SIZE * HAND_FEATURES + BOARD_SIZE * BOARD_FEATURES + SCALAR_FEATURES


# ── Action Table ──

def _build_action_table():
    actions = []
    actions.append(("EndTurn", {}))
    for i in range(HAND_SIZE):
        actions.append(("BurnFromHand", {"hand_index": i}))
    for hi in range(HAND_SIZE):
        for bs in range(BOARD_SIZE):
            actions.append(("PlayFromHand", {"hand_index": hi, "board_slot": bs}))
    for bs in range(BOARD_SIZE):
        actions.append(("BurnFromBoard", {"board_slot": bs}))
    for a, b in combinations(range(BOARD_SIZE), 2):
        actions.append(("SwapBoard", {"slot_a": a, "slot_b": b}))
    for f in range(BOARD_SIZE):
        for t in range(BOARD_SIZE):
            if f != t:
                actions.append(("MoveBoard", {"from_slot": f, "to_slot": t}))
    return actions


ACTION_TABLE = _build_action_table()
NUM_ACTIONS = len(ACTION_TABLE)  # 66


# ── Shared Opponent Pool ──

class BoardPool:
    """Thread-safe pool of opponent boards from recent rounds.

    All agents in a self-play lobby share one pool. When an agent finishes
    shopping, it posts its board and samples an opponent from the pool.
    """

    def __init__(self, max_size=200):
        self._boards = []
        self._lock = threading.Lock()
        self._max_size = max_size

    def add(self, board):
        with self._lock:
            self._boards.append(board)
            if len(self._boards) > self._max_size:
                self._boards = self._boards[-self._max_size // 2 :]

    def sample(self):
        with self._lock:
            if not self._boards:
                return []
            return random.choice(self._boards)

    def __len__(self):
        with self._lock:
            return len(self._boards)


# ── Gymnasium Environment ──

class OABEnv(gym.Env):
    """Open Auto Battler environment with self-play.

    Uses native PyO3 bindings — no HTTP server needed.

    Observation: Box(91,) normalized to [0, 1]
    Action: Discrete(66) with action masking
    Reward: +1 win, -1 loss, 0 draw (per battle round)
    """

    metadata = {"render_modes": []}

    def __init__(self, set_id=0, board_pool=None):
        super().__init__()
        self.set_id = set_id
        self.board_pool = board_pool

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # Game session (native Rust)
        self._session = None
        self._state = None
        self._hand = [None] * HAND_SIZE
        self._board = [None] * BOARD_SIZE
        self._mana = 0
        self._mana_limit = 0
        self._pending_actions = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        game_seed = int(self.np_random.integers(0, 2**32))
        self._session = oab_py.GameSession(game_seed, self.set_id)
        self._state = json.loads(self._session.get_state())
        self._sync_local_state()
        self._pending_actions = []
        return self._encode_obs(), self._make_info()

    def step(self, action):
        action_type, params = ACTION_TABLE[int(action)]

        if action_type == "EndTurn":
            return self._do_end_turn()
        else:
            self._pending_actions.append((action_type, params))
            self._apply_local_action(action_type, params)
            return self._encode_obs(), 0.0, False, False, self._make_info()

    def action_masks(self):
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        for i, (action_type, params) in enumerate(ACTION_TABLE):
            mask[i] = self._is_valid(action_type, params)
        return mask

    def get_board_as_opponent(self):
        """Return current board in OpponentUnit format."""
        opponent = []
        for slot, unit in enumerate(self._board):
            if unit is not None:
                opponent.append({"card_id": _extract_card_id(unit), "slot": slot})
        return opponent

    # ── Turn Submission ──

    def _do_end_turn(self):
        # Build actions JSON
        server_actions = []
        for action_type, params in self._pending_actions:
            action = {"type": action_type}
            action.update(params)
            server_actions.append(action)

        actions_json = json.dumps(server_actions)

        # Shop phase
        try:
            shop_result = json.loads(self._session.shop(actions_json))
        except Exception:
            # Fallback: empty shop
            try:
                shop_result = json.loads(self._session.shop("[]"))
            except Exception as e:
                return self._encode_obs(), -1.0, True, False, {
                    "error": str(e), **self._make_info()
                }

        self._state = shop_result
        self._sync_local_state()

        # Post board to pool and sample opponent
        my_board = self.get_board_as_opponent()
        if self.board_pool is not None:
            self.board_pool.add(my_board)
            opponent = self.board_pool.sample()
        else:
            opponent = []

        opponent_json = json.dumps(opponent)

        # Battle phase
        try:
            result = json.loads(self._session.battle(opponent_json))
        except Exception as e:
            return self._encode_obs(), -1.0, True, False, {
                "error": str(e), **self._make_info()
            }

        reward = float(result["reward"])
        terminated = result["game_over"]

        self._state = result["state"]
        self._sync_local_state()
        self._pending_actions = []

        info = self._make_info()
        info["battle_result"] = result["battle_result"]
        if terminated:
            info["game_result"] = result.get("game_result")

        return self._encode_obs(), reward, terminated, False, info

    # ── Local State Tracking ──

    def _sync_local_state(self):
        s = self._state
        self._hand = list(s["hand"])
        self._board = list(s["board"])
        while len(self._hand) < HAND_SIZE:
            self._hand.append(None)
        while len(self._board) < BOARD_SIZE:
            self._board.append(None)
        self._mana = s["mana"]
        self._mana_limit = s["mana_limit"]

    def _apply_local_action(self, action_type, params):
        if action_type == "BurnFromHand":
            i = params["hand_index"]
            card = self._hand[i]
            if card:
                self._mana = min(self._mana + card["burn_value"], self._mana_limit)
                self._hand[i] = None
        elif action_type == "PlayFromHand":
            i, j = params["hand_index"], params["board_slot"]
            card = self._hand[i]
            if card:
                self._mana -= card["play_cost"]
                self._board[j] = card
                self._hand[i] = None
        elif action_type == "BurnFromBoard":
            j = params["board_slot"]
            unit = self._board[j]
            if unit:
                self._mana = min(self._mana + unit["burn_value"], self._mana_limit)
                self._board[j] = None
        elif action_type == "SwapBoard":
            a, b = params["slot_a"], params["slot_b"]
            self._board[a], self._board[b] = self._board[b], self._board[a]
        elif action_type == "MoveBoard":
            f, t = params["from_slot"], params["to_slot"]
            unit = self._board[f]
            if f < t:
                for k in range(f, t):
                    self._board[k] = self._board[k + 1]
            else:
                for k in range(f, t, -1):
                    self._board[k] = self._board[k - 1]
            self._board[t] = unit

    # ── Action Masking ──

    def _is_valid(self, action_type, params):
        if action_type == "EndTurn":
            return True
        if action_type == "BurnFromHand":
            return self._hand[params["hand_index"]] is not None
        if action_type == "PlayFromHand":
            i, j = params["hand_index"], params["board_slot"]
            card = self._hand[i]
            return card is not None and self._board[j] is None and self._mana >= card["play_cost"]
        if action_type == "BurnFromBoard":
            return self._board[params["board_slot"]] is not None
        if action_type == "SwapBoard":
            a, b = params["slot_a"], params["slot_b"]
            return self._board[a] is not None and self._board[b] is not None
        if action_type == "MoveBoard":
            f, t = params["from_slot"], params["to_slot"]
            return self._board[f] is not None and self._board[t] is not None
        return False

    # ── Observation Encoding ──

    def _encode_obs(self):
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        idx = 0

        for i in range(HAND_SIZE):
            card = self._hand[i]
            if card is not None:
                obs[idx + 0] = 1.0
                obs[idx + 1] = _norm_id(card.get("id", 0))
                obs[idx + 2] = _norm(card["attack"], MAX_ATTACK)
                obs[idx + 3] = _norm(card["health"], MAX_HEALTH)
                obs[idx + 4] = _norm(card["play_cost"], MAX_COST)
                obs[idx + 5] = _norm(card["burn_value"], MAX_BURN)
                obs[idx + 6] = 1.0 if self._mana >= card["play_cost"] else 0.0
                obs[idx + 7] = _norm(len(card.get("battle_abilities", [])), MAX_ABILITIES)
                obs[idx + 8] = _norm(len(card.get("shop_abilities", [])), MAX_ABILITIES)
            idx += HAND_FEATURES

        for i in range(BOARD_SIZE):
            unit = self._board[i]
            if unit is not None:
                obs[idx + 0] = 1.0
                obs[idx + 1] = _norm_id(unit.get("id", 0))
                obs[idx + 2] = _norm(unit["attack"], MAX_ATTACK)
                obs[idx + 3] = _norm(unit["health"], MAX_HEALTH)
                obs[idx + 4] = _norm(unit["play_cost"], MAX_COST)
                obs[idx + 5] = _norm(unit["burn_value"], MAX_BURN)
                obs[idx + 6] = _norm(len(unit.get("battle_abilities", [])), MAX_ABILITIES)
                obs[idx + 7] = _norm(len(unit.get("shop_abilities", [])), MAX_ABILITIES)
            idx += BOARD_FEATURES

        s = self._state
        obs[idx + 0] = _norm(self._mana, MAX_MANA)
        obs[idx + 1] = _norm(self._mana_limit, MAX_MANA)
        obs[idx + 2] = _norm(s["round"], MAX_ROUND)
        obs[idx + 3] = _norm(s["lives"], MAX_LIVES)
        obs[idx + 4] = _norm(s["wins"], MAX_WINS)
        obs[idx + 5] = _norm(s["bag_count"], MAX_BAG)

        return obs

    def _make_info(self):
        s = self._state
        return {"round": s["round"], "lives": s["lives"], "wins": s["wins"]}


# ── Helpers ──

def _norm(value, max_val):
    return min(float(value) / max_val, 1.0) if max_val > 0 else 0.0


def _norm_id(card_id):
    if isinstance(card_id, dict):
        card_id = next(iter(card_id.values()), 0)
    return min(float(card_id) / MAX_CARD_ID, 1.0)


def _extract_card_id(unit):
    cid = unit.get("id", 0)
    if isinstance(cid, dict):
        cid = next(iter(cid.values()), 0)
    return int(cid)
