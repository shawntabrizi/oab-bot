#!/usr/bin/env python3
"""Gymnasium environment for Open Auto Battler.

Wraps the OAB game server HTTP API into a standard Gymnasium interface
with action masking support for sb3-contrib's MaskablePPO.

Each gym step is one action within a shop turn. The special 'EndTurn'
action submits all accumulated actions to the server and triggers battle.
"""

import json
import urllib.request
import urllib.error
from itertools import combinations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

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
# Hand:  5 slots × 9 features = 45
# Board: 5 slots × 8 features = 40
# Scalars: 6
# Total: 91

HAND_FEATURES = 9  # present, card_id, attack, health, cost, burn, can_afford, n_battle, n_shop
BOARD_FEATURES = 8  # present, card_id, attack, health, cost, burn, n_battle, n_shop
SCALAR_FEATURES = 6  # mana, mana_limit, round, lives, wins, bag_count

OBS_DIM = HAND_SIZE * HAND_FEATURES + BOARD_SIZE * BOARD_FEATURES + SCALAR_FEATURES


# ── Action Table ──
# Maps discrete action index → (action_type, params_dict)

def _build_action_table():
    actions = []

    # 0: EndTurn
    actions.append(("EndTurn", {}))

    # 1-5: BurnFromHand
    for i in range(HAND_SIZE):
        actions.append(("BurnFromHand", {"hand_index": i}))

    # 6-30: PlayFromHand (hand_index × board_slot)
    for hi in range(HAND_SIZE):
        for bs in range(BOARD_SIZE):
            actions.append(("PlayFromHand", {"hand_index": hi, "board_slot": bs}))

    # 31-35: BurnFromBoard
    for bs in range(BOARD_SIZE):
        actions.append(("BurnFromBoard", {"board_slot": bs}))

    # 36-45: SwapBoard (unordered pairs)
    for a, b in combinations(range(BOARD_SIZE), 2):
        actions.append(("SwapBoard", {"slot_a": a, "slot_b": b}))

    # 46-65: MoveBoard (ordered pairs, from ≠ to)
    for f in range(BOARD_SIZE):
        for t in range(BOARD_SIZE):
            if f != t:
                actions.append(("MoveBoard", {"from_slot": f, "to_slot": t}))

    return actions


ACTION_TABLE = _build_action_table()
NUM_ACTIONS = len(ACTION_TABLE)  # 66


# ── HTTP Client ──

class OABClient:
    """Minimal HTTP client for the OAB game server."""

    def __init__(self, base_url="http://localhost:3000"):
        self.base_url = base_url

    def _post(self, path, data=None):
        body = json.dumps(data).encode() if data else b""
        req = urllib.request.Request(
            self.base_url + path,
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            return json.loads(e.read().decode())

    def _get(self, path):
        req = urllib.request.Request(self.base_url + path)
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())

    def reset(self, seed=None, set_id=None):
        body = {}
        if seed is not None:
            body["seed"] = seed
        if set_id is not None:
            body["set_id"] = set_id
        return self._post("/reset", body if body else None)

    def submit(self, actions):
        return self._post("/submit", {"actions": actions})

    def state(self):
        return self._get("/state")

    def cards(self):
        return self._get("/cards")

    def sets(self):
        return self._get("/sets")


# ── Gymnasium Environment ──

class OABEnv(gym.Env):
    """Open Auto Battler environment.

    Observation: Box(91,) normalized to [0, 1]
    Action: Discrete(66) with action masking
    Reward: +1 win, -1 loss, 0 draw (per battle round)
    """

    metadata = {"render_modes": []}

    def __init__(self, server_url="http://localhost:3000", set_id=0):
        super().__init__()
        self.client = OABClient(server_url)
        self.set_id = set_id

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # Internal state tracking
        self._server_state = None
        self._hand = [None] * HAND_SIZE
        self._board = [None] * BOARD_SIZE
        self._mana = 0
        self._mana_limit = 0
        self._pending_actions = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        game_seed = int(self.np_random.integers(0, 2**32)) if seed is not None else None
        self._server_state = self.client.reset(seed=game_seed, set_id=self.set_id)
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
        """Boolean mask of valid actions for MaskablePPO."""
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        for i, (action_type, params) in enumerate(ACTION_TABLE):
            mask[i] = self._is_valid(action_type, params)
        return mask

    # ── Turn Submission ──

    def _do_end_turn(self):
        """Submit accumulated actions to server, get battle result."""
        server_actions = []
        for action_type, params in self._pending_actions:
            action = {"type": action_type}
            action.update(params)
            server_actions.append(action)

        result = self.client.submit(server_actions)

        # Handle server errors by falling back to empty submit
        if "error" in result:
            result = self.client.submit([])
            if "error" in result:
                # Unrecoverable — treat as loss
                return self._encode_obs(), -1.0, True, False, {
                    "error": result["error"],
                    **self._make_info(),
                }

        reward = float(result["reward"])
        terminated = result["game_over"]

        self._server_state = result["state"]
        self._sync_local_state()
        self._pending_actions = []

        info = self._make_info()
        info["battle_result"] = result["battle_result"]
        if terminated:
            info["game_result"] = result.get("game_result")

        return self._encode_obs(), reward, terminated, False, info

    # ── Local State Tracking ──

    def _sync_local_state(self):
        """Copy server state into local tracking for masking/encoding."""
        s = self._server_state
        self._hand = list(s["hand"])
        self._board = list(s["board"])
        self._mana = s["mana"]
        self._mana_limit = s["mana_limit"]

    def _apply_local_action(self, action_type, params):
        """Simulate action effect locally for intermediate state."""
        if action_type == "BurnFromHand":
            i = params["hand_index"]
            card = self._hand[i]
            if card:
                self._mana = min(self._mana + card["burn_value"], self._mana_limit)
                self._hand[i] = None

        elif action_type == "PlayFromHand":
            i = params["hand_index"]
            j = params["board_slot"]
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
        """Check if an action is valid given current local state."""
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
        """Encode state as normalized float32 vector."""
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        idx = 0

        # Hand slots (5 × 9)
        for i in range(HAND_SIZE):
            card = self._hand[i]
            if card is not None:
                obs[idx + 0] = 1.0  # present
                obs[idx + 1] = _norm_id(card.get("id", 0))
                obs[idx + 2] = _norm(card["attack"], MAX_ATTACK)
                obs[idx + 3] = _norm(card["health"], MAX_HEALTH)
                obs[idx + 4] = _norm(card["play_cost"], MAX_COST)
                obs[idx + 5] = _norm(card["burn_value"], MAX_BURN)
                obs[idx + 6] = 1.0 if self._mana >= card["play_cost"] else 0.0
                obs[idx + 7] = _norm(len(card.get("battle_abilities", [])), MAX_ABILITIES)
                obs[idx + 8] = _norm(len(card.get("shop_abilities", [])), MAX_ABILITIES)
            idx += HAND_FEATURES

        # Board slots (5 × 8)
        for i in range(BOARD_SIZE):
            unit = self._board[i]
            if unit is not None:
                obs[idx + 0] = 1.0  # present
                obs[idx + 1] = _norm_id(unit.get("id", 0))
                obs[idx + 2] = _norm(unit["attack"], MAX_ATTACK)
                obs[idx + 3] = _norm(unit["health"], MAX_HEALTH)
                obs[idx + 4] = _norm(unit["play_cost"], MAX_COST)
                obs[idx + 5] = _norm(unit["burn_value"], MAX_BURN)
                obs[idx + 6] = _norm(len(unit.get("battle_abilities", [])), MAX_ABILITIES)
                obs[idx + 7] = _norm(len(unit.get("shop_abilities", [])), MAX_ABILITIES)
            idx += BOARD_FEATURES

        # Scalars (6)
        s = self._server_state
        obs[idx + 0] = _norm(self._mana, MAX_MANA)
        obs[idx + 1] = _norm(self._mana_limit, MAX_MANA)
        obs[idx + 2] = _norm(s["round"], MAX_ROUND)
        obs[idx + 3] = _norm(s["lives"], MAX_LIVES)
        obs[idx + 4] = _norm(s["wins"], MAX_WINS)
        obs[idx + 5] = _norm(s["bag_count"], MAX_BAG)

        return obs

    def _make_info(self):
        s = self._server_state
        return {"round": s["round"], "lives": s["lives"], "wins": s["wins"]}


# ── Helpers ──

def _norm(value, max_val):
    """Normalize a value to [0, 1]."""
    return min(float(value) / max_val, 1.0) if max_val > 0 else 0.0


def _norm_id(card_id):
    """Normalize a CardId (may be int or dict with transparent serde)."""
    if isinstance(card_id, dict):
        card_id = next(iter(card_id.values()), 0)
    return min(float(card_id) / MAX_CARD_ID, 1.0)
