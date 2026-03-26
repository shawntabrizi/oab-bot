"""Shared constants, action table, helpers, and state tracker for OAB.

Used by env.py (training), play.py (live games), and evaluate.py (analysis).
"""

import numpy as np
from itertools import combinations

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

# ── Reward / Turn Defaults ──

DEFAULT_ACTION_COST = -0.01
DEFAULT_REPEAT_PENALTY = -0.1
DEFAULT_MAX_ACTIONS_PER_TURN = 15


# ── Action Table ──

def _build_action_table():
    """Build flat action table. 66 actions total.

    Layout:
      0:     EndTurn
      1-5:   BurnFromHand(0-4)
      6-30:  PlayFromHand(hand_index, board_slot)
      31-35: BurnFromBoard(0-4)
      36-45: SwapBoard (unordered pairs)
      46-65: MoveBoard (ordered pairs)
    """
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


# ── Helpers ──

def norm(value, max_val):
    """Normalize a value to [0, 1], clamped."""
    return min(float(value) / max_val, 1.0) if max_val > 0 else 0.0


def norm_id(card_id):
    """Normalize a card ID (handles dict-style enum variants)."""
    if isinstance(card_id, dict):
        card_id = next(iter(card_id.values()), 0)
    return min(float(card_id) / MAX_CARD_ID, 1.0)


def extract_card_id(unit):
    """Extract integer card ID from a unit dict."""
    cid = unit.get("id", 0)
    if isinstance(cid, dict):
        cid = next(iter(cid.values()), 0)
    return int(cid)


# ── Game State Tracker ──

class GameStateTracker:
    """Tracks local game state for observation encoding and action masking.

    Used by both OABEnv (training) and play.py (live games) to ensure
    identical observation/masking behavior.
    """

    def __init__(self, max_actions_per_turn=DEFAULT_MAX_ACTIONS_PER_TURN):
        self.max_actions_per_turn = max_actions_per_turn
        self.hand = [None] * HAND_SIZE
        self.board = [None] * BOARD_SIZE
        self.mana = 0
        self.mana_limit = 0
        self.state = None
        self.pending_actions = []

    def sync(self, state):
        """Sync from a game state dict (server response or PyO3 output)."""
        self.state = state
        self.hand = list(state["hand"])
        self.board = list(state["board"])
        while len(self.hand) < HAND_SIZE:
            self.hand.append(None)
        while len(self.board) < BOARD_SIZE:
            self.board.append(None)
        self.mana = state["mana"]
        self.mana_limit = state["mana_limit"]
        self.pending_actions = []

    def apply_action(self, action_type, params):
        """Apply an action locally (updates hand/board/mana before server confirms)."""
        if action_type == "BurnFromHand":
            i = params["hand_index"]
            card = self.hand[i]
            if card:
                self.mana = min(self.mana + card["burn_value"], self.mana_limit)
                self.hand[i] = None
        elif action_type == "PlayFromHand":
            i, j = params["hand_index"], params["board_slot"]
            card = self.hand[i]
            if card:
                self.mana -= card["play_cost"]
                self.board[j] = card
                self.hand[i] = None
        elif action_type == "BurnFromBoard":
            j = params["board_slot"]
            unit = self.board[j]
            if unit:
                self.mana = min(self.mana + unit["burn_value"], self.mana_limit)
                self.board[j] = None
        elif action_type == "SwapBoard":
            a, b = params["slot_a"], params["slot_b"]
            self.board[a], self.board[b] = self.board[b], self.board[a]
        elif action_type == "MoveBoard":
            f, t = params["from_slot"], params["to_slot"]
            unit = self.board[f]
            if f < t:
                for k in range(f, t):
                    self.board[k] = self.board[k + 1]
            else:
                for k in range(f, t, -1):
                    self.board[k] = self.board[k - 1]
            self.board[t] = unit

    def encode_obs(self):
        """Encode current state as a normalized observation vector."""
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        idx = 0

        for i in range(HAND_SIZE):
            card = self.hand[i]
            if card is not None:
                obs[idx + 0] = 1.0
                obs[idx + 1] = norm_id(card.get("id", 0))
                obs[idx + 2] = norm(card["attack"], MAX_ATTACK)
                obs[idx + 3] = norm(card["health"], MAX_HEALTH)
                obs[idx + 4] = norm(card["play_cost"], MAX_COST)
                obs[idx + 5] = norm(card["burn_value"], MAX_BURN)
                obs[idx + 6] = 1.0 if self.mana >= card["play_cost"] else 0.0
                obs[idx + 7] = norm(len(card.get("battle_abilities", [])), MAX_ABILITIES)
                obs[idx + 8] = norm(len(card.get("shop_abilities", [])), MAX_ABILITIES)
            idx += HAND_FEATURES

        for i in range(BOARD_SIZE):
            unit = self.board[i]
            if unit is not None:
                obs[idx + 0] = 1.0
                obs[idx + 1] = norm_id(unit.get("id", 0))
                obs[idx + 2] = norm(unit["attack"], MAX_ATTACK)
                obs[idx + 3] = norm(unit["health"], MAX_HEALTH)
                obs[idx + 4] = norm(unit["play_cost"], MAX_COST)
                obs[idx + 5] = norm(unit["burn_value"], MAX_BURN)
                obs[idx + 6] = norm(len(unit.get("battle_abilities", [])), MAX_ABILITIES)
                obs[idx + 7] = norm(len(unit.get("shop_abilities", [])), MAX_ABILITIES)
            idx += BOARD_FEATURES

        s = self.state
        obs[idx + 0] = norm(self.mana, MAX_MANA)
        obs[idx + 1] = norm(self.mana_limit, MAX_MANA)
        obs[idx + 2] = norm(s["round"], MAX_ROUND)
        obs[idx + 3] = norm(s["lives"], MAX_LIVES)
        obs[idx + 4] = norm(s["wins"], MAX_WINS)
        obs[idx + 5] = norm(s["bag_count"], MAX_BAG)

        return obs

    def action_masks(self):
        """Compute valid action mask for all 66 actions."""
        if len(self.pending_actions) >= self.max_actions_per_turn:
            mask = np.zeros(NUM_ACTIONS, dtype=bool)
            mask[0] = True  # Force EndTurn
            return mask

        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        for i, (action_type, params) in enumerate(ACTION_TABLE):
            mask[i] = self._is_valid(action_type, params)
        return mask

    def _is_valid(self, action_type, params):
        if action_type == "EndTurn":
            return True
        if action_type == "BurnFromHand":
            return self.hand[params["hand_index"]] is not None
        if action_type == "PlayFromHand":
            i, j = params["hand_index"], params["board_slot"]
            card = self.hand[i]
            return card is not None and self.board[j] is None and self.mana >= card["play_cost"]
        if action_type == "BurnFromBoard":
            return self.board[params["board_slot"]] is not None
        if action_type == "SwapBoard":
            a, b = params["slot_a"], params["slot_b"]
            return self.board[a] is not None and self.board[b] is not None
        if action_type == "MoveBoard":
            f, t = params["from_slot"], params["to_slot"]
            return self.board[f] is not None and self.board[t] is not None
        return False

    def get_board_as_opponent(self):
        """Convert current board to opponent format for battle."""
        opponent = []
        for slot, unit in enumerate(self.board):
            if unit is not None:
                opponent.append({"card_id": extract_card_id(unit), "slot": slot})
        return opponent
