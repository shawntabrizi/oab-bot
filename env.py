#!/usr/bin/env python3
"""Gymnasium environment for Open Auto Battler with self-play support.

Two-phase turn design:
  1. Economy phase: burn/play/sell cards one at a time
  2. Ordering phase: pick a board permutation (single action)

Uses native PyO3 bindings (oab_py) for direct game engine access.
"""

import json
import random
import threading
from itertools import permutations

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
# Hand:  5 slots × 9 features = 45
# Board: 5 slots × 8 features = 40
# Scalars: 7 (mana, mana_limit, round, lives, wins, bag_count, phase)
# Total: 92

HAND_FEATURES = 9
BOARD_FEATURES = 8
SCALAR_FEATURES = 7

OBS_DIM = HAND_SIZE * HAND_FEATURES + BOARD_SIZE * BOARD_FEATURES + SCALAR_FEATURES


# ── Action Layout ──
#
# Economy actions (0-15):
#   0:     DoneEconomy (transition to ordering phase)
#   1-5:   BurnFromHand(0-4)
#   6-10:  PlayFromHand(0-4) — plays to first empty board slot
#   11-15: SellFromBoard(0-4)
#
# Ordering actions (16-136):
#   16:    KeepOrder (submit turn as-is)
#   17+:   Permutation indices (up to 120 for 5 units)
#
# Total: 137 actions

ECONOMY_DONE = 0
BURN_START = 1
BURN_END = 5
PLAY_START = 6
PLAY_END = 10
SELL_START = 11
SELL_END = 15

ORDER_KEEP = 16
ORDER_PERM_START = 17

# Pre-compute all permutations for 1-5 units
# _PERMS[n] = list of permutation tuples for n items
_PERMS = {n: list(permutations(range(n))) for n in range(1, BOARD_SIZE + 1)}

# Max permutations = 5! = 120, plus KeepOrder = 121 ordering actions
MAX_ORDER_ACTIONS = 121
NUM_ACTIONS = 16 + MAX_ORDER_ACTIONS  # 137


# ── Shared Opponent Pool ──

class BoardPool:
    """Thread-safe pool of opponent boards from recent rounds."""

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
    """Open Auto Battler environment with two-phase turns.

    Phase 1 (Economy): Burn, play, sell cards. Action space: 0-15.
    Phase 2 (Ordering): Pick board permutation. Action space: 16-136.

    After ordering, the turn is submitted automatically (shop + battle).
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
        self._phase = "economy"  # "economy" or "ordering"

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        game_seed = int(self.np_random.integers(0, 2**32))
        self._session = oab_py.GameSession(game_seed, self.set_id)
        self._state = json.loads(self._session.get_state())
        self._sync_local_state()
        self._pending_actions = []
        self._phase = "economy"
        return self._encode_obs(), self._make_info()

    def step(self, action):
        action = int(action)

        if self._phase == "economy":
            return self._step_economy(action)
        else:
            return self._step_ordering(action)

    def action_masks(self):
        mask = np.zeros(NUM_ACTIONS, dtype=bool)

        if self._phase == "economy":
            # DoneEconomy is always valid
            mask[ECONOMY_DONE] = True

            # BurnFromHand
            for i in range(HAND_SIZE):
                if self._hand[i] is not None:
                    mask[BURN_START + i] = True

            # PlayFromHand (to first empty slot)
            has_empty = any(s is None for s in self._board)
            for i in range(HAND_SIZE):
                card = self._hand[i]
                if card is not None and has_empty and self._mana >= card["play_cost"]:
                    mask[PLAY_START + i] = True

            # SellFromBoard
            for i in range(BOARD_SIZE):
                if self._board[i] is not None:
                    mask[SELL_START + i] = True

        else:  # ordering phase
            occupied = [i for i in range(BOARD_SIZE) if self._board[i] is not None]
            n = len(occupied)

            # KeepOrder is always valid
            mask[ORDER_KEEP] = True

            # Permutations (skip identity which is KeepOrder)
            if n >= 2:
                identity = tuple(range(n))
                for perm_idx, perm in enumerate(_PERMS[n]):
                    if perm != identity:
                        action_idx = ORDER_PERM_START + perm_idx
                        if action_idx < NUM_ACTIONS:
                            mask[action_idx] = True

        return mask

    # ── Economy Phase ──

    def _step_economy(self, action):
        if action == ECONOMY_DONE:
            # Transition to ordering phase
            self._phase = "ordering"
            return self._encode_obs(), 0.0, False, False, self._make_info()

        if BURN_START <= action <= BURN_END:
            i = action - BURN_START
            card = self._hand[i]
            if card:
                self._pending_actions.append(("BurnFromHand", {"hand_index": i}))
                self._mana = min(self._mana + card["burn_value"], self._mana_limit)
                self._hand[i] = None

        elif PLAY_START <= action <= PLAY_END:
            i = action - PLAY_START
            card = self._hand[i]
            if card:
                # Find first empty board slot
                slot = next((j for j in range(BOARD_SIZE) if self._board[j] is None), None)
                if slot is not None:
                    self._pending_actions.append(
                        ("PlayFromHand", {"hand_index": i, "board_slot": slot})
                    )
                    self._mana -= card["play_cost"]
                    self._board[slot] = card
                    self._hand[i] = None

        elif SELL_START <= action <= SELL_END:
            j = action - SELL_START
            unit = self._board[j]
            if unit:
                self._pending_actions.append(("BurnFromBoard", {"board_slot": j}))
                self._mana = min(self._mana + unit["burn_value"], self._mana_limit)
                self._board[j] = None

        return self._encode_obs(), 0.0, False, False, self._make_info()

    # ── Ordering Phase ──

    def _step_ordering(self, action):
        occupied = [i for i in range(BOARD_SIZE) if self._board[i] is not None]
        n = len(occupied)

        if action != ORDER_KEEP and n >= 2:
            perm_idx = action - ORDER_PERM_START
            if 0 <= perm_idx < len(_PERMS[n]):
                perm = _PERMS[n][perm_idx]
                # Build swap actions to achieve the target permutation
                units = [self._board[occupied[j]] for j in range(n)]
                target_slots = [occupied[p] for p in perm]

                # Clear occupied slots and place in new order
                for slot in occupied:
                    self._board[slot] = None
                for unit, slot in zip(units, target_slots):
                    self._board[slot] = unit

                # Generate SwapBoard actions for the server
                self._generate_swap_actions(occupied, perm)

        # Submit the full turn
        return self._submit_turn()

    def _generate_swap_actions(self, occupied, perm):
        """Generate minimal SwapBoard actions to achieve a permutation.

        Uses selection sort: for each position, swap the correct element into place.
        """
        current = list(range(len(occupied)))
        for i in range(len(current)):
            target = perm[i]
            if current[i] != target:
                # Find where target currently is
                j = current.index(target)
                # Swap in current tracking
                current[i], current[j] = current[j], current[i]
                # Record the swap action using actual board slots
                self._pending_actions.append(
                    ("SwapBoard", {"slot_a": occupied[i], "slot_b": occupied[j]})
                )

    # ── Turn Submission ──

    def _submit_turn(self):
        """Submit all pending actions to the game engine."""
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
            try:
                shop_result = json.loads(self._session.shop("[]"))
            except Exception as e:
                return self._encode_obs(), -1.0, True, False, {
                    "error": str(e), **self._make_info()
                }

        self._state = shop_result
        self._sync_local_state()

        # Post board to pool and sample opponent
        my_board = self._get_board_as_opponent()
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
        self._phase = "economy"

        info = self._make_info()
        info["battle_result"] = result["battle_result"]
        if terminated:
            info["game_result"] = result.get("game_result")

        return self._encode_obs(), reward, terminated, False, info

    # ── State Tracking ──

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

    def _get_board_as_opponent(self):
        opponent = []
        for slot, unit in enumerate(self._board):
            if unit is not None:
                opponent.append({"card_id": _extract_card_id(unit), "slot": slot})
        return opponent

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
        obs[idx + 6] = 1.0 if self._phase == "ordering" else 0.0

        return obs

    def _make_info(self):
        s = self._state
        return {
            "round": s["round"],
            "lives": s["lives"],
            "wins": s["wins"],
            "phase": self._phase,
        }


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
