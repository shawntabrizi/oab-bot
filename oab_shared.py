"""Shared Python-side utilities for OAB.

Observation encoding, action masking, and state tracking are handled
by the Rust oab_py module. This file contains only what Python needs:
- MatchedPool for opponent matchmaking (shared across DummyVecEnv agents)
- ACTION_TABLE for display/logging in play.py and evaluate.py
- PhaseController for optional shop/position decomposition
- Game constants for display code
"""

import dataclasses
import random
import threading
from collections import defaultdict
from itertools import combinations

import numpy as np

# ── Game Constants (for display code) ──

HAND_SIZE = 5
BOARD_SIZE = 5

# ── Action Table (for display/logging) ──
#
# Maps action index (0-65) to (action_type, params).
# Used by play.py for formatting and evaluate.py for card tracking.
# The Rust side has the authoritative action decoding.

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

SHOP_PHASE = "shop"
POSITION_PHASE = "position"
PHASE_FEATURES = 2
POSITION_ACTION_START = 36


@dataclasses.dataclass
class PhaseController:
    """Track optional shop/position decomposition for a single turn."""

    enabled: bool = False
    shop_action_limit: int = 10
    position_action_limit: int = 5
    phase: str = SHOP_PHASE
    shop_actions_used: int = 0
    position_actions_used: int = 0

    def reset(self):
        self.phase = SHOP_PHASE
        self.shop_actions_used = 0
        self.position_actions_used = 0

    def obs_dim(self, base_obs_dim):
        if not self.enabled:
            return base_obs_dim
        return base_obs_dim + PHASE_FEATURES

    def augment_observation(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        if not self.enabled:
            return obs

        phase_features = np.array(
            [1.0, 0.0] if self.phase == SHOP_PHASE else [0.0, 1.0],
            dtype=np.float32,
        )
        return np.concatenate((obs, phase_features)).astype(np.float32, copy=False)

    def apply_mask(self, base_mask):
        mask = np.asarray(base_mask, dtype=bool)
        if not self.enabled:
            return mask

        phased_mask = np.zeros_like(mask, dtype=bool)
        phased_mask[0] = True

        if self.phase == SHOP_PHASE:
            if self.shop_actions_used < self.shop_action_limit:
                phased_mask[:POSITION_ACTION_START] = mask[:POSITION_ACTION_START]
            return phased_mask

        if self.position_actions_used < self.position_action_limit:
            phased_mask[POSITION_ACTION_START:] = mask[POSITION_ACTION_START:]
        return phased_mask

    def describe_action(self, action_index):
        if self.enabled and self.phase == SHOP_PHASE and action_index == 0:
            return ("DoneShopping", {})
        return ACTION_TABLE[action_index]

    def is_phase_transition_action(self, action_index):
        return self.enabled and self.phase == SHOP_PHASE and action_index == 0

    def advance_to_position(self):
        if self.enabled:
            self.phase = POSITION_PHASE

    def record_action(self, action_index):
        if action_index == 0:
            return

        if self.enabled and self.phase == POSITION_PHASE:
            self.position_actions_used += 1
        else:
            self.shop_actions_used += 1

    def finish_turn(self):
        self.reset()


# ── Opponent Pool ──

class MatchedPool:
    """Thread-safe opponent pool with progression-based matchmaking.

    Boards are bucketed by (round, wins, lives). Sampling can mix in
    stronger same-round challengers even when exact-match boards exist,
    which acts like a lightweight league/hall-of-fame self-play setup.
    Buckets keep recent unique boards to reduce duplicate opponents
    dominating the pool.
    """

    def __init__(self, max_per_bucket=50, challenge_probability=0.35):
        self._pool = defaultdict(list)
        self._lock = threading.Lock()
        self._max = max_per_bucket
        self._challenge_probability = challenge_probability

    @staticmethod
    def _board_key(board):
        return tuple(
            (
                int(unit.get("slot", -1)),
                int(unit.get("card_id", -1)),
                int(unit.get("perm_attack", 0)),
                int(unit.get("perm_health", 0)),
            )
            for unit in board
        )

    def _filtered_candidates(self, boards, exclude_board):
        if not boards:
            return []
        if exclude_board is None:
            return list(boards)

        exclude_key = self._board_key(exclude_board)
        filtered = [board for board in boards if self._board_key(board) != exclude_key]
        return filtered or list(boards)

    @staticmethod
    def _is_stronger_progression(key, wins, lives):
        _, candidate_wins, candidate_lives = key
        return (
            candidate_wins > wins
            or (candidate_wins == wins and candidate_lives > lives)
        )

    @staticmethod
    def _board_strength(board):
        stat_total = sum(
            max(0, int(unit.get("attack", 0))) + max(0, int(unit.get("health", 0)))
            for unit in board
        )
        perm_bonus = sum(
            max(0, int(unit.get("perm_attack", 0))) + max(0, int(unit.get("perm_health", 0)))
            for unit in board
        )
        return 5.0 * len(board) + stat_total + 0.5 * perm_bonus

    @classmethod
    def _candidate_weight(cls, board, index):
        return 1.0 + index + cls._board_strength(board)

    def _choose(self, boards):
        if not boards:
            return []
        weights = [
            self._candidate_weight(board, idx)
            for idx, board in enumerate(boards)
        ]
        return random.choices(boards, weights=weights, k=1)[0]

    def _prune_bucket(self, bucket):
        if len(bucket) <= self._max:
            return list(bucket)

        recent_keep = max(1, self._max // 2)
        recent = list(bucket[-recent_keep:])
        recent_keys = {self._board_key(board) for board in recent}

        strongest = sorted(bucket, key=self._board_strength, reverse=True)
        preserved = []
        for board in strongest:
            key = self._board_key(board)
            if key in recent_keys:
                continue
            preserved.append(board)
            if len(preserved) >= self._max - recent_keep:
                break

        return preserved + recent

    def add(self, round_num, wins, lives, board):
        key = (round_num, wins, lives)
        board_key = self._board_key(board)
        with self._lock:
            bucket = self._pool[key]
            bucket[:] = [
                existing for existing in bucket
                if self._board_key(existing) != board_key
            ]
            bucket.append(board)
            if len(bucket) > self._max:
                bucket[:] = self._prune_bucket(bucket)

    def sample(self, round_num, wins, lives, exclude_board=None):
        with self._lock:
            # Tier 1: exact match
            exact = self._filtered_candidates(
                self._pool.get((round_num, wins, lives)),
                exclude_board,
            )

            # Tier 2: same round, stronger progression
            stronger_same_round = []
            for key, boards in self._pool.items():
                if key[0] != round_num:
                    continue
                if not self._is_stronger_progression(key, wins, lives):
                    continue
                stronger_same_round.extend(
                    self._filtered_candidates(boards, exclude_board)
                )
            if stronger_same_round:
                if not exact or random.random() < self._challenge_probability:
                    return self._choose(stronger_same_round)

            if exact:
                return self._choose(exact)

            # Tier 3: same round, wins ±1, any lives
            close = []
            for w_off in (0, 1, -1):
                for l in range(4):
                    bucket = self._filtered_candidates(
                        self._pool.get((round_num, wins + w_off, l)),
                        exclude_board,
                    )
                    if bucket:
                        close.extend(bucket)
            if close:
                return self._choose(close)

            # Tier 4: same round, any progression
            same_round = []
            for key, boards in self._pool.items():
                if key[0] == round_num:
                    same_round.extend(self._filtered_candidates(boards, exclude_board))
            if same_round:
                return self._choose(same_round)

            # Tier 5: anything
            all_boards = []
            for boards in self._pool.values():
                all_boards.extend(self._filtered_candidates(boards, exclude_board))
            return self._choose(all_boards) if all_boards else []

    def __len__(self):
        with self._lock:
            return sum(len(b) for b in self._pool.values())

    def snapshot(self):
        """Return {(round, wins, lives): count} for dashboard histograms."""
        with self._lock:
            return {key: len(boards) for key, boards in self._pool.items()}
