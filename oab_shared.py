"""Shared Python-side utilities for OAB.

Observation encoding, action masking, and state tracking are handled
by the Rust oab_py module. This file contains only what Python needs:
- MatchedPool for opponent matchmaking (shared across DummyVecEnv agents)
- ACTION_TABLE for display/logging in play.py and evaluate.py
- Game constants for display code
"""

import random
import threading
from collections import defaultdict
from itertools import combinations

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


# ── Opponent Pool ──

class MatchedPool:
    """Thread-safe opponent pool with progression-based matchmaking.

    Boards are bucketed by (round, wins, lives). Sampling uses tiered
    fallback: exact match > same round + similar wins > same round > any.
    """

    def __init__(self, max_per_bucket=50):
        self._pool = defaultdict(list)
        self._lock = threading.Lock()
        self._max = max_per_bucket

    def add(self, round_num, wins, lives, board):
        key = (round_num, wins, lives)
        with self._lock:
            bucket = self._pool[key]
            bucket.append(board)
            if len(bucket) > self._max:
                bucket[:] = bucket[-self._max // 2 :]

    def sample(self, round_num, wins, lives):
        with self._lock:
            # Tier 1: exact match
            exact = self._pool.get((round_num, wins, lives))
            if exact:
                return random.choice(exact)

            # Tier 2: same round, wins ±1, any lives
            close = []
            for w_off in (0, 1, -1):
                for l in range(4):
                    bucket = self._pool.get((round_num, wins + w_off, l))
                    if bucket:
                        close.extend(bucket)
            if close:
                return random.choice(close)

            # Tier 3: same round, any progression
            same_round = []
            for key, boards in self._pool.items():
                if key[0] == round_num:
                    same_round.extend(boards)
            if same_round:
                return random.choice(same_round)

            # Tier 4: anything
            all_boards = [b for boards in self._pool.values() for b in boards]
            return random.choice(all_boards) if all_boards else []

    def __len__(self):
        with self._lock:
            return sum(len(b) for b in self._pool.values())
