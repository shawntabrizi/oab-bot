#!/usr/bin/env python3
"""Gymnasium environment for Open Auto Battler with self-play support.

Flat 66-action design with action masking. All action types available
simultaneously. Small per-action cost discourages wasted actions.
15-action safety limit prevents infinite loops.

Uses native PyO3 bindings (oab_py) for direct game engine access.
"""

import json
import random
import threading

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import oab_py
from oab_shared import (
    HAND_SIZE,
    BOARD_SIZE,
    OBS_DIM,
    NUM_ACTIONS,
    ACTION_TABLE,
    DEFAULT_ACTION_COST,
    DEFAULT_REPEAT_PENALTY,
    DEFAULT_MAX_ACTIONS_PER_TURN,
    GameStateTracker,
    extract_card_id,
)


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
    """Open Auto Battler environment with self-play.

    Flat 66-action space. All action types available simultaneously.
    Action limit per turn forces EndTurn to prevent loops.
    Small per-action cost and no-op penalty discourage wasted actions.
    """

    metadata = {"render_modes": []}

    def __init__(self, set_id=0, board_pool=None,
                 action_cost=DEFAULT_ACTION_COST,
                 repeat_penalty=DEFAULT_REPEAT_PENALTY,
                 max_actions_per_turn=DEFAULT_MAX_ACTIONS_PER_TURN):
        super().__init__()
        self.set_id = set_id
        self.board_pool = board_pool
        self.action_cost = action_cost
        self.repeat_penalty = repeat_penalty

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self._tracker = GameStateTracker(max_actions_per_turn=max_actions_per_turn)
        self._session = None

    # Expose tracker state for evaluate.py card tracking
    @property
    def _hand(self):
        return self._tracker.hand

    @property
    def _board(self):
        return self._tracker.board

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        game_seed = int(self.np_random.integers(0, 2**32))
        self._session = oab_py.GameSession(game_seed, self.set_id)
        state = json.loads(self._session.get_state())
        self._tracker.sync(state)
        return self._tracker.encode_obs(), self._make_info()

    def step(self, action):
        action_type, params = ACTION_TABLE[int(action)]

        if action_type == "EndTurn":
            return self._do_end_turn()

        # Snapshot board before action (for no-op detection)
        board_before = [id(u) for u in self._tracker.board]

        self._tracker.pending_actions.append((action_type, params))
        self._tracker.apply_action(action_type, params)

        # Determine action reward
        board_after = [id(u) for u in self._tracker.board]
        if action_type in ("SwapBoard", "MoveBoard") and board_before == board_after:
            reward = self.repeat_penalty
        else:
            reward = self.action_cost

        return self._tracker.encode_obs(), reward, False, False, self._make_info()

    def action_masks(self):
        return self._tracker.action_masks()

    # ── Turn Submission ──

    def _do_end_turn(self):
        server_actions = []
        for action_type, params in self._tracker.pending_actions:
            action = {"type": action_type}
            action.update(params)
            server_actions.append(action)

        actions_json = json.dumps(server_actions)

        try:
            shop_result = json.loads(self._session.shop(actions_json))
        except Exception:
            try:
                shop_result = json.loads(self._session.shop("[]"))
            except Exception as e:
                return self._tracker.encode_obs(), -1.0, True, False, {
                    "error": str(e), **self._make_info()
                }

        state = shop_result
        self._tracker.sync(state)

        my_board = self._tracker.get_board_as_opponent()
        if self.board_pool is not None:
            self.board_pool.add(my_board)
            opponent = self.board_pool.sample()
        else:
            opponent = []

        opponent_json = json.dumps(opponent)

        try:
            result = json.loads(self._session.battle(opponent_json))
        except Exception as e:
            return self._tracker.encode_obs(), -1.0, True, False, {
                "error": str(e), **self._make_info()
            }

        reward = float(result["reward"])
        terminated = result["game_over"]

        self._tracker.sync(result["state"])

        info = self._make_info()
        info["battle_result"] = result["battle_result"]
        if terminated:
            info["game_result"] = result.get("game_result")

        return self._tracker.encode_obs(), reward, terminated, False, info

    # ── Info ──

    def _make_info(self):
        s = self._tracker.state
        return {"round": s["round"], "lives": s["lives"], "wins": s["wins"]}
