#!/usr/bin/env python3
"""Gymnasium environment for Open Auto Battler with self-play support.

Thin wrapper around the Rust oab_py.GameSession which handles observation
encoding, action masking, and state tracking natively. No JSON on the
training hot path.
"""

import json

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import oab_py
from oab_shared import MatchedPool


class OABEnv(gym.Env):
    """Open Auto Battler environment with self-play.

    Uses Rust-side observation encoding and action masking for performance.
    Python handles only the gym interface and opponent pool integration.
    """

    metadata = {"render_modes": []}

    def __init__(self, set_id=0, board_pool=None,
                 action_cost=-0.01, repeat_penalty=-0.1):
        super().__init__()
        self.set_id = set_id
        self.board_pool = board_pool
        self.action_cost = action_cost
        self.repeat_penalty = repeat_penalty

        # Create a temporary session to get dynamic obs_dim for this card set
        probe = oab_py.GameSession(0, set_id)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(probe.obs_dim(),), dtype=np.float32
        )
        self.action_space = spaces.Discrete(oab_py.GameSession.num_actions())

        self._session = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        game_seed = int(self.np_random.integers(0, 2**32))
        self._session = oab_py.GameSession(game_seed, self.set_id)
        obs = np.array(self._session.get_observation(), dtype=np.float32)
        return obs, self._make_info()

    def step(self, action):
        action = int(action)

        if action == 0:  # EndTurn
            return self._do_end_turn()

        # Apply action locally in Rust
        changed = self._session.apply_action(action)
        reward = self.action_cost if changed else self.repeat_penalty

        obs = np.array(self._session.get_observation(), dtype=np.float32)
        return obs, reward, False, False, self._make_info()

    def action_masks(self):
        return np.array(self._session.get_action_mask(), dtype=bool)

    # Expose card names for evaluate.py
    def get_hand_names(self):
        return self._session.get_hand_names()

    def get_board_names(self):
        return self._session.get_board_names()

    def _do_end_turn(self):
        rnd, lives, wins = self._session.get_info()
        board = json.loads(self._session.get_board_as_opponent())

        if self.board_pool is not None:
            self.board_pool.add(rnd, wins, lives, board)
            opponent = self.board_pool.sample(rnd, wins, lives)
            opponent_json = json.dumps(opponent)
        else:
            opponent_json = "[]"

        reward, terminated, info_json = self._session.commit_turn_and_battle(opponent_json)
        obs = np.array(self._session.get_observation(), dtype=np.float32)
        info = json.loads(info_json)

        return obs, reward, terminated, False, info

    def _make_info(self):
        rnd, lives, wins = self._session.get_info()
        return {"round": rnd, "lives": lives, "wins": wins}
