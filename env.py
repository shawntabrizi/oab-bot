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
from oab_shared import MatchedPool, PhaseController


class OABEnv(gym.Env):
    """Open Auto Battler environment with self-play.

    Uses Rust-side observation encoding and action masking for performance.
    Python handles only the gym interface and opponent pool integration.
    """

    metadata = {"render_modes": []}

    def __init__(self, set_id=0, board_pool=None,
                 action_cost=-0.01, repeat_penalty=-0.1,
                 play_reward=0.05,
                 reorder_penalty=-0.05,
                 board_unit_reward=0.03,
                 empty_board_penalty=-0.2,
                 phase_decomposition=False,
                 shop_action_limit=10,
                 position_action_limit=5,
                 max_rounds=20):
        super().__init__()
        self.set_id = set_id
        self.board_pool = board_pool
        self.action_cost = action_cost
        self.repeat_penalty = repeat_penalty
        self.play_reward = play_reward
        self.reorder_penalty = reorder_penalty
        self.board_unit_reward = board_unit_reward
        self.empty_board_penalty = empty_board_penalty
        self.max_rounds = max_rounds
        self.phase_controller = PhaseController(
            enabled=phase_decomposition,
            shop_action_limit=shop_action_limit,
            position_action_limit=position_action_limit,
        )

        # Create a temporary session to get dynamic obs_dim for this card set
        probe = oab_py.GameSession(0, set_id)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.phase_controller.obs_dim(probe.obs_dim()),),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(oab_py.GameSession.num_actions())

        self._session = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        game_seed = int(self.np_random.integers(0, 2**32))
        self._session = oab_py.GameSession(game_seed, self.set_id)
        self._session.set_max_rounds(self.max_rounds)
        self.phase_controller.reset()
        obs = self._get_observation()
        return obs, self._make_info()

    def step(self, action):
        action = int(action)

        if self.phase_controller.is_phase_transition_action(action):
            self.phase_controller.advance_to_position()
            return self._get_observation(), 0.0, False, False, self._make_info()

        if action == 0:  # EndTurn
            return self._do_end_turn()

        action_type, _ = self.describe_action(action)

        # Apply action locally in Rust
        changed = self._session.apply_action(action)
        self.phase_controller.record_action(action)
        reward = self._action_reward(action_type, changed)

        obs = self._get_observation()
        return obs, reward, False, False, self._make_info()

    def action_masks(self):
        base_mask = np.array(self._session.get_action_mask(), dtype=bool)
        return self.phase_controller.apply_mask(base_mask)

    def describe_action(self, action):
        return self.phase_controller.describe_action(int(action))

    # Expose card names for evaluate.py
    def get_hand_names(self):
        return self._session.get_hand_names()

    def get_board_names(self):
        return self._session.get_board_names()

    def _do_end_turn(self):
        rnd, lives, wins = self._session.get_info()
        board = json.loads(self._session.get_board_as_opponent())
        board_units = len(board)

        if self.board_pool is not None:
            opponent = self.board_pool.sample(rnd, wins, lives, exclude_board=board)
            opponent_json = json.dumps(opponent)
            if board:
                self.board_pool.add(rnd, wins, lives, board)
        else:
            opponent_json = "[]"

        reward, terminated, info_json = self._session.commit_turn_and_battle(opponent_json)
        end_turn_reward = self._end_turn_reward(board_units)
        reward += end_turn_reward
        self.phase_controller.finish_turn()
        obs = self._get_observation()
        info = json.loads(info_json)
        info["board_units"] = board_units
        info["end_turn_reward"] = end_turn_reward

        return obs, reward, terminated, False, info

    def _action_reward(self, action_type, changed):
        if not changed:
            return self.repeat_penalty

        if action_type in ("SwapBoard", "MoveBoard"):
            return self.reorder_penalty

        reward = self.action_cost
        if action_type == "PlayFromHand":
            reward += self.play_reward
        return reward

    def _end_turn_reward(self, board_units):
        if board_units == 0:
            return self.empty_board_penalty
        return board_units * self.board_unit_reward

    def _get_observation(self):
        obs = np.array(self._session.get_observation(), dtype=np.float32)
        return self.phase_controller.augment_observation(obs)

    def _make_info(self):
        rnd, lives, wins = self._session.get_info()
        info = {"round": rnd, "lives": lives, "wins": wins}
        if self.phase_controller.enabled:
            info["phase"] = self.phase_controller.phase
        return info
