#!/usr/bin/env python3
"""Train an RL agent for Open Auto Battler using self-play.

Uses native PyO3 bindings — no game server needed.

Runs a lobby of N agents that share a BoardPool. Each agent plays
independently but faces opponents drawn from the pool of boards
produced by all agents across recent rounds.

Usage:
    python train.py
    python train.py --config experiment.json
    python train.py --timesteps 5000000
"""

import argparse
import os

import torch
torch.distributions.Distribution.set_default_validate_args(False)

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from config import TrainConfig, load_config, save_config
from env import OABEnv, BoardPool


def make_env(config, board_pool):
    """Factory for creating an OABEnv with shared board pool and config."""
    def _init():
        return OABEnv(
            set_id=config.set_id,
            board_pool=board_pool,
            action_cost=config.action_cost,
            repeat_penalty=config.repeat_penalty,
            max_actions_per_turn=config.max_actions_per_turn,
        )
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train OAB RL agent with self-play")
    parser.add_argument("--config", default=None,
                        help="Path to JSON config file (overrides defaults)")
    parser.add_argument("--set-id", type=int, default=None,
                        help="Card set ID to train on")
    parser.add_argument("--lobby-size", type=int, default=None,
                        help="Number of agents in self-play lobby")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Total training timesteps")
    parser.add_argument("--save-path", default=None,
                        help="Path to save trained model (without .zip)")
    parser.add_argument("--log-dir", default=None,
                        help="TensorBoard log directory")
    parser.add_argument("--resume", default=None,
                        help="Path to model to resume training from")
    args = parser.parse_args()

    # Load config from file, then override with CLI args
    config = load_config(args.config)
    if args.set_id is not None:
        config.set_id = args.set_id
    if args.lobby_size is not None:
        config.lobby_size = args.lobby_size
    if args.timesteps is not None:
        config.timesteps = args.timesteps
    if args.save_path is not None:
        config.save_path = args.save_path
    if args.log_dir is not None:
        config.log_dir = args.log_dir

    os.makedirs(os.path.dirname(config.save_path) or ".", exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    board_pool = BoardPool(max_size=config.board_pool_size)

    env_fns = [make_env(config, board_pool) for _ in range(config.lobby_size)]
    vec_env = DummyVecEnv(env_fns)

    if args.resume:
        print(f"Resuming training from {args.resume}")
        model = MaskablePPO.load(args.resume, env=vec_env)
    else:
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=config.log_dir,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=os.path.dirname(config.save_path) or "models",
        name_prefix="oab_checkpoint",
    )

    print(f"Training with {config.lobby_size}-agent self-play lobby (native engine)")
    print(f"  Timesteps: {config.timesteps}")
    print(f"  Set: {config.set_id}")
    print(f"  LR: {config.learning_rate}, Batch: {config.batch_size}, Epochs: {config.n_epochs}")
    print(f"  Action cost: {config.action_cost}, Repeat penalty: {config.repeat_penalty}")
    print(f"  TensorBoard: tensorboard --logdir {config.log_dir}")

    model.learn(
        total_timesteps=config.timesteps,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    model.save(config.save_path)
    config_path = f"{config.save_path}_config.json"
    save_config(config, config_path)
    print(f"Model saved to {config.save_path}.zip")
    print(f"Config saved to {config_path}")


if __name__ == "__main__":
    main()
