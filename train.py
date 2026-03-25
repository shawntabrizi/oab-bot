#!/usr/bin/env python3
"""Train an RL agent for Open Auto Battler using self-play.

Uses native PyO3 bindings — no game server needed.

Runs a lobby of N agents that share a BoardPool. Each agent plays
independently but faces opponents drawn from the pool of boards
produced by all agents across recent rounds.

Usage:
    python train.py
    python train.py --lobby-size 10 --timesteps 5000000
"""

import argparse
import os

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from env import OABEnv, BoardPool


def make_env(set_id, board_pool):
    """Factory for creating an OABEnv with shared board pool."""
    def _init():
        return OABEnv(set_id=set_id, board_pool=board_pool)
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train OAB RL agent with self-play")
    parser.add_argument("--set-id", type=int, default=0,
                        help="Card set ID to train on")
    parser.add_argument("--lobby-size", type=int, default=10,
                        help="Number of agents in self-play lobby")
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Total training timesteps")
    parser.add_argument("--save-path", default="models/oab_agent",
                        help="Path to save trained model (without .zip)")
    parser.add_argument("--log-dir", default="logs",
                        help="TensorBoard log directory")
    parser.add_argument("--resume", default=None,
                        help="Path to model to resume training from")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    board_pool = BoardPool(max_size=200)

    env_fns = [make_env(args.set_id, board_pool) for _ in range(args.lobby_size)]
    vec_env = DummyVecEnv(env_fns)

    if args.resume:
        print(f"Resuming training from {args.resume}")
        model = MaskablePPO.load(args.resume, env=vec_env)
    else:
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=args.log_dir,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=os.path.dirname(args.save_path) or "models",
        name_prefix="oab_checkpoint",
    )

    print(f"Training with {args.lobby_size}-agent self-play lobby (native engine)")
    print(f"  Timesteps: {args.timesteps}")
    print(f"  Set: {args.set_id}")
    print(f"  TensorBoard: tensorboard --logdir {args.log_dir}")

    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    model.save(args.save_path)
    print(f"Model saved to {args.save_path}.zip")


if __name__ == "__main__":
    main()
