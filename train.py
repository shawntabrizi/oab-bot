#!/usr/bin/env python3
"""Train an RL agent for Open Auto Battler using MaskablePPO.

Usage:
    # Start the game server first:
    cd /path/to/open-auto-battler && cargo run -p oab-server --release

    # Then train:
    python train.py
    python train.py --timesteps 1000000 --set-id 1
"""

import argparse
import os

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback

from env import OABEnv


def main():
    parser = argparse.ArgumentParser(description="Train OAB RL agent")
    parser.add_argument("--server-url", default="http://localhost:3000",
                        help="Game server URL")
    parser.add_argument("--set-id", type=int, default=0,
                        help="Card set ID to train on")
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

    env = OABEnv(server_url=args.server_url, set_id=args.set_id)

    if args.resume:
        print(f"Resuming training from {args.resume}")
        model = MaskablePPO.load(args.resume, env=env)
    else:
        model = MaskablePPO(
            "MlpPolicy",
            env,
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

    print(f"Training for {args.timesteps} timesteps on set {args.set_id}...")
    print(f"Server: {args.server_url}")
    print(f"TensorBoard: tensorboard --logdir {args.log_dir}")

    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    model.save(args.save_path)
    print(f"Model saved to {args.save_path}.zip")


if __name__ == "__main__":
    main()
