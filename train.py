#!/usr/bin/env python3
"""Train an RL agent for Open Auto Battler using self-play.

Uses native PyO3 bindings — no game server needed.

Runs a lobby of N agents that share a MatchedPool. Each agent plays
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

from config import TrainConfig, load_config, load_saved_model_config, save_config
from env import OABEnv
from oab_shared import MatchedPool


def build_env(config, board_pool):
    """Create a concrete OABEnv from config."""
    return OABEnv(
        set_id=config.set_id,
        board_pool=board_pool,
        action_cost=config.action_cost,
        repeat_penalty=config.repeat_penalty,
        play_reward=config.play_reward,
        reorder_penalty=config.reorder_penalty,
        board_unit_reward=config.board_unit_reward,
        empty_board_penalty=config.empty_board_penalty,
        phase_decomposition=config.phase_decomposition,
        shop_action_limit=config.shop_action_limit,
        position_action_limit=config.position_action_limit,
        max_rounds=config.max_rounds,
    )


def make_env(config, board_pool):
    """Factory for creating an OABEnv with shared board pool and config."""
    def _init():
        return build_env(config, board_pool)
    return _init


def seed_pool_from_models(train_config, board_pool):
    """Populate the shared board pool using historical checkpoints."""
    if not train_config.seed_opponent_models:
        return

    for model_path in train_config.seed_opponent_models:
        seed_config = load_saved_model_config(model_path)
        if seed_config.set_id != train_config.set_id:
            raise ValueError(
                f"Seed model {model_path} uses set {seed_config.set_id}, "
                f"but training set is {train_config.set_id}"
            )

        model = MaskablePPO.load(model_path)
        env = build_env(seed_config, board_pool)
        boards_before = len(board_pool)

        for game_idx in range(train_config.seed_games_per_model):
            obs, _ = env.reset(seed=game_idx)
            while True:
                masks = env.action_masks()
                action, _ = model.predict(obs, action_masks=masks, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(int(action))
                if terminated or truncated:
                    break

        boards_after = len(board_pool)
        print(
            f"  Seeded pool from {model_path}: "
            f"+{boards_after - boards_before} boards ({boards_after} total)"
        )


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
    parser.add_argument("--phase-decomposition", action="store_true",
                        help="Split each turn into shop and positioning phases")
    parser.add_argument("--shop-action-limit", type=int, default=None,
                        help="Max non-positioning actions before only DoneShopping remains")
    parser.add_argument("--position-action-limit", type=int, default=None,
                        help="Max positioning actions before only EndTurn remains")
    parser.add_argument("--max-rounds", type=int, default=None,
                        help="Automatic loss if the game reaches this round without victory")
    parser.add_argument("--play-reward", type=float, default=None,
                        help="Extra reward for successfully playing a card")
    parser.add_argument("--reorder-penalty", type=float, default=None,
                        help="Reward used for successful swap/move actions")
    parser.add_argument("--board-unit-reward", type=float, default=None,
                        help="Reward granted per occupied board slot when ending the turn")
    parser.add_argument("--empty-board-penalty", type=float, default=None,
                        help="Penalty applied when ending the turn with an empty board")
    parser.add_argument("--challenge-probability", type=float, default=None,
                        help="Chance to sample a stronger same-round challenger over an exact match")
    parser.add_argument("--seed-games-per-model", type=int, default=None,
                        help="How many deterministic games to use when seeding the pool per model")
    parser.add_argument("--seed-opponent-model", action="append", default=None,
                        help="Historical model to pre-seed the opponent pool from; repeatable")
    args = parser.parse_args()

    # Prefer the saved model config when resuming unless the caller supplied a config file.
    if args.resume and args.config is None:
        config = load_saved_model_config(args.resume)
    else:
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
    if args.phase_decomposition:
        config.phase_decomposition = True
    if args.shop_action_limit is not None:
        config.shop_action_limit = args.shop_action_limit
    if args.position_action_limit is not None:
        config.position_action_limit = args.position_action_limit
    if args.max_rounds is not None:
        config.max_rounds = args.max_rounds
    if args.play_reward is not None:
        config.play_reward = args.play_reward
    if args.reorder_penalty is not None:
        config.reorder_penalty = args.reorder_penalty
    if args.board_unit_reward is not None:
        config.board_unit_reward = args.board_unit_reward
    if args.empty_board_penalty is not None:
        config.empty_board_penalty = args.empty_board_penalty
    if args.challenge_probability is not None:
        config.challenge_probability = args.challenge_probability
    if args.seed_games_per_model is not None:
        config.seed_games_per_model = args.seed_games_per_model
    if args.seed_opponent_model is not None:
        config.seed_opponent_models = args.seed_opponent_model

    os.makedirs(os.path.dirname(config.save_path) or ".", exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    board_pool = MatchedPool(
        max_per_bucket=config.max_boards_per_bucket,
        challenge_probability=config.challenge_probability,
    )
    if config.seed_opponent_models:
        print("Seeding shared pool from historical models:")
        seed_pool_from_models(config, board_pool)

    env_fns = [make_env(config, board_pool) for _ in range(config.lobby_size)]
    vec_env = DummyVecEnv(env_fns)

    if args.resume:
        print(f"Resuming training from {args.resume}")
        model = MaskablePPO.load(args.resume, env=vec_env)
        model.tensorboard_log = config.log_dir
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
            ent_coef=config.ent_coef,
            policy_kwargs=dict(net_arch=[256, 128]),
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
    print(
        "  Rewards: "
        f"action {config.action_cost}, repeat {config.repeat_penalty}, "
        f"play +{config.play_reward}, reorder {config.reorder_penalty}, "
        f"board +{config.board_unit_reward}/unit, empty {config.empty_board_penalty}"
    )
    print(
        f"  Pool: matched (max {config.max_boards_per_bucket}/bucket, "
        f"challenge {config.challenge_probability:.2f})"
    )
    if config.seed_opponent_models:
        print(
            f"  Historical seeding: {len(config.seed_opponent_models)} model(s), "
            f"{config.seed_games_per_model} games/model"
        )
    if config.phase_decomposition:
        print(
            "  Phase decomposition: shop -> position -> battle "
            f"({config.shop_action_limit} shop / {config.position_action_limit} position)"
        )
    print(f"  Round cap: {config.max_rounds}")
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
