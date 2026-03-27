#!/usr/bin/env python3 -u
"""Evaluate a trained RL agent on Open Auto Battler.

Uses native PyO3 bindings. Runs single-agent games with a shared
board pool for self-play opponents. Reports win/loss stats and
per-card pick/play/burn rates for balance analysis.

Usage:
    python evaluate.py
    python evaluate.py --model-path models/oab_agent_5m --num-games 100
    python evaluate.py --num-games 200 --output results.json
"""

import argparse
import csv
import json
import math
from collections import defaultdict

import numpy as np
from sb3_contrib import MaskablePPO

from config import load_saved_model_config
from env import OABEnv
from oab_shared import MatchedPool


def wilson_ci(successes, total, z=1.96):
    """Wilson score 95% confidence interval for a proportion.

    Returns (lower, upper) as percentages.
    """
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return (max(0.0, center - spread) * 100, min(1.0, center + spread) * 100)


def run_evaluation(model, config, num_games, shared_pool=False):
    """Run games and collect statistics."""
    pool = MatchedPool(
        max_per_bucket=config.max_boards_per_bucket,
        challenge_probability=config.challenge_probability,
    )

    results = []
    card_stats = defaultdict(lambda: {
        "seen": 0,
        "played": 0,
        "burned": 0,
        "games_played": 0,
        "games_won": 0,
    })

    for game in range(1, num_games + 1):
        if not shared_pool:
            pool = MatchedPool(
                max_per_bucket=config.max_boards_per_bucket,
                challenge_probability=config.challenge_probability,
            )
        env = OABEnv(
            set_id=config.set_id,
            board_pool=pool,
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

        obs, info = env.reset()
        total_reward = 0
        steps = 0
        cards_played_this_game = set()

        # Record initial hand (count every instance, not unique per game)
        for name in env.get_hand_names():
            if name is not None:
                card_stats[name]["seen"] += 1

        while True:
            masks = env.action_masks()
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            action = int(action)

            # Track card-level actions before stepping
            action_type, params = env.describe_action(action)
            hand_names = env.get_hand_names()
            if action_type == "BurnFromHand":
                name = hand_names[params["hand_index"]]
                if name:
                    card_stats[name]["burned"] += 1
            elif action_type == "PlayFromHand":
                name = hand_names[params["hand_index"]]
                if name:
                    card_stats[name]["played"] += 1
                    cards_played_this_game.add(name)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            # After EndTurn, record new hand cards (every instance)
            if action_type == "EndTurn" and not terminated:
                for name in env.get_hand_names():
                    if name is not None:
                        card_stats[name]["seen"] += 1

            if terminated or truncated:
                game_result = info.get("game_result", "unknown")
                wins = info.get("wins", 0)
                is_victory = game_result == "victory"

                for cname in cards_played_this_game:
                    card_stats[cname]["games_played"] += 1
                    if is_victory:
                        card_stats[cname]["games_won"] += 1

                results.append({
                    "game": game,
                    "result": game_result,
                    "wins": wins,
                    "reward": total_reward,
                    "steps": steps,
                })
                print(
                    f"  Game {game:3d}: {game_result:7s}  "
                    f"wins={wins:2d}  reward={total_reward:+.0f}  steps={steps}"
                )
                break

    return results, dict(card_stats)


def print_summary(results, card_stats, set_id):
    num_games = len(results)
    if num_games == 0:
        print("No games completed.")
        return

    victories = sum(1 for r in results if r["result"] == "victory")
    defeats = sum(1 for r in results if r["result"] == "defeat")
    win_pcts = [r["wins"] for r in results]
    avg_wins = np.mean(win_pcts)
    sem_wins = np.std(win_pcts, ddof=1) / np.sqrt(num_games) if num_games > 1 else 0
    avg_reward = np.mean([r["reward"] for r in results])
    avg_steps = np.mean([r["steps"] for r in results])

    victory_lo, victory_hi = wilson_ci(victories, num_games)

    print(f"\n{'=' * 60}")
    print(f"  Evaluation Results (Set {set_id}, {num_games} games)")
    print(f"{'=' * 60}")
    print(f"  Full victories:    {victories}/{num_games} ({100*victories/num_games:.1f}%)"
          f"  95% CI: [{victory_lo:.1f}%, {victory_hi:.1f}%]")
    print(f"  Defeats:           {defeats} ({100*defeats/num_games:.1f}%)")
    print(f"  Avg wins/game:     {avg_wins:.2f} +/- {sem_wins:.2f}")
    print(f"  Avg reward/game:   {avg_reward:+.2f}")
    print(f"  Avg steps/game:    {avg_steps:.1f}")

    if card_stats:
        print(f"\n{'=' * 60}")
        print(f"  Card Statistics")
        print(f"{'=' * 60}")
        print(
            f"  {'Card':<20s} {'Seen':>5s} {'Play':>5s} {'Burn':>5s} "
            f"{'Play%':>6s} {'WinRate':>8s} {'WR 95% CI':>14s}"
        )
        print(f"  {'-'*20} {'-'*5} {'-'*5} {'-'*5} {'-'*6} {'-'*8} {'-'*14}")

        for name in sorted(card_stats.keys()):
            s = card_stats[name]
            play_pct = 100 * s["played"] / s["seen"] if s["seen"] > 0 else 0
            win_rate = (
                100 * s["games_won"] / s["games_played"]
                if s["games_played"] > 0
                else 0
            )
            wr_lo, wr_hi = wilson_ci(s["games_won"], s["games_played"])
            ci_str = f"[{wr_lo:4.1f},{wr_hi:5.1f}]" if s["games_played"] > 0 else "       -"
            print(
                f"  {name:<20s} {s['seen']:5d} {s['played']:5d} {s['burned']:5d} "
                f"{play_pct:5.1f}% {win_rate:6.1f}%  {ci_str}"
            )


def export_results(results, card_stats, output_path):
    """Export results and card stats to JSON or CSV."""
    data = {
        "games": results,
        "card_stats": card_stats,
    }

    if output_path.endswith(".json"):
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        print(f"\nExported to {output_path}")

    elif output_path.endswith(".csv"):
        # Write two CSVs: games and card stats
        games_path = output_path
        cards_path = output_path.replace(".csv", "_cards.csv")

        with open(games_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["game", "result", "wins", "reward", "steps"])
            writer.writeheader()
            writer.writerows(results)

        with open(cards_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["card", "seen", "played", "burned", "games_played", "games_won"])
            for name in sorted(card_stats.keys()):
                s = card_stats[name]
                writer.writerow([name, s["seen"], s["played"], s["burned"],
                                 s["games_played"], s["games_won"]])

        print(f"\nExported to {games_path} and {cards_path}")
    else:
        print(f"\nUnknown output format: {output_path} (use .json or .csv)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate OAB RL agent")
    parser.add_argument("--model-path", default="models/oab_agent",
                        help="Path to trained model (without .zip)")
    parser.add_argument("--set-id", type=int, default=None,
                        help="Card set ID")
    parser.add_argument("--num-games", type=int, default=100,
                        help="Number of games to evaluate")
    parser.add_argument("--shared-pool", action="store_true",
                        help="Share board pool across games (default: fresh pool per game)")
    parser.add_argument("--output", default=None,
                        help="Export results to file (.json or .csv)")
    args = parser.parse_args()

    model = MaskablePPO.load(args.model_path)
    config = load_saved_model_config(args.model_path)
    if args.set_id is not None:
        config.set_id = args.set_id

    print(f"Evaluating {args.model_path} for {args.num_games} games on set {config.set_id}...")
    if config.phase_decomposition:
        print(
            "Using phase decomposition: "
            f"{config.shop_action_limit} shop / {config.position_action_limit} position"
        )
    print(
        "Reward shaping: "
        f"action {config.action_cost}, repeat {config.repeat_penalty}, "
        f"play +{config.play_reward}, reorder {config.reorder_penalty}, "
        f"board +{config.board_unit_reward}/unit, empty {config.empty_board_penalty}; "
        f"round cap {config.max_rounds}"
    )
    print(
        "Opponent pool: "
        f"max {config.max_boards_per_bucket}/bucket, "
        f"challenge {config.challenge_probability:.2f}"
    )
    results, card_stats = run_evaluation(model, config, args.num_games,
                                          shared_pool=args.shared_pool)
    print_summary(results, card_stats, config.set_id)

    if args.output:
        export_results(results, card_stats, args.output)


if __name__ == "__main__":
    main()
