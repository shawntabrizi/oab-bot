#!/usr/bin/env python3 -u
"""Evaluate a trained RL agent on Open Auto Battler.

Uses native PyO3 bindings. Runs single-agent games with a shared
board pool for self-play opponents. Reports win/loss stats and
per-card pick/play/burn rates for balance analysis.

Usage:
    python evaluate.py
    python evaluate.py --model-path models/oab_agent_5m --num-games 100
"""

import argparse
from collections import defaultdict

import numpy as np
from sb3_contrib import MaskablePPO

from env import (
    OABEnv, BoardPool,
    ECONOMY_DONE, BURN_START, BURN_END, PLAY_START, PLAY_END,
    SELL_START, SELL_END, ORDER_KEEP,
)


def run_evaluation(model, set_id, num_games):
    """Run games and collect statistics."""
    board_pool = BoardPool(max_size=200)
    env = OABEnv(set_id=set_id, board_pool=board_pool)

    results = []
    card_stats = defaultdict(lambda: {
        "seen": 0,
        "played": 0,
        "burned": 0,
        "games_played": 0,
        "games_won": 0,
    })

    for game in range(1, num_games + 1):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        cards_played_this_game = set()
        cards_seen_this_game = set()

        # Record initial hand
        for card in env._hand:
            if card is not None:
                card_stats[card["name"]]["seen"] += 1
                cards_seen_this_game.add(card["name"])

        while True:
            masks = env.action_masks()
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            action = int(action)

            # Track card-level actions before stepping
            if BURN_START <= action <= BURN_END:
                i = action - BURN_START
                card = env._hand[i]
                if card:
                    card_stats[card["name"]]["burned"] += 1
            elif PLAY_START <= action <= PLAY_END:
                i = action - PLAY_START
                card = env._hand[i]
                if card:
                    card_stats[card["name"]]["played"] += 1
                    cards_played_this_game.add(card["name"])

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            # After ordering action (turn submitted), record new hand cards
            if action >= ORDER_KEEP and not terminated:
                for card in env._hand:
                    if card is not None:
                        cname = card["name"]
                        if cname not in cards_seen_this_game:
                            card_stats[cname]["seen"] += 1
                            cards_seen_this_game.add(cname)

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
    avg_wins = np.mean([r["wins"] for r in results])
    avg_reward = np.mean([r["reward"] for r in results])
    avg_steps = np.mean([r["steps"] for r in results])

    print(f"\n{'=' * 50}")
    print(f"  Evaluation Results (Set {set_id})")
    print(f"{'=' * 50}")
    print(f"  Games played:      {num_games}")
    print(f"  Full victories:    {victories} ({100*victories/num_games:.1f}%)")
    print(f"  Defeats:           {defeats} ({100*defeats/num_games:.1f}%)")
    print(f"  Avg wins/game:     {avg_wins:.2f}")
    print(f"  Avg reward/game:   {avg_reward:+.2f}")
    print(f"  Avg steps/game:    {avg_steps:.1f}")

    if card_stats:
        print(f"\n{'=' * 50}")
        print(f"  Card Statistics")
        print(f"{'=' * 50}")
        print(
            f"  {'Card':<20s} {'Seen':>5s} {'Play':>5s} {'Burn':>5s} "
            f"{'Play%':>6s} {'WinRate':>8s}"
        )
        print(f"  {'-'*20} {'-'*5} {'-'*5} {'-'*5} {'-'*6} {'-'*8}")

        for name in sorted(card_stats.keys()):
            s = card_stats[name]
            play_pct = 100 * s["played"] / s["seen"] if s["seen"] > 0 else 0
            win_rate = (
                100 * s["games_won"] / s["games_played"]
                if s["games_played"] > 0
                else 0
            )
            print(
                f"  {name:<20s} {s['seen']:5d} {s['played']:5d} {s['burned']:5d} "
                f"{play_pct:5.1f}% {win_rate:6.1f}%"
            )


def main():
    parser = argparse.ArgumentParser(description="Evaluate OAB RL agent")
    parser.add_argument("--model-path", default="models/oab_agent",
                        help="Path to trained model (without .zip)")
    parser.add_argument("--set-id", type=int, default=0,
                        help="Card set ID")
    parser.add_argument("--num-games", type=int, default=100,
                        help="Number of games to evaluate")
    args = parser.parse_args()

    model = MaskablePPO.load(args.model_path)

    print(f"Evaluating {args.model_path} for {args.num_games} games on set {args.set_id}...")
    results, card_stats = run_evaluation(model, args.set_id, args.num_games)
    print_summary(results, card_stats, args.set_id)


if __name__ == "__main__":
    main()
