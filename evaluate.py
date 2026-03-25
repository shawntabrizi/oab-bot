#!/usr/bin/env python3
"""Evaluate a trained RL agent on Open Auto Battler.

Runs the model for N games in a self-play lobby and reports win/loss
stats plus per-card pick/play/burn rates for balance analysis.

Usage:
    python evaluate.py
    python evaluate.py --model-path models/oab_agent --num-games 500
"""

import argparse
from collections import defaultdict

import numpy as np
from sb3_contrib import MaskablePPO

from env import OABEnv, BoardPool, ACTION_TABLE


def run_evaluation(model, server_url, set_id, num_agents, num_games):
    """Run games with self-play and collect statistics."""
    board_pool = BoardPool(max_size=200)

    # Track stats for agent 0 (all agents share the same policy)
    results = []
    card_stats = defaultdict(lambda: {
        "seen": 0,
        "played": 0,
        "burned": 0,
        "games_played": 0,
        "games_won": 0,
    })

    games_completed = 0

    while games_completed < num_games:
        # Create a lobby of agents
        envs = [
            OABEnv(
                server_url=server_url,
                set_id=set_id,
                agent_id=f"eval_{i}",
                board_pool=board_pool,
            )
            for i in range(num_agents)
        ]

        obs_list = []
        for env in envs:
            obs, _ = env.reset()
            obs_list.append(obs)

        alive = list(range(num_agents))
        game_rewards = [0.0] * num_agents
        game_steps = [0] * num_agents
        cards_played = [set() for _ in range(num_agents)]
        cards_seen = [set() for _ in range(num_agents)]

        # Record initial hands
        for i in alive:
            for card in envs[i]._hand:
                if card is not None:
                    cname = card["name"]
                    card_stats[cname]["seen"] += 1
                    cards_seen[i].add(cname)

        while alive:
            for i in list(alive):
                env = envs[i]
                masks = env.action_masks()
                action, _ = model.predict(
                    obs_list[i], action_masks=masks, deterministic=True
                )
                action = int(action)

                # Track card actions
                action_type, params = ACTION_TABLE[action]
                if action_type == "BurnFromHand":
                    card = env._hand[params["hand_index"]]
                    if card:
                        card_stats[card["name"]]["burned"] += 1
                elif action_type == "PlayFromHand":
                    card = env._hand[params["hand_index"]]
                    if card:
                        card_stats[card["name"]]["played"] += 1
                        cards_played[i].add(card["name"])

                obs, reward, terminated, truncated, info = env.step(action)
                obs_list[i] = obs
                game_rewards[i] += reward
                game_steps[i] += 1

                # After EndTurn, record new hand cards
                if action_type == "EndTurn" and not terminated:
                    for card in env._hand:
                        if card is not None:
                            cname = card["name"]
                            if cname not in cards_seen[i]:
                                card_stats[cname]["seen"] += 1
                                cards_seen[i].add(cname)

                if terminated or truncated:
                    alive.remove(i)
                    game_result = info.get("game_result", "unknown")
                    wins = info.get("wins", 0)
                    is_victory = game_result == "victory"

                    for cname in cards_played[i]:
                        card_stats[cname]["games_played"] += 1
                        if is_victory:
                            card_stats[cname]["games_won"] += 1

                    games_completed += 1
                    results.append({
                        "game": games_completed,
                        "result": game_result,
                        "wins": wins,
                        "reward": game_rewards[i],
                        "steps": game_steps[i],
                    })
                    print(
                        f"  Game {games_completed:3d}: {game_result:7s}  "
                        f"wins={wins:2d}  reward={game_rewards[i]:+.0f}  "
                        f"steps={game_steps[i]}"
                    )

                    if games_completed >= num_games:
                        break

            if games_completed >= num_games:
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
    parser.add_argument("--server-url", default="http://localhost:3000",
                        help="Game server URL")
    parser.add_argument("--set-id", type=int, default=0,
                        help="Card set ID")
    parser.add_argument("--num-agents", type=int, default=10,
                        help="Number of agents in self-play lobby")
    parser.add_argument("--num-games", type=int, default=100,
                        help="Total number of games to evaluate")
    args = parser.parse_args()

    model = MaskablePPO.load(args.model_path)

    print(
        f"Evaluating {args.model_path} — "
        f"{args.num_games} games, {args.num_agents}-agent lobby, set {args.set_id}"
    )
    results, card_stats = run_evaluation(
        model, args.server_url, args.set_id, args.num_agents, args.num_games
    )
    print_summary(results, card_stats, args.set_id)


if __name__ == "__main__":
    main()
