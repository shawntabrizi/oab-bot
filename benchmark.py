#!/usr/bin/env python3 -u
"""Head-to-head benchmark: RL model vs scripted bots and bots vs each other.

Each match pairs two agents. Both get independent game sessions (different
seeds/hands) but face each other's boards every round. The match ends when
either player's game finishes (10 wins or 0 lives). The player with more
game-wins at that point wins the match.

Usage:
    python benchmark.py
    python benchmark.py --model-path models/oab_phase_cap20_poolv3_500k_20260327_0310
    python benchmark.py --num-games 200 --set-id 1
"""

import argparse
import importlib.util
import json
import math
import os
import time
from itertools import combinations

import numpy as np

import oab_py
from oab_shared import PhaseController


# ── Helpers ──

def wilson_ci(successes, total, z=1.96):
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return (max(0.0, center - spread) * 100, min(1.0, center + spread) * 100)


# ── Agent abstractions ──

class RLAgent:
    """Wraps a MaskablePPO model for head-to-head play."""

    def __init__(self, name, model, config):
        self.name = name
        self.model = model
        self.config = config
        self._pc = PhaseController(
            enabled=config.phase_decomposition,
            shop_action_limit=config.shop_action_limit,
            position_action_limit=config.position_action_limit,
        )

    def do_turn(self, session):
        """Run the shopping phase action-by-action. Stops at EndTurn."""
        self._pc.reset()
        for _ in range(200):  # safety limit
            obs = np.array(session.get_observation(), dtype=np.float32)
            obs = self._pc.augment_observation(obs)
            masks = np.array(session.get_action_mask(), dtype=bool)
            masks = self._pc.apply_mask(masks)

            action, _ = self.model.predict(obs, action_masks=masks, deterministic=True)
            action = int(action)

            if self._pc.is_phase_transition_action(action):
                self._pc.advance_to_position()
                continue
            if action == 0:  # EndTurn
                return
            session.apply_action(action)
            self._pc.record_action(action)

    def do_battle(self, session, opponent_board_json):
        """Battle the opponent's board. Returns (game_over, info_dict)."""
        _reward, game_over, info_json = session.commit_turn_and_battle(opponent_board_json)
        return game_over, json.loads(info_json)


class ScriptedAgent:
    """Wraps a scripted bot's decide() function for head-to-head play."""

    def __init__(self, name, decide_fn):
        self.name = name
        self.decide_fn = decide_fn

    def do_turn(self, session):
        """Run the shopping phase via the legacy batch API."""
        state = json.loads(session.get_state())
        actions = self.decide_fn(state)
        try:
            session.shop(json.dumps(actions))
        except Exception:
            # Invalid actions — submit empty turn
            session.shop("[]")

    def do_battle(self, session, opponent_board_json):
        """Battle the opponent's board. Returns (game_over, info_dict)."""
        result = json.loads(session.battle(opponent_board_json))
        return result["game_over"], result


# ── Match runner ──

def play_match(agent_a, agent_b, seed_a, seed_b, set_id, max_rounds):
    """Play one head-to-head match. Returns (winner, a_wins, b_wins, a_lives, b_lives)."""
    sess_a = oab_py.GameSession(seed_a, set_id)
    sess_b = oab_py.GameSession(seed_b, set_id)
    sess_a.set_max_rounds(max_rounds)
    sess_b.set_max_rounds(max_rounds)

    for _ in range(max_rounds + 5):  # safety
        # Both agents shop
        agent_a.do_turn(sess_a)
        agent_b.do_turn(sess_b)

        # Swap boards for battle
        board_a = sess_a.get_board_as_opponent()
        board_b = sess_b.get_board_as_opponent()

        a_over, _ = agent_a.do_battle(sess_a, board_b)
        b_over, _ = agent_b.do_battle(sess_b, board_a)

        if a_over or b_over:
            break

    _, a_lives, a_wins = sess_a.get_info()
    _, b_lives, b_wins = sess_b.get_info()

    # Determine winner: victory > alive > defeat; tiebreak by wins then lives
    def score(wins, lives):
        if wins >= 10:
            return (2, wins, lives)
        if lives > 0:
            return (1, wins, lives)
        return (0, wins, lives)

    sa, sb = score(a_wins, a_lives), score(b_wins, b_lives)
    if sa > sb:
        winner = "a"
    elif sb > sa:
        winner = "b"
    else:
        winner = "draw"

    return winner, a_wins, b_wins, a_lives, b_lives


def run_matchup(agent_a, agent_b, num_games, set_id, max_rounds, rng):
    """Run num_games matches between two agents. Returns stats dict."""
    a_match_wins = 0
    b_match_wins = 0
    draws = 0
    a_game_wins = []
    b_game_wins = []

    for _ in range(num_games):
        seed_a = int(rng.integers(0, 2**32))
        seed_b = int(rng.integers(0, 2**32))
        winner, aw, bw, _al, _bl = play_match(
            agent_a, agent_b, seed_a, seed_b, set_id, max_rounds
        )
        if winner == "a":
            a_match_wins += 1
        elif winner == "b":
            b_match_wins += 1
        else:
            draws += 1
        a_game_wins.append(aw)
        b_game_wins.append(bw)

    return {
        "a_name": agent_a.name,
        "b_name": agent_b.name,
        "a_match_wins": a_match_wins,
        "b_match_wins": b_match_wins,
        "draws": draws,
        "a_avg_wins": float(np.mean(a_game_wins)),
        "b_avg_wins": float(np.mean(b_game_wins)),
    }


# ── Load scripted bots ──

SCRIPT_DIR = os.path.join(os.path.dirname(__file__), "scripts", "agents")

def load_decide_fn(script_name):
    path = os.path.join(SCRIPT_DIR, script_name)
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.decide


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Head-to-head benchmark")
    parser.add_argument("--model-path", default="models/oab_agent",
                        help="Path to trained model (without .zip)")
    parser.add_argument("--set-id", type=int, default=None,
                        help="Card set ID (default: from model config)")
    parser.add_argument("--num-games", type=int, default=100,
                        help="Games per matchup")
    parser.add_argument("--max-rounds", type=int, default=None,
                        help="Round cap (default: from model config)")
    args = parser.parse_args()

    from sb3_contrib import MaskablePPO
    from config import load_saved_model_config

    print(f"Loading model from {args.model_path}...")
    model = MaskablePPO.load(args.model_path)
    config = load_saved_model_config(args.model_path)
    if args.set_id is not None:
        config.set_id = args.set_id
    if args.max_rounds is not None:
        config.max_rounds = args.max_rounds

    set_id = config.set_id
    max_rounds = config.max_rounds

    # Build agent roster
    agents = [RLAgent("RL Model", model, config)]
    for name, script in [("Greedy", "greedy.py"), ("Aggro", "aggro.py"),
                         ("Tank", "tank.py"), ("Economy", "economy.py")]:
        agents.append(ScriptedAgent(name, load_decide_fn(script)))

    num_games = args.num_games
    rng = np.random.default_rng(42)

    print(f"\nHead-to-head: {num_games} games per matchup, set {set_id}, round cap {max_rounds}")
    print(f"Agents: {', '.join(a.name for a in agents)}")
    print(f"Matchups: {len(agents) * (len(agents) - 1) // 2}")
    print(f"{'=' * 72}")

    # Round-robin
    matchup_results = []
    # Track per-agent totals for leaderboard
    agent_stats = {a.name: {"match_wins": 0, "match_losses": 0, "draws": 0,
                             "game_wins": [], "matches": 0} for a in agents}

    for agent_a, agent_b in combinations(agents, 2):
        label = f"{agent_a.name} vs {agent_b.name}"
        print(f"\n  {label}...", end="", flush=True)
        t0 = time.time()
        result = run_matchup(agent_a, agent_b, num_games, set_id, max_rounds, rng)
        elapsed = time.time() - t0
        matchup_results.append(result)

        print(
            f"  {result['a_match_wins']}-{result['b_match_wins']}-{result['draws']}"
            f"  (avg {result['a_avg_wins']:.1f} vs {result['b_avg_wins']:.1f})"
            f"  [{elapsed:.1f}s]"
        )

        # Accumulate
        sa = agent_stats[agent_a.name]
        sb = agent_stats[agent_b.name]
        sa["match_wins"] += result["a_match_wins"]
        sa["match_losses"] += result["b_match_wins"]
        sa["draws"] += result["draws"]
        sa["matches"] += num_games
        sb["match_wins"] += result["b_match_wins"]
        sb["match_losses"] += result["a_match_wins"]
        sb["draws"] += result["draws"]
        sb["matches"] += num_games

    # ── Matchup table ──
    print(f"\n{'=' * 72}")
    print(f"  MATCHUP RESULTS (Set {set_id}, {num_games} games each, round cap {max_rounds})")
    print(f"{'=' * 72}")
    print(f"  {'Matchup':<28s} {'W-L-D':>10s} {'A Avg':>7s} {'B Avg':>7s} {'A Rate':>8s}")
    print(f"  {'-'*28} {'-'*10} {'-'*7} {'-'*7} {'-'*8}")
    for r in matchup_results:
        wld = f"{r['a_match_wins']}-{r['b_match_wins']}-{r['draws']}"
        matchup = f"{r['a_name']} vs {r['b_name']}"
        rate = 100 * r["a_match_wins"] / num_games
        print(
            f"  {matchup:<28s} {wld:>10s} {r['a_avg_wins']:>6.1f}  {r['b_avg_wins']:>6.1f}  {rate:>6.1f}%"
        )

    # ── Leaderboard ──
    leaderboard = []
    for name, st in agent_stats.items():
        total = st["matches"]
        win_rate = 100 * st["match_wins"] / total if total else 0
        lo, hi = wilson_ci(st["match_wins"], total)
        leaderboard.append({
            "name": name,
            "match_wins": st["match_wins"],
            "match_losses": st["match_losses"],
            "draws": st["draws"],
            "total": total,
            "win_rate": win_rate,
            "ci_lo": lo,
            "ci_hi": hi,
        })
    leaderboard.sort(key=lambda x: x["win_rate"], reverse=True)

    print(f"\n{'=' * 72}")
    print(f"  LEADERBOARD")
    print(f"{'=' * 72}")
    print(f"  {'#':>2s}  {'Agent':<16s} {'W':>5s} {'L':>5s} {'D':>5s} {'Rate':>7s} {'95% CI':>16s}")
    print(f"  {'--':>2s}  {'-'*16} {'-'*5} {'-'*5} {'-'*5} {'-'*7} {'-'*16}")
    for rank, entry in enumerate(leaderboard, 1):
        ci = f"[{entry['ci_lo']:.1f}%, {entry['ci_hi']:.1f}%]"
        print(
            f"  {rank:>2d}  {entry['name']:<16s} "
            f"{entry['match_wins']:>5d} {entry['match_losses']:>5d} {entry['draws']:>5d} "
            f"{entry['win_rate']:>5.1f}%  {ci:>16s}"
        )
    print()


if __name__ == "__main__":
    main()
