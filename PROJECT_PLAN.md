# OAB Bot — Reinforcement Learning for Open Auto Battler

## Goal

Train an RL agent that learns to play Open Auto Battler well, then use it for:

1. **Balance testing** — Collect card-level statistics (pick rate, win rate, burn rate) from a skilled player to identify overpowered/underpowered cards.
2. **Smart AI opponent** — Drop-in replacement for heuristic bots that plays with learned strategy.
3. **Sensitivity analysis** — Retrain after card stat changes to measure balance impact at high skill levels.

## Architecture

```
open-auto-battler (separate repo)          oab-bot (this repo)
┌──────────────────────────┐              ┌──────────────────────────┐
│  Rust game server        │   HTTP       │  Python RL training      │
│  localhost:3000          │◄────────────►│  Gymnasium + SB3         │
│  POST /reset             │              │                          │
│  POST /submit            │              │  env.py    (gym wrapper) │
│  GET  /state             │              │  train.py  (PPO)         │
│  GET  /cards             │              │  evaluate.py (stats)     │
│  GET  /sets              │              │  models/   (saved .zip)  │
└──────────────────────────┘              └──────────────────────────┘
```

**Prerequisite**: Start the game server from open-auto-battler before training:

```bash
cd /path/to/open-auto-battler
cargo run -p oab-server --release
# Server runs on localhost:3000
```

## Game Server API

### POST /reset

Start a new game. Returns initial `GameStateResponse`.

```json
// Request (optional body)
{ "seed": 12345, "set_id": 0 }
```

### POST /submit

Submit turn actions, triggers battle. Returns `StepResponse`.

```json
// Request
{
  "actions": [
    { "type": "BurnFromHand", "hand_index": 0 },
    { "type": "PlayFromHand", "hand_index": 1, "board_slot": 0 }
  ]
}
```

### GET /state

Returns current `GameStateResponse`.

### GET /cards

Returns all card definitions in the current set.

### GET /sets

Returns all available card sets.

## Game State (Observation)

```json
{
  "round": 1,
  "lives": 3,
  "wins": 0,
  "mana": 3,
  "mana_limit": 3,
  "phase": "shop",
  "bag_count": 45,
  "hand": [
    { "id": 5, "name": "Militia", "attack": 2, "health": 3,
      "play_cost": 2, "burn_value": 1,
      "shop_abilities": [], "battle_abilities": [...] },
    null, null, null, null
  ],
  "board": [null, null, null, null, null],
  "can_afford": [true, false, false, false, false]
}
```

### Key constants

| Constant | Value |
|----------|-------|
| HAND_SIZE | 5 |
| BOARD_SIZE | 5 |
| STARTING_LIVES | 3 |
| STARTING_MANA_LIMIT | 3 |
| MAX_MANA_LIMIT | 10 |
| WINS_TO_VICTORY | 10 |
| STARTING_BAG_SIZE | 50 |

## Action Space

Five action types, submitted as an ordered list per turn:

| Action | Parameters | Description |
|--------|-----------|-------------|
| `BurnFromHand` | `hand_index: 0-4` | Destroy hand card, gain its burn_value as mana |
| `PlayFromHand` | `hand_index: 0-4, board_slot: 0-4` | Place hand card on board (costs play_cost mana) |
| `BurnFromBoard` | `board_slot: 0-4` | Sell board unit, gain its burn_value as mana |
| `SwapBoard` | `slot_a: 0-4, slot_b: 0-4` | Swap two board positions |
| `MoveBoard` | `from_slot: 0-4, to_slot: 0-4` | Reorder board (shifts intermediate units) |

**Multi-action turns**: A turn is a sequence of actions. An empty action list `[]` skips the shop phase and goes straight to battle. The RL agent models this as: each step = one action, with "end turn" (submit empty) as a distinct action.

## Reward Signal

From the server's `StepResponse`:

- `reward: +1` — Won the round
- `reward: -1` — Lost the round
- `reward: 0` — Draw
- `game_over: true` + `game_result: "victory"` — Won the game (reached 10 wins)
- `game_over: true` + `game_result: "defeat"` — Lost the game (0 lives)

## Card Abilities Reference

Cards have battle abilities and shop abilities with rich targeting/trigger systems.

### Battle triggers

`OnStart`, `OnFaint`, `OnAllyFaint`, `OnHurt`, `OnSpawn`, `OnAllySpawn`, `OnEnemySpawn`, `BeforeUnitAttack`, `AfterUnitAttack`, `BeforeAnyAttack`, `AfterAnyAttack`, `AfterAllyAttack`, `AfterEnemyAttack`

### Shop triggers

`OnBuy`, `OnSell`, `OnShopStart`, `AfterLoss`, `AfterWin`, `AfterDraw`

### Effects

`Damage`, `ModifyStats`, `ModifyStatsPermanent`, `SpawnUnit`, `Destroy`, `GainMana`

### Targeting

- `Position` — Absolute or relative board index
- `Adjacent` — Neighbors of a unit
- `Random` — N random from scope
- `Standard` — Top N by stat (highest/lowest attack/health)
- `All` — Every unit in scope

### Scopes

`SelfUnit`, `Allies`, `Enemies`, `All`, `AlliesOther`, `TriggerSource`, `Aggressor`

## Battle Mechanics

- Front units (position 0) clash each round
- Triggers fire in priority order: OnStart > BeforeAnyAttack > BeforeUnitAttack > Clash > AfterUnitAttack > AfterAnyAttack > death checks > OnFaint/OnAllyFaint
- Deterministic RNG via XorShiftRng (battle_seed = round_number)
- Board position matters: slot 0 = front line, abilities use positional targeting

## Implementation Plan

### Phase 1: Gymnasium Environment Wrapper

Build `env.py` — a `gymnasium.Env` subclass that:

- Manages HTTP connection to the game server
- Encodes JSON game state into a fixed-size numpy observation vector
- Provides `valid_action_mask()` for action masking
- Maps discrete action indices to `TurnAction` JSON payloads
- Handles multi-action turns (each gym step = one action within a turn, with "end turn" as a distinct action)

**Observation encoding** (approximate dimensions):
- Per hand slot (5 slots): card_id one-hot or embedding, attack, health, play_cost, burn_value, has_ability flags
- Per board slot (5 slots): same as hand + perm_attack, perm_health deltas
- Scalars: mana, mana_limit, round, lives, wins, bag_count
- can_afford flags (5)
- Total: ~80-150 floats depending on card encoding approach

**Action encoding** (approximate count):
- BurnFromHand: 5 actions (one per hand slot)
- PlayFromHand: 25 actions (5 hand slots x 5 board slots)
- BurnFromBoard: 5 actions (one per board slot)
- SwapBoard: 10 actions (unique pairs)
- MoveBoard: 20 actions (ordered pairs)
- EndTurn: 1 action
- Total: ~66 discrete actions

### Phase 2: Training Script

- Install: `gymnasium`, `sb3-contrib`, `stable-baselines3`, `torch`, `numpy`
- Use `MaskablePPO` from sb3-contrib (handles invalid action masking natively)
- Train for ~500K-1M timesteps
- Save model as `.zip`

### Phase 3: Evaluation & Balance Analysis

- Load trained model, run N games per card set
- Track per-card statistics: pick rate, win rate when played, burn rate
- Compare against heuristic agent baselines
- Sensitivity analysis: tweak card stats, retrain/fine-tune, re-measure

## Existing Heuristic Baselines (in open-auto-battler)

Four agents in `open-auto-battler/scripts/agents/` for comparison:

| Agent | Strategy |
|-------|----------|
| **Greedy** | Maximize (attack + health + ability_bonus) / cost ratio |
| **Aggro** | Prioritize attack damage, sell weak units for upgrades |
| **Tank** | Maximize health, put beefiest unit at front |
| **Economy** | Burn early (rounds 1-3), invest in expensive cards late |

Run with: `python3 scripts/agents/greedy.py [port] [num_games] [set_id]`

## Dependencies

```
gymnasium>=1.0
stable-baselines3>=2.0
sb3-contrib>=2.0
torch>=2.0
numpy>=1.24
```
