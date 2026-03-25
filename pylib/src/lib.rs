//! Python bindings for the OAB game engine via PyO3.
//!
//! Exposes `GameSession` directly to Python, bypassing the HTTP server.
//! All state is returned as JSON strings for simplicity.

use std::collections::BTreeMap;

use oab_core::battle::{
    player_permanent_stat_deltas_from_events, player_shop_mana_delta_from_events, resolve_battle,
    BattleResult, CombatEvent, CombatUnit, UnitId,
};
use oab_core::commit::{apply_shop_start_triggers, apply_shop_start_triggers_with_result};
use oab_core::rng::XorShiftRng;
use oab_core::state::*;
use oab_core::types::*;
use oab_core::units::create_starting_bag;
use oab_core::view::GameView;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use serde_json::json;

/// A local game session exposed to Python.
#[pyclass]
struct GameSession {
    state: GameState,
    card_set: CardSet,
}

#[pymethods]
impl GameSession {
    /// Create a new game session with the given seed and card set ID.
    #[new]
    fn new(seed: u64, set_id: u32) -> PyResult<Self> {
        let card_pool = oab_core::cards::build_card_pool();
        let all_sets = oab_core::cards::get_all_sets();
        let card_set = all_sets
            .into_iter()
            .nth(set_id as usize)
            .ok_or_else(|| PyRuntimeError::new_err(format!("Card set {} not found", set_id)))?;

        let mut state = GameState::new(seed);
        state.card_pool = card_pool;
        state.set_id = set_id;
        state.local_state.bag = create_starting_bag(&card_set, seed);
        state.local_state.next_card_id = 1000;
        state.draw_hand();
        apply_shop_start_triggers(&mut state);

        Ok(Self { state, card_set })
    }

    /// Reset the game with a new seed. Returns state JSON.
    fn reset(&mut self, seed: u64, set_id: Option<u32>) -> PyResult<String> {
        if let Some(set_id) = set_id {
            *self = GameSession::new(seed, set_id)?;
        } else {
            let card_pool = std::mem::take(&mut self.state.card_pool);
            let sid = self.state.set_id;
            self.state = GameState::new(seed);
            self.state.card_pool = card_pool;
            self.state.set_id = sid;
            self.state.local_state.bag = create_starting_bag(&self.card_set, seed);
            self.state.local_state.next_card_id = 1000;
            self.state.draw_hand();
            apply_shop_start_triggers(&mut self.state);
        }
        Ok(self.state_json())
    }

    /// Get current game state as JSON.
    fn get_state(&self) -> String {
        self.state_json()
    }

    /// Get all cards in the card pool as JSON.
    fn get_cards(&self) -> String {
        let cards: Vec<oab_core::view::CardView> = self
            .state
            .card_pool
            .values()
            .map(oab_core::view::CardView::from)
            .collect();
        serde_json::to_string(&cards).unwrap_or_else(|_| "[]".into())
    }

    /// Apply shop actions. Takes actions as JSON string. Returns state JSON.
    fn shop(&mut self, actions_json: &str) -> PyResult<String> {
        if self.state.phase == GamePhase::Completed {
            return Err(PyRuntimeError::new_err("Game is already over."));
        }
        if self.state.phase != GamePhase::Shop {
            return Err(PyRuntimeError::new_err(format!(
                "Wrong phase: {:?}",
                self.state.phase
            )));
        }

        let actions: Vec<TurnAction> = serde_json::from_str(actions_json)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid actions JSON: {}", e)))?;

        let action = CommitTurnAction { actions };
        oab_core::commit::verify_and_apply_turn(&mut self.state, &action)
            .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))?;

        self.state.shop_mana = 0;
        self.state.phase = GamePhase::Battle;

        Ok(self.state_json())
    }

    /// Run battle against opponent. Takes opponent as JSON string. Returns step result JSON.
    fn battle(&mut self, opponent_json: &str) -> PyResult<String> {
        if self.state.phase == GamePhase::Completed {
            return Err(PyRuntimeError::new_err("Game is already over."));
        }
        if self.state.phase != GamePhase::Battle {
            return Err(PyRuntimeError::new_err(format!(
                "Wrong phase: {:?}. Call shop() first.",
                self.state.phase
            )));
        }

        let opponent: Vec<OpponentUnit> = serde_json::from_str(opponent_json)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid opponent JSON: {}", e)))?;

        let completed_round = self.state.round;
        let outcome = self.run_battle(&opponent);

        let game_over = self.state.wins >= WINS_TO_VICTORY || self.state.lives <= 0;
        let game_result = if game_over {
            self.state.phase = GamePhase::Completed;
            Some(if self.state.wins >= WINS_TO_VICTORY {
                "victory"
            } else {
                "defeat"
            })
        } else {
            self.state.round += 1;
            self.state.mana_limit = self.state.calculate_mana_limit();
            self.state.phase = GamePhase::Shop;
            self.state.draw_hand();
            apply_shop_start_triggers_with_result(&mut self.state, Some(outcome.result.clone()));
            None
        };

        let reward = match &outcome.result {
            BattleResult::Victory => 1,
            BattleResult::Defeat => -1,
            BattleResult::Draw => 0,
        };

        let battle_result_str = match &outcome.result {
            BattleResult::Victory => "Victory",
            BattleResult::Defeat => "Defeat",
            BattleResult::Draw => "Draw",
        };

        let result = json!({
            "completed_round": completed_round,
            "battle_result": battle_result_str,
            "game_over": game_over,
            "game_result": game_result,
            "reward": reward,
            "state": serde_json::from_str::<serde_json::Value>(&self.state_json()).unwrap(),
        });

        Ok(result.to_string())
    }
}

// ── Private helpers (not exposed to Python) ──

/// Opponent unit for deserialization.
#[derive(serde::Deserialize)]
struct OpponentUnit {
    card_id: u32,
    slot: u32,
    #[serde(default)]
    perm_attack: i32,
    #[serde(default)]
    perm_health: i32,
}

struct BattleOutcome {
    result: BattleResult,
}

impl GameSession {
    fn state_json(&self) -> String {
        let hand_used = vec![false; self.state.hand.len()];
        let view = GameView::from_state(&self.state, self.state.shop_mana, &hand_used, false);

        // Build bag summary
        let mut bag_counts: BTreeMap<CardId, u32> = BTreeMap::new();
        for card_id in self.state.bag.iter() {
            *bag_counts.entry(*card_id).or_insert(0) += 1;
        }
        let bag: Vec<serde_json::Value> = bag_counts
            .into_iter()
            .filter_map(|(card_id, count)| {
                let card = self.state.card_pool.get(&card_id)?;
                Some(json!({
                    "card_id": card_id.0,
                    "name": card.name,
                    "attack": card.stats.attack,
                    "health": card.stats.health,
                    "play_cost": card.economy.play_cost,
                    "burn_value": card.economy.burn_value,
                    "count": count,
                }))
            })
            .collect();

        let state = json!({
            "round": view.round,
            "lives": view.lives,
            "wins": view.wins,
            "mana": view.mana,
            "mana_limit": view.mana_limit,
            "phase": view.phase,
            "bag_count": view.bag_count,
            "hand": view.hand,
            "board": view.board,
            "can_afford": view.can_afford,
            "bag": bag,
        });

        state.to_string()
    }

    fn run_battle(&mut self, opponent: &[OpponentUnit]) -> BattleOutcome {
        let mut player_slots = Vec::new();
        let player_units: Vec<CombatUnit> = self
            .state
            .board
            .iter()
            .enumerate()
            .filter_map(|(slot, unit)| {
                let u = unit.as_ref()?;
                player_slots.push(slot);
                let card = self.state.card_pool.get(&u.card_id)?;
                let mut cu = CombatUnit::from_card(card.clone());
                cu.attack_buff = u.perm_attack;
                cu.health_buff = u.perm_health;
                cu.health = cu.health.saturating_add(u.perm_health).max(0);
                Some(cu)
            })
            .collect();

        let enemy_units = self.build_opponent(opponent);

        let battle_seed = self.state.round as u64;
        let mut rng = XorShiftRng::seed_from_u64(battle_seed);
        let events = resolve_battle(player_units, enemy_units, &mut rng, &self.state.card_pool);

        self.state.shop_mana = player_shop_mana_delta_from_events(&events).max(0);
        let permanent_deltas = player_permanent_stat_deltas_from_events(&events);
        apply_permanent_deltas(&mut self.state, &player_slots, &permanent_deltas);

        let result = events
            .iter()
            .rev()
            .find_map(|e| {
                if let CombatEvent::BattleEnd { result } = e {
                    Some(result.clone())
                } else {
                    None
                }
            })
            .unwrap_or(BattleResult::Draw);

        match &result {
            BattleResult::Victory => self.state.wins += 1,
            BattleResult::Defeat => self.state.lives -= 1,
            BattleResult::Draw => {}
        }

        BattleOutcome { result }
    }

    fn build_opponent(&self, units: &[OpponentUnit]) -> Vec<CombatUnit> {
        let mut board: Vec<Option<CombatUnit>> = vec![None; BOARD_SIZE];
        for u in units {
            let slot = u.slot as usize;
            if slot >= board.len() {
                continue;
            }
            if let Some(card) = self.state.card_pool.get(&CardId(u.card_id)) {
                let mut cu = CombatUnit::from_card(card.clone());
                cu.attack_buff = u.perm_attack;
                cu.health_buff = u.perm_health;
                cu.health = cu.health.saturating_add(u.perm_health).max(0);
                board[slot] = Some(cu);
            }
        }
        board.into_iter().flatten().collect()
    }
}

/// Apply permanent stat deltas from battle events to the player's board.
fn apply_permanent_deltas(
    state: &mut GameState,
    player_slots: &[usize],
    deltas: &BTreeMap<UnitId, (i32, i32)>,
) {
    for (unit_id, (attack_delta, health_delta)) in deltas {
        let unit_index = unit_id.raw() as usize;
        if unit_index == 0 || unit_index > player_slots.len() {
            continue;
        }

        let slot = player_slots[unit_index - 1];

        let death_check =
            if let Some(board_unit) = state.board.get_mut(slot).and_then(|s| s.as_mut()) {
                board_unit.perm_attack = board_unit.perm_attack.saturating_add(*attack_delta);
                board_unit.perm_health = board_unit.perm_health.saturating_add(*health_delta);
                Some((board_unit.card_id, board_unit.perm_health))
            } else {
                None
            };

        let should_remove = death_check
            .and_then(|(card_id, perm_health)| {
                state
                    .card_pool
                    .get(&card_id)
                    .map(|card| card.stats.health.saturating_add(perm_health) <= 0)
            })
            .unwrap_or(false);

        if should_remove {
            state.board[slot] = None;
        }
    }
}

/// Python module definition.
#[pymodule]
fn oab_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GameSession>()?;
    Ok(())
}
