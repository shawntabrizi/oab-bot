//! Python bindings for the OAB game engine via PyO3.
//!
//! Exposes `GameSession` directly to Python. The hot-path training API
//! returns observation vectors and action masks directly (no JSON).
//! Legacy JSON methods are kept for play.py and debugging.

mod actions;
mod obs;

use std::collections::BTreeMap;

use oab_battle::battle::{
    player_permanent_stat_deltas_from_events, player_shop_mana_delta_from_events, resolve_battle,
    BattleResult, CombatEvent, CombatUnit, UnitId,
};
use oab_battle::commit::{apply_shop_start_triggers, apply_shop_start_triggers_with_result};
use oab_battle::rng::XorShiftRng;
use oab_battle::types::*;
use oab_game::sealed::{create_starting_bag, default_config};
use oab_game::view::{CardView, GameView};
use oab_game::GamePhase;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use serde_json::json;

const DEFAULT_MAX_ACTIONS_PER_TURN: usize = 15;

/// A local game session exposed to Python.
#[pyclass]
struct GameSession {
    state: oab_game::GameState,
    card_set: oab_battle::state::CardSet,
    // Shadow state for local action tracking (avoids JSON round-trips)
    shadow_hand: Vec<Option<CardId>>,
    shadow_board: Vec<Option<BoardUnit>>,
    shadow_mana: i32,
    pending_actions: Vec<TurnAction>,
    max_actions_per_turn: usize,
}

impl GameSession {
    /// Sync shadow state from the canonical game state.
    fn sync_shadow(&mut self) {
        self.shadow_hand = self.state.shop.hand.iter().map(|cid| Some(*cid)).collect();
        while self.shadow_hand.len() < obs::HAND_SIZE {
            self.shadow_hand.push(None);
        }
        self.shadow_board = self.state.shop.board.clone();
        while self.shadow_board.len() < obs::BOARD_SIZE {
            self.shadow_board.push(None);
        }
        self.shadow_mana = self.state.shop.shop_mana;
        self.pending_actions.clear();
    }
}

#[pymethods]
impl GameSession {
    // ── Construction ──

    /// Create a new game session with the given seed and card set ID.
    #[new]
    fn new(seed: u64, set_id: u32) -> PyResult<Self> {
        let card_pool = oab_assets::cards::build_pool();
        let all_sets = oab_assets::sets::get_all();
        let card_set = all_sets
            .into_iter()
            .nth(set_id as usize)
            .ok_or_else(|| PyRuntimeError::new_err(format!("Card set {} not found", set_id)))?;

        let config = default_config();
        let bag_size = config.bag_size as usize;

        let mut state = oab_game::GameState::new(seed, config);
        state.shop.card_pool = card_pool;
        state.shop.set_id = set_id;
        state.bag = create_starting_bag(&card_set, seed, bag_size);
        state.next_card_id = 1000;
        state.lives = state.config.starting_lives;
        let hand_size = state.config.hand_size as usize;
        state.draw_hand(hand_size);
        apply_shop_start_triggers(&mut state.shop);

        let mut session = Self {
            state,
            card_set,
            shadow_hand: Vec::new(),
            shadow_board: Vec::new(),
            shadow_mana: 0,
            pending_actions: Vec::new(),
            max_actions_per_turn: DEFAULT_MAX_ACTIONS_PER_TURN,
        };
        session.sync_shadow();
        Ok(session)
    }

    // ── Training API (no JSON on hot path) ──

    /// Get observation as a flat list of floats.
    fn get_observation(&self) -> Vec<f32> {
        obs::encode_observation(
            &self.shadow_hand,
            &self.shadow_board,
            self.shadow_mana,
            &self.state,
        )
    }

    /// Get action mask as a list of bools (66 elements).
    fn get_action_mask(&self) -> Vec<bool> {
        actions::compute_action_mask(
            &self.shadow_hand,
            &self.shadow_board,
            self.shadow_mana,
            self.pending_actions.len(),
            self.max_actions_per_turn,
            &self.state.shop.card_pool,
        )
    }

    /// Apply an action locally (updates shadow state, adds to pending).
    /// Returns true if the board/hand actually changed (false = no-op swap/move).
    fn apply_action(&mut self, action_index: u32) -> PyResult<bool> {
        let turn_action = actions::action_to_turn_action(action_index)
            .ok_or_else(|| {
                PyRuntimeError::new_err("Action 0 (EndTurn) should not be passed to apply_action")
            })?;

        let pool = &self.state.shop.card_pool;
        let board_before: Vec<Option<CardId>> = self
            .shadow_board
            .iter()
            .map(|s| s.as_ref().map(|u| u.card_id))
            .collect();

        match &turn_action {
            TurnAction::BurnFromHand { hand_index } => {
                let i = *hand_index as usize;
                if let Some(Some(card_id)) = self.shadow_hand.get(i) {
                    if let Some(card) = pool.get(card_id) {
                        self.shadow_mana = (self.shadow_mana + card.economy.burn_value)
                            .min(self.state.shop.mana_limit);
                    }
                    self.shadow_hand[i] = None;
                }
            }
            TurnAction::PlayFromHand {
                hand_index,
                board_slot,
            } => {
                let hi = *hand_index as usize;
                let bs = *board_slot as usize;
                if let Some(Some(card_id)) = self.shadow_hand.get(hi) {
                    if let Some(card) = pool.get(card_id) {
                        self.shadow_mana -= card.economy.play_cost;
                    }
                    self.shadow_board[bs] = Some(BoardUnit::new(*card_id));
                    self.shadow_hand[hi] = None;
                }
            }
            TurnAction::BurnFromBoard { board_slot } => {
                let bs = *board_slot as usize;
                if let Some(Some(unit)) = self.shadow_board.get(bs) {
                    if let Some(card) = pool.get(&unit.card_id) {
                        self.shadow_mana = (self.shadow_mana + card.economy.burn_value)
                            .min(self.state.shop.mana_limit);
                    }
                }
                if bs < self.shadow_board.len() {
                    self.shadow_board[bs] = None;
                }
            }
            TurnAction::SwapBoard { slot_a, slot_b } => {
                let a = *slot_a as usize;
                let b = *slot_b as usize;
                self.shadow_board.swap(a, b);
            }
            TurnAction::MoveBoard {
                from_slot,
                to_slot,
            } => {
                let f = *from_slot as usize;
                let t = *to_slot as usize;
                let unit = self.shadow_board[f].take();
                if f < t {
                    for k in f..t {
                        self.shadow_board[k] = self.shadow_board[k + 1].take();
                    }
                } else {
                    for k in (t..f).rev() {
                        let next = self.shadow_board[k].take();
                        self.shadow_board[k + 1] = next;
                    }
                }
                self.shadow_board[t] = unit;
            }
        }

        self.pending_actions.push(turn_action);

        // Check if board actually changed (for no-op detection)
        let board_after: Vec<Option<CardId>> = self
            .shadow_board
            .iter()
            .map(|s| s.as_ref().map(|u| u.card_id))
            .collect();
        Ok(board_before != board_after
            || !matches!(
                self.pending_actions.last(),
                Some(TurnAction::SwapBoard { .. } | TurnAction::MoveBoard { .. })
            ))
    }

    /// Commit pending actions and run battle. Returns (reward, terminated, info_json).
    fn commit_turn_and_battle(
        &mut self,
        opponent_json: &str,
    ) -> PyResult<(f32, bool, String)> {
        // Submit shop actions
        let action = CommitTurnAction {
            actions: std::mem::take(&mut self.pending_actions),
        };

        if let Err(e) = oab_battle::commit::verify_and_apply_turn(&mut self.state.shop, &action) {
            // Fallback: submit empty actions
            let empty = CommitTurnAction {
                actions: Vec::new(),
            };
            oab_battle::commit::verify_and_apply_turn(&mut self.state.shop, &empty)
                .map_err(|e2| PyRuntimeError::new_err(format!("Shop failed: {:?}, {:?}", e, e2)))?;
        }

        self.state.shop.shop_mana = 0;
        self.state.phase = GamePhase::Battle;

        // Run battle
        let opponent: Vec<OpponentUnit> = serde_json::from_str(opponent_json)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid opponent JSON: {}", e)))?;

        let outcome = self.run_battle(&opponent);

        let game_over =
            self.state.wins >= self.state.config.wins_to_victory || self.state.lives <= 0;

        let reward = match &outcome.result {
            BattleResult::Victory => 1.0,
            BattleResult::Defeat => -1.0,
            BattleResult::Draw => 0.0,
        };

        let battle_result_str = match &outcome.result {
            BattleResult::Victory => "Victory",
            BattleResult::Defeat => "Defeat",
            BattleResult::Draw => "Draw",
        };

        let game_result = if game_over {
            self.state.phase = GamePhase::Completed;
            Some(if self.state.wins >= self.state.config.wins_to_victory {
                "victory"
            } else {
                "defeat"
            })
        } else {
            self.state.shop.round += 1;
            self.state.shop.mana_limit = self
                .state
                .config
                .mana_limit_for_round(self.state.shop.round);
            if self.state.config.full_mana_each_round {
                self.state.shop.shop_mana = self.state.shop.mana_limit;
            }
            self.state.phase = GamePhase::Shop;
            let hand_size = self.state.config.hand_size as usize;
            self.state.draw_hand(hand_size);
            apply_shop_start_triggers_with_result(
                &mut self.state.shop,
                Some(outcome.result.clone()),
            );
            None
        };

        // Re-sync shadow from canonical state
        self.sync_shadow();

        let info = json!({
            "round": self.state.shop.round,
            "lives": self.state.lives,
            "wins": self.state.wins,
            "battle_result": battle_result_str,
            "game_result": game_result,
        });

        Ok((reward, game_over, info.to_string()))
    }

    /// Get board as opponent JSON for pool storage (includes perm stats).
    fn get_board_as_opponent(&self) -> String {
        let mut entries = Vec::new();
        for (slot, unit) in self.shadow_board.iter().enumerate() {
            if let Some(bu) = unit {
                let mut entry = json!({"card_id": bu.card_id.0, "slot": slot});
                if bu.perm_attack != 0 {
                    entry["perm_attack"] = json!(bu.perm_attack);
                }
                if bu.perm_health != 0 {
                    entry["perm_health"] = json!(bu.perm_health);
                }
                entries.push(entry);
            }
        }
        serde_json::to_string(&entries).unwrap_or_else(|_| "[]".into())
    }

    /// Get (round, lives, wins) tuple for info dict.
    fn get_info(&self) -> (i32, i32, i32) {
        (self.state.shop.round, self.state.lives, self.state.wins)
    }

    /// Get hand card names (for evaluate.py card tracking).
    fn get_hand_names(&self) -> Vec<Option<String>> {
        self.shadow_hand
            .iter()
            .map(|slot| {
                slot.and_then(|cid| {
                    self.state
                        .shop
                        .card_pool
                        .get(&cid)
                        .map(|c| c.name.clone())
                })
            })
            .collect()
    }

    /// Get board card names (for evaluate.py card tracking).
    fn get_board_names(&self) -> Vec<Option<String>> {
        self.shadow_board
            .iter()
            .map(|slot| {
                slot.as_ref().and_then(|bu| {
                    self.state
                        .shop
                        .card_pool
                        .get(&bu.card_id)
                        .map(|c| c.name.clone())
                })
            })
            .collect()
    }

    /// Number of pending actions this turn.
    fn pending_action_count(&self) -> usize {
        self.pending_actions.len()
    }

    /// Observation vector dimension.
    #[staticmethod]
    fn obs_dim() -> usize {
        obs::OBS_DIM
    }

    /// Number of discrete actions.
    #[staticmethod]
    fn num_actions() -> usize {
        actions::NUM_ACTIONS
    }

    // ── Legacy JSON API (for play.py and debugging) ──

    /// Reset the game with a new seed. Returns state JSON.
    fn reset(&mut self, seed: u64, set_id: Option<u32>) -> PyResult<String> {
        if let Some(set_id) = set_id {
            *self = GameSession::new(seed, set_id)?;
        } else {
            let card_pool = std::mem::take(&mut self.state.shop.card_pool);
            let sid = self.state.shop.set_id;
            let config = self.state.config.clone();
            let bag_size = config.bag_size as usize;
            let hand_size = config.hand_size as usize;

            self.state = oab_game::GameState::new(seed, config);
            self.state.shop.card_pool = card_pool;
            self.state.shop.set_id = sid;
            self.state.bag = create_starting_bag(&self.card_set, seed, bag_size);
            self.state.next_card_id = 1000;
            self.state.lives = self.state.config.starting_lives;
            self.state.draw_hand(hand_size);
            apply_shop_start_triggers(&mut self.state.shop);
            self.sync_shadow();
        }
        Ok(self.state_json())
    }

    /// Get current game state as JSON.
    fn get_state(&self) -> String {
        self.state_json()
    }

    /// Get all cards in the card pool as JSON.
    fn get_cards(&self) -> String {
        let cards: Vec<CardView> = self
            .state
            .shop
            .card_pool
            .values()
            .map(CardView::from)
            .collect();
        serde_json::to_string(&cards).unwrap_or_else(|_| "[]".into())
    }

    /// Apply shop actions via JSON. Returns state JSON.
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
        oab_battle::commit::verify_and_apply_turn(&mut self.state.shop, &action)
            .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))?;

        self.state.shop.shop_mana = 0;
        self.state.phase = GamePhase::Battle;
        self.sync_shadow();

        Ok(self.state_json())
    }

    /// Run battle via JSON. Returns step result JSON.
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

        let completed_round = self.state.shop.round;
        let outcome = self.run_battle(&opponent);

        let game_over =
            self.state.wins >= self.state.config.wins_to_victory || self.state.lives <= 0;
        let game_result = if game_over {
            self.state.phase = GamePhase::Completed;
            Some(if self.state.wins >= self.state.config.wins_to_victory {
                "victory"
            } else {
                "defeat"
            })
        } else {
            self.state.shop.round += 1;
            self.state.shop.mana_limit = self
                .state
                .config
                .mana_limit_for_round(self.state.shop.round);
            if self.state.config.full_mana_each_round {
                self.state.shop.shop_mana = self.state.shop.mana_limit;
            }
            self.state.phase = GamePhase::Shop;
            let hand_size = self.state.config.hand_size as usize;
            self.state.draw_hand(hand_size);
            apply_shop_start_triggers_with_result(
                &mut self.state.shop,
                Some(outcome.result.clone()),
            );
            None
        };

        self.sync_shadow();

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

// ── Private helpers ──

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
        let hand_used = vec![false; self.state.shop.hand.len()];
        let view = GameView::from_state(
            &self.state,
            self.state.shop.shop_mana,
            &hand_used,
            false,
        );

        let mut bag_counts: BTreeMap<CardId, u32> = BTreeMap::new();
        for card_id in self.state.bag.iter() {
            *bag_counts.entry(*card_id).or_insert(0) += 1;
        }
        let bag: Vec<serde_json::Value> = bag_counts
            .into_iter()
            .filter_map(|(card_id, count)| {
                let card = self.state.shop.card_pool.get(&card_id)?;
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

        let board_with_perms: Vec<serde_json::Value> = self
            .state
            .shop
            .board
            .iter()
            .zip(view.board.iter())
            .map(|(raw, view_entry)| match (raw, view_entry) {
                (Some(unit), Some(view_unit)) => {
                    let mut val = serde_json::to_value(view_unit).unwrap_or(json!(null));
                    if let Some(obj) = val.as_object_mut() {
                        obj.insert("perm_attack".into(), json!(unit.perm_attack));
                        obj.insert("perm_health".into(), json!(unit.perm_health));
                    }
                    val
                }
                _ => json!(null),
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
            "board": board_with_perms,
            "can_afford": view.can_afford,
            "bag": bag,
        });

        state.to_string()
    }

    fn run_battle(&mut self, opponent: &[OpponentUnit]) -> BattleOutcome {
        let board_size = self.state.config.board_size as usize;
        let mut player_slots = Vec::new();
        let player_units: Vec<CombatUnit> = self
            .state
            .shop
            .board
            .iter()
            .enumerate()
            .filter_map(|(slot, unit)| {
                let u = unit.as_ref()?;
                player_slots.push(slot);
                let card = self.state.shop.card_pool.get(&u.card_id)?;
                let mut cu = CombatUnit::from_card(card.clone());
                cu.attack_buff = u.perm_attack;
                cu.health_buff = u.perm_health;
                cu.health = cu.health.saturating_add(u.perm_health).max(0);
                Some(cu)
            })
            .collect();

        let enemy_units = self.build_opponent(opponent);

        let battle_seed = self.state.shop.round as u64;
        let mut rng = XorShiftRng::seed_from_u64(battle_seed);
        let events = resolve_battle(
            player_units,
            enemy_units,
            &mut rng,
            &self.state.shop.card_pool,
            board_size,
        );

        self.state.shop.shop_mana = player_shop_mana_delta_from_events(&events).max(0);
        let permanent_deltas = player_permanent_stat_deltas_from_events(&events);
        apply_permanent_deltas(&mut self.state.shop, &player_slots, &permanent_deltas);

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
        let board_size = self.state.config.board_size as usize;
        let mut board: Vec<Option<CombatUnit>> = vec![None; board_size];
        for u in units {
            let slot = u.slot as usize;
            if slot >= board.len() {
                continue;
            }
            if let Some(card) = self.state.shop.card_pool.get(&CardId(u.card_id)) {
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

fn apply_permanent_deltas(
    state: &mut oab_battle::state::ShopState,
    player_slots: &[usize],
    deltas: &BTreeMap<UnitId, (i32, i32)>,
) {
    for (unit_id, (attack_delta, health_delta)) in deltas {
        let unit_index = unit_id.raw() as usize;
        if unit_index == 0 || unit_index > player_slots.len() {
            continue;
        }

        let slot = player_slots[unit_index - 1];

        let death_check = if let Some(board_unit) =
            state.board.get_mut(slot).and_then(|s| s.as_mut())
        {
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

#[pymodule]
fn oab_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GameSession>()?;
    Ok(())
}
