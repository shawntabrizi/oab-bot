//! Python bindings for the OAB game engine via PyO3.
//!
//! The training API applies each action through the real game engine
//! (apply_single_action) — no shadow simulator. Observations and masks
//! are derived from the authoritative engine state after every action.

mod actions;
mod obs;

use std::collections::BTreeMap;

use oab_battle::battle::{
    player_permanent_stat_deltas_from_events, player_shop_mana_delta_from_events, resolve_battle,
    BattleResult, CombatEvent, CombatUnit, UnitId,
};
use oab_battle::commit::{
    apply_shop_start_triggers, apply_shop_start_triggers_with_result, apply_single_action,
    finalize_turn, ShopTurnContext,
};
use oab_battle::rng::XorShiftRng;
use oab_battle::types::*;
use oab_game::sealed::{create_starting_bag, default_config};
use oab_game::view::{CardView, GameView};
use oab_game::GamePhase;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use serde_json::json;

const DEFAULT_MAX_ACTIONS_PER_TURN: usize = 15;
const DEFAULT_MAX_ROUNDS_PER_GAME: i32 = 20;

/// A local game session exposed to Python.
#[pyclass]
struct GameSession {
    state: oab_game::GameState,
    card_set: oab_battle::state::CardSet,
    obs_constants: obs::ObsConstants,
    /// Engine context for incremental action application.
    turn_ctx: ShopTurnContext,
    /// Number of actions applied this turn.
    action_count: usize,
    max_actions_per_turn: usize,
    max_rounds_per_game: i32,
}

impl GameSession {
    /// Reset the turn context from the current canonical state.
    fn reset_turn(&mut self) {
        self.turn_ctx = ShopTurnContext::new(&self.state.shop);
        self.action_count = 0;
    }

    /// Build the hand view for observation encoding.
    /// Returns a Vec matching original hand size, with used cards as None.
    fn hand_view(&self) -> Vec<Option<CardId>> {
        self.state
            .shop
            .hand
            .iter()
            .enumerate()
            .map(|(i, cid)| {
                if self.turn_ctx.hand_used.get(i).copied().unwrap_or(false) {
                    None
                } else {
                    Some(*cid)
                }
            })
            .collect()
    }
}

#[pymethods]
impl GameSession {
    // ── Construction ──

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

        let obs_constants = obs::ObsConstants::from_card_set(
            &state.shop.card_pool,
            &card_set,
            &state.config,
        );

        let turn_ctx = ShopTurnContext::new(&state.shop);

        Ok(Self {
            state,
            card_set,
            obs_constants,
            turn_ctx,
            action_count: 0,
            max_actions_per_turn: DEFAULT_MAX_ACTIONS_PER_TURN,
            max_rounds_per_game: DEFAULT_MAX_ROUNDS_PER_GAME,
        })
    }

    // ── Training API (engine-authoritative, no shadow) ──

    /// Get observation from the real engine state.
    fn get_observation(&self) -> Vec<f32> {
        let hand = self.hand_view();
        obs::encode_observation(
            &hand,
            &self.state.shop.board,
            self.turn_ctx.current_mana,
            &self.state,
            &self.obs_constants,
        )
    }

    /// Get action mask from the real engine state.
    fn get_action_mask(&self) -> Vec<bool> {
        let hand = self.hand_view();
        actions::compute_action_mask(
            &hand,
            &self.state.shop.board,
            self.turn_ctx.current_mana,
            self.action_count,
            self.max_actions_per_turn,
            &self.state.shop.card_pool,
        )
    }

    /// Apply a single action through the real game engine.
    /// Returns true if the board changed (false = no-op swap/move).
    fn apply_action(&mut self, action_index: u32) -> PyResult<bool> {
        let turn_action = actions::action_to_turn_action(action_index)
            .ok_or_else(|| {
                PyRuntimeError::new_err("Action 0 (EndTurn) should not be passed to apply_action")
            })?;

        let board_before: Vec<Option<CardId>> = self
            .state
            .shop
            .board
            .iter()
            .map(|s| s.as_ref().map(|u| u.card_id))
            .collect();

        // Apply through the REAL engine — includes insert-shift, OnBuy/OnSell triggers
        apply_single_action(&mut self.state.shop, &mut self.turn_ctx, &turn_action)
            .map_err(|e| PyRuntimeError::new_err(format!("Action failed: {:?}", e)))?;

        self.action_count += 1;

        let board_after: Vec<Option<CardId>> = self
            .state
            .shop
            .board
            .iter()
            .map(|s| s.as_ref().map(|u| u.card_id))
            .collect();

        let is_reorder = matches!(
            turn_action,
            TurnAction::SwapBoard { .. } | TurnAction::MoveBoard { .. }
        );

        Ok(!is_reorder || board_before != board_after)
    }

    /// Finalize the shop turn and run battle.
    /// Returns (reward, terminated, info_json).
    fn commit_turn_and_battle(
        &mut self,
        opponent_json: &str,
    ) -> PyResult<(f32, bool, String)> {
        // Finalize: remove used hand cards
        let ctx = std::mem::replace(&mut self.turn_ctx, ShopTurnContext::new(&self.state.shop));
        finalize_turn(&mut self.state.shop, ctx);

        self.state.shop.shop_mana = 0;
        self.state.phase = GamePhase::Battle;

        // Run battle
        let opponent: Vec<OpponentUnit> = serde_json::from_str(opponent_json)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid opponent JSON: {}", e)))?;

        let completed_round = self.state.shop.round;
        let outcome = self.run_battle(&opponent);

        let won_game = self.state.wins >= self.state.config.wins_to_victory;
        let lost_game = self.state.lives <= 0;
        let capped_out =
            !won_game && !lost_game && self.max_rounds_per_game > 0 && completed_round >= self.max_rounds_per_game;
        let game_over = won_game || lost_game || capped_out;

        // Reward scales with win count: winning when you already have 8 wins
        // is worth more than your first win. This incentivizes closing out games
        // fast rather than grinding slowly.
        let wins = self.state.wins as f32; // wins AFTER this battle's result
        let reward = if capped_out {
            -1.0
        } else {
            match &outcome.result {
                BattleResult::Victory => wins,     // 1st win = +1, 10th win = +10
                BattleResult::Defeat => -1.0,
                BattleResult::Draw => -0.1,
            }
        };

        let battle_result_str = match &outcome.result {
            BattleResult::Victory => "Victory",
            BattleResult::Defeat => "Defeat",
            BattleResult::Draw => "Draw",
        };

        let game_result = if game_over {
            if capped_out {
                self.state.lives = 0;
            }
            self.state.phase = GamePhase::Completed;
            Some(if won_game {
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

        // Reset turn context for new round
        self.reset_turn();

        let info = json!({
            "round": self.state.shop.round,
            "lives": self.state.lives,
            "wins": self.state.wins,
            "battle_result": battle_result_str,
            "game_result": game_result,
            "termination_reason": if capped_out { Some("max_rounds") } else { None::<&str> },
        });

        Ok((reward, game_over, info.to_string()))
    }

    /// Get board as opponent JSON for pool storage (includes perm stats).
    fn get_board_as_opponent(&self) -> String {
        let mut entries = Vec::new();
        for (slot, unit) in self.state.shop.board.iter().enumerate() {
            if let Some(bu) = unit {
                let mut entry = json!({"card_id": bu.card_id.0, "slot": slot});
                if let Some(card) = self.state.shop.card_pool.get(&bu.card_id) {
                    entry["attack"] = json!(card.stats.attack.saturating_add(bu.perm_attack));
                    entry["health"] = json!(card.stats.health.saturating_add(bu.perm_health));
                }
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

    /// Get (round, lives, wins) tuple.
    fn get_info(&self) -> (i32, i32, i32) {
        (self.state.shop.round, self.state.lives, self.state.wins)
    }

    /// Get hand card names (for evaluate.py card tracking).
    fn get_hand_names(&self) -> Vec<Option<String>> {
        self.hand_view()
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
        self.state
            .shop
            .board
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

    fn pending_action_count(&self) -> usize {
        self.action_count
    }

    fn set_max_rounds(&mut self, max_rounds: i32) {
        self.max_rounds_per_game = max_rounds;
    }

    /// Observation vector dimension (depends on the card set).
    fn obs_dim(&self) -> usize {
        self.obs_constants.obs_dim
    }

    /// Number of discrete actions.
    #[staticmethod]
    fn num_actions() -> usize {
        actions::NUM_ACTIONS
    }

    /// Sync internal state from a server JSON response.
    /// Used by play.py to keep the local session in sync with the server
    /// so that get_observation/get_action_mask produce correct results.
    fn sync_from_state_json(&mut self, state_json: &str) -> PyResult<()> {
        let state: serde_json::Value = serde_json::from_str(state_json)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid state JSON: {}", e)))?;

        // Rebuild hand from JSON hand array (CardIds looked up by name/id)
        self.state.shop.hand.clear();
        if let Some(hand) = state["hand"].as_array() {
            for entry in hand {
                if entry.is_null() {
                    continue;
                }
                // Find the CardId by matching the card name or id
                if let Some(card_id) = self.find_card_id_from_json(entry) {
                    self.state.shop.hand.push(card_id);
                }
            }
        }

        // Rebuild board from JSON board array
        let board_size = self.state.config.board_size as usize;
        self.state.shop.board = vec![None; board_size];
        if let Some(board) = state["board"].as_array() {
            for (i, entry) in board.iter().enumerate() {
                if i >= board_size || entry.is_null() {
                    continue;
                }
                if let Some(card_id) = self.find_card_id_from_json(entry) {
                    let perm_attack = entry["perm_attack"].as_i64().unwrap_or(0) as i32;
                    let perm_health = entry["perm_health"].as_i64().unwrap_or(0) as i32;
                    self.state.shop.board[i] = Some(BoardUnit {
                        card_id,
                        perm_attack,
                        perm_health,
                    });
                }
            }
        }

        // Sync scalars
        if let Some(mana) = state["mana"].as_i64() {
            self.state.shop.shop_mana = mana as i32;
        }
        if let Some(mana_limit) = state["mana_limit"].as_i64() {
            self.state.shop.mana_limit = mana_limit as i32;
        }
        if let Some(round) = state["round"].as_i64() {
            self.state.shop.round = round as i32;
        }
        if let Some(lives) = state["lives"].as_i64() {
            self.state.lives = lives as i32;
        }
        if let Some(wins) = state["wins"].as_i64() {
            self.state.wins = wins as i32;
        }
        if let Some(seed) = state["game_seed"].as_u64() {
            self.state.shop.game_seed = seed;
        }

        // Rebuild bag from JSON bag array
        self.state.bag.clear();
        if let Some(bag) = state["bag"].as_array() {
            for entry in bag {
                let count = entry["count"].as_u64().unwrap_or(0) as usize;
                if let Some(card_id_val) = entry["card_id"].as_u64() {
                    let cid = CardId(card_id_val as u32);
                    for _ in 0..count {
                        self.state.bag.push(cid);
                    }
                }
            }
        }

        self.reset_turn();
        Ok(())
    }

    // ── Legacy JSON API (for play.py and debugging) ──

    fn reset(&mut self, seed: u64, set_id: Option<u32>) -> PyResult<String> {
        let max_rounds_per_game = self.max_rounds_per_game;
        if let Some(set_id) = set_id {
            *self = GameSession::new(seed, set_id)?;
            self.max_rounds_per_game = max_rounds_per_game;
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
            self.reset_turn();
        }
        Ok(self.state_json())
    }

    fn get_state(&self) -> String {
        self.state_json()
    }

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
        self.reset_turn();

        Ok(self.state_json())
    }

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

        let won_game = self.state.wins >= self.state.config.wins_to_victory;
        let lost_game = self.state.lives <= 0;
        let capped_out =
            !won_game && !lost_game && self.max_rounds_per_game > 0 && completed_round >= self.max_rounds_per_game;
        let game_over = won_game || lost_game || capped_out;
        let game_result = if game_over {
            if capped_out {
                self.state.lives = 0;
            }
            self.state.phase = GamePhase::Completed;
            Some(if won_game {
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

        self.reset_turn();

        let reward = if capped_out {
            -1
        } else {
            match &outcome.result {
                BattleResult::Victory => 1,
                BattleResult::Defeat => -1,
                BattleResult::Draw => 0,
            }
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
            "termination_reason": if capped_out { Some("max_rounds") } else { None::<&str> },
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

impl GameSession {
    /// Find a CardId from a JSON card entry by matching id or name.
    fn find_card_id_from_json(&self, entry: &serde_json::Value) -> Option<CardId> {
        // Try direct card_id field first
        if let Some(id) = entry["card_id"].as_u64() {
            let cid = CardId(id as u32);
            if self.state.shop.card_pool.contains_key(&cid) {
                return Some(cid);
            }
        }
        // Try "id" field (from CardView serialization)
        if let Some(id) = entry["id"].as_u64() {
            let cid = CardId(id as u32);
            if self.state.shop.card_pool.contains_key(&cid) {
                return Some(cid);
            }
        }
        // Try matching by name
        if let Some(name) = entry["name"].as_str() {
            for (cid, card) in &self.state.shop.card_pool {
                if card.name == name {
                    return Some(*cid);
                }
            }
        }
        None
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
