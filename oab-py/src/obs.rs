//! Observation encoding for the RL agent.
//!
//! Encodes game state into a fixed-size float vector using the actual Rust
//! game types. No JSON serialization, no string scanning — just direct
//! enum matching. If a new ability trigger or effect is added to the game,
//! the Rust compiler will force this encoder to handle it.
//!
//! DESIGN RULE: Encode ALL data from every game type. Do not selectively
//! exclude fields. The model decides what's relevant, not the engineer.
//!
//! All normalization constants and structural sizes are derived dynamically
//! from the actual card pool and game config — nothing is hardcoded.

use std::collections::BTreeMap;
use oab_battle::state::CardSet;
use oab_battle::types::*;
use oab_game::GameConfig;

// ── Fixed structural sizes from game type definitions ──
// These come from the Rust enum variant counts and are truly constant.

const NUM_BATTLE_TRIGGERS: usize = 13;
const NUM_SHOP_TRIGGERS: usize = 6;
const NUM_BATTLE_EFFECTS: usize = 6;
const NUM_SHOP_EFFECTS: usize = 4;
const NUM_TARGET_SCOPES: usize = 7;
const NUM_SHOP_SCOPES: usize = 5;
const NUM_TARGET_MODES: usize = 5;   // Position, Adjacent, Random, Standard, All
const NUM_SHOP_TARGET_MODES: usize = 4; // Position, Random, Standard, All
const NUM_SPAWN_LOCATIONS: usize = 3; // Front, Back, DeathPosition
const NUM_STAT_TYPES: usize = 3;     // Health, Attack, Mana
const NUM_SORT_ORDERS: usize = 2;    // Ascending, Descending
const NUM_COMPARE_OPS: usize = 5;    // GT, LT, EQ, GTE, LTE
const NUM_BATTLE_MATCHERS: usize = 5; // StatValueCompare, TargetStatValueCompare, StatStatCompare, UnitCount, IsPosition
const NUM_SHOP_MATCHERS: usize = 3;  // StatValueCompare, UnitCount, IsPosition

// ── Normalization constants derived from card pool + config ──

/// Dynamic normalization and layout constants computed from actual game data.
#[derive(Debug, Clone)]
pub struct ObsConstants {
    // Normalization maxes (from card pool scan)
    pub max_attack: f32,
    pub max_health: f32,
    pub max_cost: f32,
    pub max_burn: f32,
    pub max_card_id: f32,
    pub max_effect_value: f32,
    pub max_target_count: f32,
    pub max_triggers_limit: f32,
    pub max_condition_value: f32,
    // From game config
    pub max_mana: f32,
    pub max_round: f32,
    pub max_lives: f32,
    pub max_wins: f32,
    pub max_bag: f32,
    pub max_position_index: f32,
    // Layout sizes (from card pool scan)
    pub hand_size: usize,
    pub board_size: usize,
    pub max_battle_abilities: usize,
    pub max_shop_abilities: usize,
    pub max_conditions_per_ability: usize,
    pub max_bag_card_types: usize,
    // Derived sizes (computed from above)
    pub battle_matcher_features: usize,
    pub shop_matcher_features: usize,
    pub battle_condition_features: usize,
    pub shop_condition_features: usize,
    pub battle_ability_features: usize,
    pub shop_ability_features: usize,
    pub ability_features: usize,
    pub hand_features: usize,
    pub board_features: usize,
    pub bag_card_features: usize,
    pub obs_dim: usize,
}

const HAND_BASE: usize = 7; // presence, id, atk, hp, cost, burn, can_afford
const BOARD_BASE: usize = 8; // presence, id, atk, hp, cost, burn, perm_atk, perm_hp
const SCALAR_FEATURES: usize = 6;
const BAG_CARD_BASE: usize = 6; // count_fraction, id, atk, hp, cost, burn

// Per battle ability (not counting conditions):
//   has_ability(1) + trigger(13) + effect_type(6) +
//   damage(1) + buff_atk(1) + buff_hp(1) +
//   spawn_id(1) + spawn_atk(1) + spawn_hp(1) + spawn_loc(3) +
//   gain_mana(1) +
//   target_scope(7) + target_mode(5) + target_count(1) +
//   target_stat(3) + target_order(2) + target_pos(1) +
//   max_triggers(1) + num_conditions(1)
const BATTLE_ABILITY_BASE: usize = 1 + NUM_BATTLE_TRIGGERS + NUM_BATTLE_EFFECTS
    + 3 + 3 + NUM_SPAWN_LOCATIONS + 1
    + NUM_TARGET_SCOPES + NUM_TARGET_MODES + 1
    + NUM_STAT_TYPES + NUM_SORT_ORDERS + 1
    + 1 + 1; // 51

// Per shop ability (not counting conditions):
const SHOP_ABILITY_BASE: usize = 1 + NUM_SHOP_TRIGGERS + NUM_SHOP_EFFECTS
    + 2 + 3 + NUM_SPAWN_LOCATIONS + 1
    + NUM_SHOP_SCOPES + NUM_SHOP_TARGET_MODES + 1
    + NUM_STAT_TYPES + NUM_SORT_ORDERS + 1
    + 1 + 1; // 38

// Per battle matcher:
//   type(5) + scope(7) + stat(3) + op(5) + value(1) +
//   target_scope(7) + target_mode(5) + target_count(1) + target_stat(3) + target_order(2) + target_pos(1) +
//   second_stat(3) + second_scope(7) + position_index(1)
const BATTLE_MATCHER_FEATURES: usize = NUM_BATTLE_MATCHERS + NUM_TARGET_SCOPES + NUM_STAT_TYPES
    + NUM_COMPARE_OPS + 1
    + NUM_TARGET_SCOPES + NUM_TARGET_MODES + 1 + NUM_STAT_TYPES + NUM_SORT_ORDERS + 1
    + NUM_STAT_TYPES + NUM_TARGET_SCOPES + 1; // 51

// Per shop matcher:
//   type(3) + scope(5) + stat(3) + op(5) + value(1) + position_index(1)
const SHOP_MATCHER_FEATURES: usize = NUM_SHOP_MATCHERS + NUM_SHOP_SCOPES + NUM_STAT_TYPES
    + NUM_COMPARE_OPS + 1 + 1; // 18

impl ObsConstants {
    /// Compute all constants from the active card set and game config.
    /// Only scans cards that are in the set being trained on.
    pub fn from_card_set(
        card_pool: &BTreeMap<CardId, UnitCard>,
        card_set: &CardSet,
        config: &GameConfig,
    ) -> Self {
        // Collect card IDs in this set
        let set_card_ids: std::collections::BTreeSet<CardId> =
            card_set.cards.iter().map(|entry| entry.card_id).collect();

        let mut max_attack = 0i32;
        let mut max_health = 0i32;
        let mut max_cost = 0i32;
        let mut max_burn = 0i32;
        let mut max_card_id = 0u32;
        let mut max_effect_value = 0i32;
        let mut max_target_count = 0u32;
        let mut max_triggers_limit = 0u32;
        let mut max_condition_value = 0i32;
        let mut max_battle_abilities = 0usize;
        let mut max_shop_abilities = 0usize;
        let mut max_conditions = 0usize;

        for card_id in &set_card_ids {
            let card = match card_pool.get(card_id) {
                Some(c) => c,
                None => continue,
            };

            max_card_id = max_card_id.max(card_id.0);
            max_attack = max_attack.max(card.stats.attack);
            max_health = max_health.max(card.stats.health);
            max_cost = max_cost.max(card.economy.play_cost);
            max_burn = max_burn.max(card.economy.burn_value);
            max_battle_abilities = max_battle_abilities.max(card.battle_abilities.len());
            max_shop_abilities = max_shop_abilities.max(card.shop_abilities.len());

            for ability in &card.battle_abilities {
                max_conditions = max_conditions.max(ability.conditions.len());
                if let Some(n) = ability.max_triggers {
                    max_triggers_limit = max_triggers_limit.max(n);
                }
                scan_battle_effect_maxes(
                    &ability.effect,
                    &mut max_effect_value,
                    &mut max_target_count,
                );
                for cond in &ability.conditions {
                    scan_battle_condition_maxes(cond, &mut max_condition_value);
                }
                // Also scan spawned card stats
                if let AbilityEffect::SpawnUnit { card_id: spawn_id, .. } = &ability.effect {
                    if let Some(spawned) = card_pool.get(spawn_id) {
                        max_attack = max_attack.max(spawned.stats.attack);
                        max_health = max_health.max(spawned.stats.health);
                    }
                }
            }
            for ability in &card.shop_abilities {
                max_conditions = max_conditions.max(ability.conditions.len());
                if let Some(n) = ability.max_triggers {
                    max_triggers_limit = max_triggers_limit.max(n);
                }
                scan_shop_effect_maxes(&ability.effect, &mut max_effect_value);
                for cond in &ability.conditions {
                    scan_shop_condition_maxes(cond, &mut max_condition_value);
                }
                if let ShopEffect::SpawnUnit { card_id: spawn_id, .. } = &ability.effect {
                    if let Some(spawned) = card_pool.get(spawn_id) {
                        max_attack = max_attack.max(spawned.stats.attack);
                        max_health = max_health.max(spawned.stats.health);
                    }
                }
            }
        }

        // max_bag_card_types = number of unique cards in this set
        let max_bag_card_types = set_card_ids.len();

        // Add headroom for perm buffs on attack/health
        let perm_headroom = max_effect_value.max(4);
        let max_attack_f = (max_attack + perm_headroom) as f32;
        let max_health_f = (max_health + perm_headroom) as f32;

        // Ensure minimums of 1 to avoid division by zero
        let max_card_id = max_card_id.max(1) as f32;
        let max_cost_f = max_cost.max(1) as f32;
        let max_burn_f = max_burn.max(1) as f32;
        let max_effect_f = max_effect_value.max(1) as f32;
        let max_target_count_f = max_target_count.max(1) as f32;
        let max_triggers_f = max_triggers_limit.max(1) as f32;
        let max_condition_f = max_condition_value.max(1) as f32;

        // Layout sizes — at least 1 to avoid zero-size allocations
        let max_battle_abilities = max_battle_abilities.max(1);
        let max_shop_abilities = max_shop_abilities.max(1);
        let max_conditions_per_ability = max_conditions.max(1);

        // max_bag_card_types already computed from set above

        let hand_size = config.hand_size as usize;
        let board_size = config.board_size as usize;

        // Compute derived sizes
        let battle_condition_features = max_conditions_per_ability * BATTLE_MATCHER_FEATURES;
        let shop_condition_features = max_conditions_per_ability * SHOP_MATCHER_FEATURES;
        let battle_ability_features = BATTLE_ABILITY_BASE + battle_condition_features;
        let shop_ability_features = SHOP_ABILITY_BASE + shop_condition_features;
        let ability_features = max_battle_abilities * battle_ability_features
            + max_shop_abilities * shop_ability_features;
        let hand_features = HAND_BASE + ability_features;
        let board_features = BOARD_BASE + ability_features;
        let bag_card_features = BAG_CARD_BASE + ability_features;
        let bag_features = max_bag_card_types * bag_card_features;

        let obs_dim = hand_size * hand_features
            + board_size * board_features
            + SCALAR_FEATURES
            + bag_features;

        Self {
            max_attack: max_attack_f,
            max_health: max_health_f,
            max_cost: max_cost_f,
            max_burn: max_burn_f,
            max_card_id,
            max_effect_value: max_effect_f,
            max_target_count: max_target_count_f,
            max_triggers_limit: max_triggers_f,
            max_condition_value: max_condition_f,
            max_mana: config.max_mana_limit as f32,
            max_round: 20.0, // practical max (wins_to_victory + starting_lives + draws)
            max_lives: config.starting_lives as f32,
            max_wins: config.wins_to_victory as f32,
            max_bag: config.bag_size as f32,
            max_position_index: config.board_size as f32,
            hand_size,
            board_size,
            max_battle_abilities,
            max_shop_abilities,
            max_conditions_per_ability,
            max_bag_card_types,
            battle_matcher_features: BATTLE_MATCHER_FEATURES,
            shop_matcher_features: SHOP_MATCHER_FEATURES,
            battle_condition_features,
            shop_condition_features,
            battle_ability_features,
            shop_ability_features,
            ability_features,
            hand_features,
            board_features,
            bag_card_features,
            obs_dim,
        }
    }
}

fn scan_battle_effect_maxes(effect: &AbilityEffect, max_val: &mut i32, max_count: &mut u32) {
    match effect {
        AbilityEffect::Damage { amount, target } => {
            *max_val = (*max_val).max(amount.abs());
            scan_battle_target_count(target, max_count);
        }
        AbilityEffect::ModifyStats { health, attack, target } => {
            *max_val = (*max_val).max(attack.abs()).max(health.abs());
            scan_battle_target_count(target, max_count);
        }
        AbilityEffect::ModifyStatsPermanent { health, attack, target } => {
            *max_val = (*max_val).max(attack.abs()).max(health.abs());
            scan_battle_target_count(target, max_count);
        }
        AbilityEffect::GainMana { amount } => {
            *max_val = (*max_val).max(amount.abs());
        }
        AbilityEffect::SpawnUnit { .. } | AbilityEffect::Destroy { .. } => {}
    }
}

fn scan_battle_target_count(target: &AbilityTarget, max_count: &mut u32) {
    match target {
        AbilityTarget::Random { count, .. } | AbilityTarget::Standard { count, .. } => {
            *max_count = (*max_count).max(*count);
        }
        _ => {}
    }
}

fn scan_shop_effect_maxes(effect: &ShopEffect, max_val: &mut i32) {
    match effect {
        ShopEffect::ModifyStatsPermanent { health, attack, .. } => {
            *max_val = (*max_val).max(attack.abs()).max(health.abs());
        }
        ShopEffect::GainMana { amount } => {
            *max_val = (*max_val).max(amount.abs());
        }
        ShopEffect::SpawnUnit { .. } | ShopEffect::Destroy { .. } => {}
    }
}

fn scan_battle_condition_maxes(cond: &Condition, max_val: &mut i32) {
    match cond {
        Condition::Is(m) => scan_matcher_value(m, max_val),
        Condition::AnyOf(ms) => {
            for m in ms {
                scan_matcher_value(m, max_val);
            }
        }
    }
}

fn scan_shop_condition_maxes(cond: &ShopCondition, max_val: &mut i32) {
    match cond {
        ShopCondition::Is(m) => scan_shop_matcher_value(m, max_val),
        ShopCondition::AnyOf(ms) => {
            for m in ms {
                scan_shop_matcher_value(m, max_val);
            }
        }
    }
}

fn scan_matcher_value(m: &Matcher, max_val: &mut i32) {
    match m {
        Matcher::StatValueCompare { value, .. } => *max_val = (*max_val).max(value.abs()),
        Matcher::TargetStatValueCompare { value, .. } => *max_val = (*max_val).max(value.abs()),
        Matcher::UnitCount { value, .. } => *max_val = (*max_val).max(*value as i32),
        Matcher::IsPosition { index, .. } => *max_val = (*max_val).max(index.abs()),
        Matcher::StatStatCompare { .. } => {}
    }
}

fn scan_shop_matcher_value(m: &ShopMatcher, max_val: &mut i32) {
    match m {
        ShopMatcher::StatValueCompare { value, .. } => *max_val = (*max_val).max(value.abs()),
        ShopMatcher::UnitCount { value, .. } => *max_val = (*max_val).max(*value as i32),
        ShopMatcher::IsPosition { index, .. } => *max_val = (*max_val).max(index.abs()),
    }
}

// ── Helpers ──

fn norm(value: f32, max_val: f32) -> f32 {
    if max_val > 0.0 {
        (value / max_val).min(1.0).max(0.0)
    } else {
        0.0
    }
}

// ── Battle Matcher Encoding ──

fn encode_battle_matcher(matcher: &Matcher, c: &ObsConstants, out: &mut [f32]) {
    match matcher {
        Matcher::StatValueCompare { scope, stat, op, value } => {
            out[0] = 1.0;
            out[NUM_BATTLE_MATCHERS + scope_index(scope)] = 1.0;
            out[NUM_BATTLE_MATCHERS + NUM_TARGET_SCOPES + stat_type_index(stat)] = 1.0;
            out[NUM_BATTLE_MATCHERS + NUM_TARGET_SCOPES + NUM_STAT_TYPES + compare_op_index(op)] = 1.0;
            out[NUM_BATTLE_MATCHERS + NUM_TARGET_SCOPES + NUM_STAT_TYPES + NUM_COMPARE_OPS] =
                norm(*value as f32, c.max_condition_value);
        }
        Matcher::TargetStatValueCompare { target, stat, op, value } => {
            out[1] = 1.0;
            out[NUM_BATTLE_MATCHERS + NUM_TARGET_SCOPES + stat_type_index(stat)] = 1.0;
            out[NUM_BATTLE_MATCHERS + NUM_TARGET_SCOPES + NUM_STAT_TYPES + compare_op_index(op)] = 1.0;
            out[NUM_BATTLE_MATCHERS + NUM_TARGET_SCOPES + NUM_STAT_TYPES + NUM_COMPARE_OPS] =
                norm(*value as f32, c.max_condition_value);
            // Encode target at offset after value
            let t_off = NUM_BATTLE_MATCHERS + NUM_TARGET_SCOPES + NUM_STAT_TYPES + NUM_COMPARE_OPS + 1;
            encode_battle_target_into(target, c, &mut out[t_off..]);
        }
        Matcher::StatStatCompare { source_stat, op, target_scope, target_stat } => {
            out[2] = 1.0;
            out[NUM_BATTLE_MATCHERS + NUM_TARGET_SCOPES + stat_type_index(source_stat)] = 1.0;
            out[NUM_BATTLE_MATCHERS + NUM_TARGET_SCOPES + NUM_STAT_TYPES + compare_op_index(op)] = 1.0;
            // second_stat and second_scope at end
            let ss_off = BATTLE_MATCHER_FEATURES - 1 - NUM_TARGET_SCOPES - NUM_STAT_TYPES;
            out[ss_off + stat_type_index(target_stat)] = 1.0;
            out[ss_off + NUM_STAT_TYPES + scope_index(target_scope)] = 1.0;
        }
        Matcher::UnitCount { scope, op, value } => {
            out[3] = 1.0;
            out[NUM_BATTLE_MATCHERS + scope_index(scope)] = 1.0;
            out[NUM_BATTLE_MATCHERS + NUM_TARGET_SCOPES + NUM_STAT_TYPES + compare_op_index(op)] = 1.0;
            out[NUM_BATTLE_MATCHERS + NUM_TARGET_SCOPES + NUM_STAT_TYPES + NUM_COMPARE_OPS] =
                norm(*value as f32, c.max_condition_value);
        }
        Matcher::IsPosition { scope, index } => {
            out[4] = 1.0;
            out[NUM_BATTLE_MATCHERS + scope_index(scope)] = 1.0;
            out[BATTLE_MATCHER_FEATURES - 1] = norm(*index as f32, c.max_position_index);
        }
    }
}

fn encode_shop_matcher(matcher: &ShopMatcher, c: &ObsConstants, out: &mut [f32]) {
    match matcher {
        ShopMatcher::StatValueCompare { scope, stat, op, value } => {
            out[0] = 1.0;
            out[NUM_SHOP_MATCHERS + shop_scope_index(scope)] = 1.0;
            out[NUM_SHOP_MATCHERS + NUM_SHOP_SCOPES + stat_type_index(stat)] = 1.0;
            out[NUM_SHOP_MATCHERS + NUM_SHOP_SCOPES + NUM_STAT_TYPES + compare_op_index(op)] = 1.0;
            out[NUM_SHOP_MATCHERS + NUM_SHOP_SCOPES + NUM_STAT_TYPES + NUM_COMPARE_OPS] =
                norm(*value as f32, c.max_condition_value);
        }
        ShopMatcher::UnitCount { scope, op, value } => {
            out[1] = 1.0;
            out[NUM_SHOP_MATCHERS + shop_scope_index(scope)] = 1.0;
            out[NUM_SHOP_MATCHERS + NUM_SHOP_SCOPES + NUM_STAT_TYPES + compare_op_index(op)] = 1.0;
            out[NUM_SHOP_MATCHERS + NUM_SHOP_SCOPES + NUM_STAT_TYPES + NUM_COMPARE_OPS] =
                norm(*value as f32, c.max_condition_value);
        }
        ShopMatcher::IsPosition { scope, index } => {
            out[2] = 1.0;
            out[NUM_SHOP_MATCHERS + shop_scope_index(scope)] = 1.0;
            out[SHOP_MATCHER_FEATURES - 1] = norm(*index as f32, c.max_position_index);
        }
    }
}

// ── Battle Ability Encoding ──

fn encode_battle_ability(
    ability: &Ability,
    c: &ObsConstants,
    card_pool: &BTreeMap<CardId, UnitCard>,
    out: &mut [f32],
) {
    let mut idx = 0;

    out[idx] = 1.0; idx += 1; // has_ability

    // trigger (13)
    let mut trigger = [0.0f32; NUM_BATTLE_TRIGGERS];
    trigger[ability.trigger.index()] = 1.0;
    out[idx..idx + NUM_BATTLE_TRIGGERS].copy_from_slice(&trigger);
    idx += NUM_BATTLE_TRIGGERS;

    // effect type + all sub-fields
    let mut effect_type = [0.0f32; NUM_BATTLE_EFFECTS];
    let mut damage = 0.0f32;
    let mut buff_atk = 0.0f32;
    let mut buff_hp = 0.0f32;
    let mut spawn_id = 0.0f32;
    let mut spawn_atk = 0.0f32;
    let mut spawn_hp = 0.0f32;
    let mut spawn_loc = [0.0f32; NUM_SPAWN_LOCATIONS];
    let mut gain_mana = 0.0f32;
    let mut target_scope = [0.0f32; NUM_TARGET_SCOPES];
    let mut target_mode = [0.0f32; NUM_TARGET_MODES];
    let mut target_count = 0.0f32;
    let mut target_stat = [0.0f32; NUM_STAT_TYPES];
    let mut target_order = [0.0f32; NUM_SORT_ORDERS];
    let mut target_pos = 0.0f32;

    match &ability.effect {
        AbilityEffect::Damage { amount, target } => {
            effect_type[0] = 1.0;
            damage = norm(amount.abs() as f32, c.max_effect_value);
            fill_battle_target(target, c, &mut target_scope, &mut target_mode, &mut target_count, &mut target_stat, &mut target_order, &mut target_pos);
        }
        AbilityEffect::ModifyStats { health, attack, target } => {
            effect_type[1] = 1.0;
            buff_atk = norm(*attack as f32, c.max_effect_value);
            buff_hp = norm(*health as f32, c.max_effect_value);
            fill_battle_target(target, c, &mut target_scope, &mut target_mode, &mut target_count, &mut target_stat, &mut target_order, &mut target_pos);
        }
        AbilityEffect::ModifyStatsPermanent { health, attack, target } => {
            effect_type[2] = 1.0;
            buff_atk = norm(*attack as f32, c.max_effect_value);
            buff_hp = norm(*health as f32, c.max_effect_value);
            fill_battle_target(target, c, &mut target_scope, &mut target_mode, &mut target_count, &mut target_stat, &mut target_order, &mut target_pos);
        }
        AbilityEffect::SpawnUnit { card_id, spawn_location: loc } => {
            effect_type[3] = 1.0;
            spawn_id = norm(card_id.0 as f32, c.max_card_id);
            if let Some(card) = card_pool.get(card_id) {
                spawn_atk = norm(card.stats.attack as f32, c.max_attack);
                spawn_hp = norm(card.stats.health as f32, c.max_health);
            }
            match loc {
                SpawnLocation::Front => spawn_loc[0] = 1.0,
                SpawnLocation::Back => spawn_loc[1] = 1.0,
                SpawnLocation::DeathPosition => spawn_loc[2] = 1.0,
            }
        }
        AbilityEffect::Destroy { target } => {
            effect_type[4] = 1.0;
            fill_battle_target(target, c, &mut target_scope, &mut target_mode, &mut target_count, &mut target_stat, &mut target_order, &mut target_pos);
        }
        AbilityEffect::GainMana { amount } => {
            effect_type[5] = 1.0;
            gain_mana = norm(*amount as f32, c.max_effect_value);
        }
    }

    out[idx..idx + NUM_BATTLE_EFFECTS].copy_from_slice(&effect_type); idx += NUM_BATTLE_EFFECTS;
    out[idx] = damage; idx += 1;
    out[idx] = buff_atk; idx += 1;
    out[idx] = buff_hp; idx += 1;
    out[idx] = spawn_id; idx += 1;
    out[idx] = spawn_atk; idx += 1;
    out[idx] = spawn_hp; idx += 1;
    out[idx..idx + NUM_SPAWN_LOCATIONS].copy_from_slice(&spawn_loc); idx += NUM_SPAWN_LOCATIONS;
    out[idx] = gain_mana; idx += 1;
    out[idx..idx + NUM_TARGET_SCOPES].copy_from_slice(&target_scope); idx += NUM_TARGET_SCOPES;
    out[idx..idx + NUM_TARGET_MODES].copy_from_slice(&target_mode); idx += NUM_TARGET_MODES;
    out[idx] = target_count; idx += 1;
    out[idx..idx + NUM_STAT_TYPES].copy_from_slice(&target_stat); idx += NUM_STAT_TYPES;
    out[idx..idx + NUM_SORT_ORDERS].copy_from_slice(&target_order); idx += NUM_SORT_ORDERS;
    out[idx] = target_pos; idx += 1;
    // max_triggers
    out[idx] = match ability.max_triggers {
        Some(n) => norm(n as f32, c.max_triggers_limit),
        None => 0.0,
    };
    idx += 1;
    // num_conditions
    out[idx] = norm(ability.conditions.len() as f32, c.max_conditions_per_ability as f32);
    idx += 1;
    // conditions
    for ci in 0..c.max_conditions_per_ability {
        let c_start = idx + ci * BATTLE_MATCHER_FEATURES;
        if let Some(condition) = ability.conditions.get(ci) {
            match condition {
                Condition::Is(matcher) => encode_battle_matcher(matcher, c, &mut out[c_start..c_start + BATTLE_MATCHER_FEATURES]),
                Condition::AnyOf(matchers) => {
                    if let Some(m) = matchers.first() {
                        encode_battle_matcher(m, c, &mut out[c_start..c_start + BATTLE_MATCHER_FEATURES]);
                    }
                }
            }
        }
    }
}

fn encode_shop_ability(
    ability: &ShopAbility,
    c: &ObsConstants,
    card_pool: &BTreeMap<CardId, UnitCard>,
    out: &mut [f32],
) {
    let mut idx = 0;

    out[idx] = 1.0; idx += 1;

    let mut trigger = [0.0f32; NUM_SHOP_TRIGGERS];
    trigger[shop_trigger_index(&ability.trigger)] = 1.0;
    out[idx..idx + NUM_SHOP_TRIGGERS].copy_from_slice(&trigger); idx += NUM_SHOP_TRIGGERS;

    let mut effect_type = [0.0f32; NUM_SHOP_EFFECTS];
    let mut buff_atk = 0.0f32;
    let mut buff_hp = 0.0f32;
    let mut spawn_id = 0.0f32;
    let mut spawn_atk = 0.0f32;
    let mut spawn_hp = 0.0f32;
    let mut spawn_loc = [0.0f32; NUM_SPAWN_LOCATIONS];
    let mut gain_mana = 0.0f32;
    let mut target_scope = [0.0f32; NUM_SHOP_SCOPES];
    let mut target_mode = [0.0f32; NUM_SHOP_TARGET_MODES];
    let mut target_count = 0.0f32;
    let mut target_stat = [0.0f32; NUM_STAT_TYPES];
    let mut target_order = [0.0f32; NUM_SORT_ORDERS];
    let mut target_pos = 0.0f32;

    match &ability.effect {
        ShopEffect::ModifyStatsPermanent { health, attack, target } => {
            effect_type[0] = 1.0;
            buff_atk = norm(*attack as f32, c.max_effect_value);
            buff_hp = norm(*health as f32, c.max_effect_value);
            fill_shop_target(target, c, &mut target_scope, &mut target_mode, &mut target_count, &mut target_stat, &mut target_order, &mut target_pos);
        }
        ShopEffect::SpawnUnit { card_id, spawn_location: loc } => {
            effect_type[1] = 1.0;
            spawn_id = norm(card_id.0 as f32, c.max_card_id);
            if let Some(card) = card_pool.get(card_id) {
                spawn_atk = norm(card.stats.attack as f32, c.max_attack);
                spawn_hp = norm(card.stats.health as f32, c.max_health);
            }
            match loc {
                SpawnLocation::Front => spawn_loc[0] = 1.0,
                SpawnLocation::Back => spawn_loc[1] = 1.0,
                SpawnLocation::DeathPosition => spawn_loc[2] = 1.0,
            }
        }
        ShopEffect::Destroy { target } => {
            effect_type[2] = 1.0;
            fill_shop_target(target, c, &mut target_scope, &mut target_mode, &mut target_count, &mut target_stat, &mut target_order, &mut target_pos);
        }
        ShopEffect::GainMana { amount } => {
            effect_type[3] = 1.0;
            gain_mana = norm(*amount as f32, c.max_effect_value);
        }
    }

    out[idx..idx + NUM_SHOP_EFFECTS].copy_from_slice(&effect_type); idx += NUM_SHOP_EFFECTS;
    out[idx] = buff_atk; idx += 1;
    out[idx] = buff_hp; idx += 1;
    out[idx] = spawn_id; idx += 1;
    out[idx] = spawn_atk; idx += 1;
    out[idx] = spawn_hp; idx += 1;
    out[idx..idx + NUM_SPAWN_LOCATIONS].copy_from_slice(&spawn_loc); idx += NUM_SPAWN_LOCATIONS;
    out[idx] = gain_mana; idx += 1;
    out[idx..idx + NUM_SHOP_SCOPES].copy_from_slice(&target_scope); idx += NUM_SHOP_SCOPES;
    out[idx..idx + NUM_SHOP_TARGET_MODES].copy_from_slice(&target_mode); idx += NUM_SHOP_TARGET_MODES;
    out[idx] = target_count; idx += 1;
    out[idx..idx + NUM_STAT_TYPES].copy_from_slice(&target_stat); idx += NUM_STAT_TYPES;
    out[idx..idx + NUM_SORT_ORDERS].copy_from_slice(&target_order); idx += NUM_SORT_ORDERS;
    out[idx] = target_pos; idx += 1;
    out[idx] = match ability.max_triggers {
        Some(n) => norm(n as f32, c.max_triggers_limit),
        None => 0.0,
    };
    idx += 1;
    out[idx] = norm(ability.conditions.len() as f32, c.max_conditions_per_ability as f32);
    idx += 1;
    for ci in 0..c.max_conditions_per_ability {
        let c_start = idx + ci * SHOP_MATCHER_FEATURES;
        if let Some(condition) = ability.conditions.get(ci) {
            match condition {
                ShopCondition::Is(matcher) => encode_shop_matcher(matcher, c, &mut out[c_start..c_start + SHOP_MATCHER_FEATURES]),
                ShopCondition::AnyOf(matchers) => {
                    if let Some(m) = matchers.first() {
                        encode_shop_matcher(m, c, &mut out[c_start..c_start + SHOP_MATCHER_FEATURES]);
                    }
                }
            }
        }
    }
}

// ── Target field helpers ──

fn fill_battle_target(
    target: &AbilityTarget, c: &ObsConstants,
    scope: &mut [f32; NUM_TARGET_SCOPES], mode: &mut [f32; NUM_TARGET_MODES],
    count: &mut f32, stat: &mut [f32; NUM_STAT_TYPES],
    order: &mut [f32; NUM_SORT_ORDERS], pos: &mut f32,
) {
    match target {
        AbilityTarget::Position { scope: s, index: i } => { scope[scope_index(s)] = 1.0; mode[0] = 1.0; *pos = norm(*i as f32, c.max_position_index); }
        AbilityTarget::Adjacent { scope: s } => { scope[scope_index(s)] = 1.0; mode[1] = 1.0; }
        AbilityTarget::Random { scope: s, count: n } => { scope[scope_index(s)] = 1.0; mode[2] = 1.0; *count = norm(*n as f32, c.max_target_count); }
        AbilityTarget::Standard { scope: s, stat: st, order: o, count: n } => { scope[scope_index(s)] = 1.0; mode[3] = 1.0; *count = norm(*n as f32, c.max_target_count); stat[stat_type_index(st)] = 1.0; order[sort_order_index(o)] = 1.0; }
        AbilityTarget::All { scope: s } => { scope[scope_index(s)] = 1.0; mode[4] = 1.0; }
    }
}

fn encode_battle_target_into(target: &AbilityTarget, c: &ObsConstants, out: &mut [f32]) {
    let mut scope = [0.0f32; NUM_TARGET_SCOPES];
    let mut mode = [0.0f32; NUM_TARGET_MODES];
    let mut count = 0.0f32;
    let mut stat = [0.0f32; NUM_STAT_TYPES];
    let mut order = [0.0f32; NUM_SORT_ORDERS];
    let mut pos = 0.0f32;
    fill_battle_target(target, c, &mut scope, &mut mode, &mut count, &mut stat, &mut order, &mut pos);
    let mut i = 0;
    out[i..i+NUM_TARGET_SCOPES].copy_from_slice(&scope); i += NUM_TARGET_SCOPES;
    out[i..i+NUM_TARGET_MODES].copy_from_slice(&mode); i += NUM_TARGET_MODES;
    out[i] = count; i += 1;
    out[i..i+NUM_STAT_TYPES].copy_from_slice(&stat); i += NUM_STAT_TYPES;
    out[i..i+NUM_SORT_ORDERS].copy_from_slice(&order); i += NUM_SORT_ORDERS;
    out[i] = pos;
}

fn fill_shop_target(
    target: &ShopTarget, c: &ObsConstants,
    scope: &mut [f32; NUM_SHOP_SCOPES], mode: &mut [f32; NUM_SHOP_TARGET_MODES],
    count: &mut f32, stat: &mut [f32; NUM_STAT_TYPES],
    order: &mut [f32; NUM_SORT_ORDERS], pos: &mut f32,
) {
    match target {
        ShopTarget::Position { scope: s, index: i } => { scope[shop_scope_index(s)] = 1.0; mode[0] = 1.0; *pos = norm(*i as f32, c.max_position_index); }
        ShopTarget::Random { scope: s, count: n } => { scope[shop_scope_index(s)] = 1.0; mode[1] = 1.0; *count = norm(*n as f32, c.max_target_count); }
        ShopTarget::Standard { scope: s, stat: st, order: o, count: n } => { scope[shop_scope_index(s)] = 1.0; mode[2] = 1.0; *count = norm(*n as f32, c.max_target_count); stat[stat_type_index(st)] = 1.0; order[sort_order_index(o)] = 1.0; }
        ShopTarget::All { scope: s } => { scope[shop_scope_index(s)] = 1.0; mode[3] = 1.0; }
    }
}

// ── Index helpers ──

fn scope_index(scope: &TargetScope) -> usize {
    match scope { TargetScope::SelfUnit => 0, TargetScope::Allies => 1, TargetScope::Enemies => 2, TargetScope::All => 3, TargetScope::AlliesOther => 4, TargetScope::TriggerSource => 5, TargetScope::Aggressor => 6 }
}
fn shop_scope_index(scope: &ShopScope) -> usize {
    match scope { ShopScope::SelfUnit => 0, ShopScope::Allies => 1, ShopScope::All => 2, ShopScope::AlliesOther => 3, ShopScope::TriggerSource => 4 }
}
fn shop_trigger_index(trigger: &ShopTrigger) -> usize {
    match trigger { ShopTrigger::OnBuy => 0, ShopTrigger::OnSell => 1, ShopTrigger::OnShopStart => 2, ShopTrigger::AfterLoss => 3, ShopTrigger::AfterWin => 4, ShopTrigger::AfterDraw => 5 }
}
fn stat_type_index(stat: &StatType) -> usize {
    match stat { StatType::Health => 0, StatType::Attack => 1, StatType::Mana => 2 }
}
fn sort_order_index(order: &SortOrder) -> usize {
    match order { SortOrder::Ascending => 0, SortOrder::Descending => 1 }
}
fn compare_op_index(op: &CompareOp) -> usize {
    match op { CompareOp::GreaterThan => 0, CompareOp::LessThan => 1, CompareOp::Equal => 2, CompareOp::GreaterThanOrEqual => 3, CompareOp::LessThanOrEqual => 4 }
}

// ── Full Card Ability Encoding ──

fn encode_card_abilities(
    card: &UnitCard, c: &ObsConstants,
    card_pool: &BTreeMap<CardId, UnitCard>,
    out: &mut [f32],
) {
    let mut idx = 0;
    for i in 0..c.max_battle_abilities {
        if let Some(ability) = card.battle_abilities.get(i) {
            encode_battle_ability(ability, c, card_pool, &mut out[idx..idx + c.battle_ability_features]);
        }
        idx += c.battle_ability_features;
    }
    for i in 0..c.max_shop_abilities {
        if let Some(ability) = card.shop_abilities.get(i) {
            encode_shop_ability(ability, c, card_pool, &mut out[idx..idx + c.shop_ability_features]);
        }
        idx += c.shop_ability_features;
    }
}

// ── Full Observation Encoding ──

pub fn encode_observation(
    shadow_hand: &[Option<CardId>],
    shadow_board: &[Option<BoardUnit>],
    shadow_mana: i32,
    state: &oab_game::GameState,
    c: &ObsConstants,
) -> Vec<f32> {
    let mut obs = vec![0.0f32; c.obs_dim];
    let pool = &state.shop.card_pool;
    let mut idx = 0;

    // Hand cards
    for i in 0..c.hand_size {
        if let Some(Some(cid)) = shadow_hand.get(i) {
            if let Some(card) = pool.get(cid) {
                obs[idx + 0] = 1.0;
                obs[idx + 1] = norm(cid.0 as f32, c.max_card_id);
                obs[idx + 2] = norm(card.stats.attack as f32, c.max_attack);
                obs[idx + 3] = norm(card.stats.health as f32, c.max_health);
                obs[idx + 4] = norm(card.economy.play_cost as f32, c.max_cost);
                obs[idx + 5] = norm(card.economy.burn_value as f32, c.max_burn);
                obs[idx + 6] = if shadow_mana >= card.economy.play_cost { 1.0 } else { 0.0 };
                encode_card_abilities(card, c, pool, &mut obs[idx + HAND_BASE..idx + c.hand_features]);
            }
        }
        idx += c.hand_features;
    }

    // Board units
    for i in 0..c.board_size {
        if let Some(Some(bu)) = shadow_board.get(i) {
            if let Some(card) = pool.get(&bu.card_id) {
                obs[idx + 0] = 1.0;
                obs[idx + 1] = norm(bu.card_id.0 as f32, c.max_card_id);
                obs[idx + 2] = norm((card.stats.attack + bu.perm_attack) as f32, c.max_attack);
                obs[idx + 3] = norm((card.stats.health + bu.perm_health) as f32, c.max_health);
                obs[idx + 4] = norm(card.economy.play_cost as f32, c.max_cost);
                obs[idx + 5] = norm(card.economy.burn_value as f32, c.max_burn);
                obs[idx + 6] = norm(bu.perm_attack as f32, c.max_attack);
                obs[idx + 7] = norm(bu.perm_health as f32, c.max_health);
                encode_card_abilities(card, c, pool, &mut obs[idx + BOARD_BASE..idx + c.board_features]);
            }
        }
        idx += c.board_features;
    }

    // Scalars
    obs[idx + 0] = norm(shadow_mana as f32, c.max_mana);
    obs[idx + 1] = norm(state.shop.mana_limit as f32, c.max_mana);
    obs[idx + 2] = norm(state.shop.round as f32, c.max_round);
    obs[idx + 3] = norm(state.lives as f32, c.max_lives);
    obs[idx + 4] = norm(state.wins as f32, c.max_wins);
    obs[idx + 5] = norm(state.bag.len() as f32, c.max_bag);
    idx += SCALAR_FEATURES;

    // Bag: full card data per unique type INCLUDING abilities
    let mut bag_counts: BTreeMap<CardId, u32> = BTreeMap::new();
    for card_id in &state.bag {
        *bag_counts.entry(*card_id).or_insert(0) += 1;
    }
    let total_bag = state.bag.len().max(1) as f32;
    for (ci, (card_id, count)) in bag_counts.iter().enumerate() {
        if ci >= c.max_bag_card_types {
            break;
        }
        let base = idx + ci * c.bag_card_features;
        obs[base + 0] = *count as f32 / total_bag;
        obs[base + 1] = norm(card_id.0 as f32, c.max_card_id);
        if let Some(card) = pool.get(card_id) {
            obs[base + 2] = norm(card.stats.attack as f32, c.max_attack);
            obs[base + 3] = norm(card.stats.health as f32, c.max_health);
            obs[base + 4] = norm(card.economy.play_cost as f32, c.max_cost);
            obs[base + 5] = norm(card.economy.burn_value as f32, c.max_burn);
            encode_card_abilities(card, c, pool, &mut obs[base + BAG_CARD_BASE..base + c.bag_card_features]);
        }
    }

    obs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_structural_constants() {
        assert_eq!(BATTLE_ABILITY_BASE, 51);
        assert_eq!(SHOP_ABILITY_BASE, 38);
        assert_eq!(BATTLE_MATCHER_FEATURES, 51);
        assert_eq!(SHOP_MATCHER_FEATURES, 18);
    }
}
