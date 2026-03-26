//! Observation encoding for the RL agent.
//!
//! Encodes game state into a fixed-size float vector using the actual Rust
//! game types. No JSON serialization, no string scanning — just direct
//! enum matching. If a new ability trigger or effect is added to the game,
//! the Rust compiler will force this encoder to handle it.
//!
//! DESIGN RULE: Encode ALL data from every game type. Do not selectively
//! exclude fields. The model decides what's relevant, not the engineer.

use oab_battle::types::*;

// ── Normalization Constants ──

// Derived from actual card data (assets/data/cards.json) and GameConfig.
// If the game adds stronger cards, update these.
const MAX_ATTACK: f32 = 12.0;      // max base attack across all cards (8) + perm buff headroom
const MAX_HEALTH: f32 = 16.0;      // max base health across all cards (12) + perm buff headroom
const MAX_COST: f32 = 10.0;        // max play_cost (10, matches max mana)
const MAX_BURN: f32 = 3.0;         // max burn_value across all cards
const MAX_MANA: f32 = 10.0;        // GameConfig::max_mana_limit
const MAX_ROUND: f32 = 20.0;       // practical max rounds before game ends
const MAX_LIVES: f32 = 3.0;        // GameConfig::starting_lives
const MAX_WINS: f32 = 10.0;        // GameConfig::wins_to_victory
const MAX_BAG: f32 = 50.0;         // GameConfig::bag_size
const MAX_CARD_ID: f32 = 120.0;    // 111 cards exist, leave headroom
const MAX_EFFECT_VALUE: f32 = 8.0;  // max damage amount across all cards
const MAX_TARGET_COUNT: f32 = 5.0;  // max random/standard target count
const MAX_TRIGGERS_LIMIT: f32 = 5.0; // max_triggers cap observed
const MAX_POSITION_INDEX: f32 = 5.0; // board size
const MAX_CONDITION_VALUE: f32 = 10.0; // condition comparison values

// ── Observation Layout ──

pub const HAND_SIZE: usize = 5;
pub const BOARD_SIZE: usize = 5;
const MAX_BATTLE_ABILITIES: usize = 3;
const MAX_SHOP_ABILITIES: usize = 2;
const MAX_CONDITIONS_PER_ABILITY: usize = 1; // max observed in card data
const MAX_BAG_CARD_TYPES: usize = 70; // largest set has 70 unique card types

const HAND_BASE: usize = 7; // presence, id, atk, hp, cost, burn, can_afford
const BOARD_BASE: usize = 8; // presence, id, atk, hp, cost, burn, perm_atk, perm_hp

// ── Condition encoding ──
// Battle Matcher (5 variants):
//   type_onehot(5) + scope(7) + stat(3) + compare_op(5) + value(1) +
//   target_scope(7) + target_mode(5) + target_count(1) + target_stat(3) + target_order(2) + target_pos(1) +
//   second_stat(3) + second_scope(7) + position_index(1)
//   = 56
const BATTLE_MATCHER_FEATURES: usize = 56;
const BATTLE_CONDITION_FEATURES: usize = MAX_CONDITIONS_PER_ABILITY * BATTLE_MATCHER_FEATURES;

// Shop Matcher (3 variants):
//   type_onehot(3) + scope(5) + stat(3) + compare_op(5) + value(1) + position_index(1)
//   = 18
const SHOP_MATCHER_FEATURES: usize = 18;
const SHOP_CONDITION_FEATURES: usize = MAX_CONDITIONS_PER_ABILITY * SHOP_MATCHER_FEATURES;

// ── Ability slot sizes ──
// Per battle ability:
//   has_ability(1) + trigger(13) + effect_type(6) +
//   damage_amount(1) + buff_atk(1) + buff_hp(1) +
//   spawn_card_id(1) + spawn_card_atk(1) + spawn_card_hp(1) + spawn_location(3) +
//   gain_mana_amount(1) +
//   target_scope(7) + target_mode(5) + target_count(1) +
//   target_stat_type(3) + target_sort_order(2) + target_position_index(1) +
//   max_triggers(1) + num_conditions(1) + conditions(BATTLE_CONDITION_FEATURES)
const BATTLE_ABILITY_FEATURES: usize = 51 - 1 + 1 + BATTLE_CONDITION_FEATURES;
// 51 from before, minus has_conditions(1), plus num_conditions(1), plus full conditions

// Per shop ability:
//   has_ability(1) + trigger(6) + effect_type(4) +
//   buff_atk(1) + buff_hp(1) +
//   spawn_card_id(1) + spawn_card_atk(1) + spawn_card_hp(1) + spawn_location(3) +
//   gain_mana_amount(1) +
//   target_scope(5) + target_mode(4) + target_count(1) +
//   target_stat_type(3) + target_sort_order(2) + target_position_index(1) +
//   max_triggers(1) + num_conditions(1) + conditions(SHOP_CONDITION_FEATURES)
const SHOP_ABILITY_FEATURES: usize = 38 - 1 + 1 + SHOP_CONDITION_FEATURES;

pub const ABILITY_FEATURES: usize =
    MAX_BATTLE_ABILITIES * BATTLE_ABILITY_FEATURES + MAX_SHOP_ABILITIES * SHOP_ABILITY_FEATURES;

pub const HAND_FEATURES: usize = HAND_BASE + ABILITY_FEATURES;
pub const BOARD_FEATURES: usize = BOARD_BASE + ABILITY_FEATURES;
const SCALAR_FEATURES: usize = 6;

// Bag: per card type — full card data INCLUDING abilities
const BAG_CARD_BASE: usize = 6; // count_fraction, id, atk, hp, cost, burn
const BAG_CARD_FEATURES: usize = BAG_CARD_BASE + ABILITY_FEATURES;
const BAG_FEATURES: usize = MAX_BAG_CARD_TYPES * BAG_CARD_FEATURES;

pub const OBS_DIM: usize =
    HAND_SIZE * HAND_FEATURES + BOARD_SIZE * BOARD_FEATURES + SCALAR_FEATURES + BAG_FEATURES;

// ── Helpers ──

fn norm(value: f32, max_val: f32) -> f32 {
    if max_val > 0.0 {
        (value / max_val).min(1.0).max(0.0)
    } else {
        0.0
    }
}

// ── Battle Matcher Encoding (56 features) ──

fn encode_battle_matcher(matcher: &Matcher, out: &mut [f32]) {
    // type_onehot: StatValueCompare(0), TargetStatValueCompare(1), StatStatCompare(2), UnitCount(3), IsPosition(4)
    match matcher {
        Matcher::StatValueCompare {
            scope,
            stat,
            op,
            value,
        } => {
            out[0] = 1.0;
            out[5 + scope_index(scope)] = 1.0;
            out[12 + stat_type_index(stat)] = 1.0;
            out[15 + compare_op_index(op)] = 1.0;
            out[20] = norm(*value as f32, MAX_CONDITION_VALUE);
        }
        Matcher::TargetStatValueCompare {
            target,
            stat,
            op,
            value,
        } => {
            out[1] = 1.0;
            out[12 + stat_type_index(stat)] = 1.0;
            out[15 + compare_op_index(op)] = 1.0;
            out[20] = norm(*value as f32, MAX_CONDITION_VALUE);
            // Encode the target into the target fields (offset 21)
            let mut t_scope = [0.0f32; 7];
            let mut t_mode = [0.0f32; 5];
            let mut t_count = 0.0f32;
            let mut t_stat = [0.0f32; 3];
            let mut t_order = [0.0f32; 2];
            let mut t_pos = 0.0f32;
            encode_battle_target_fields(
                target, &mut t_scope, &mut t_mode, &mut t_count,
                &mut t_stat, &mut t_order, &mut t_pos,
            );
            out[21..28].copy_from_slice(&t_scope);
            out[28..33].copy_from_slice(&t_mode);
            out[33] = t_count;
            out[34..37].copy_from_slice(&t_stat);
            out[37..39].copy_from_slice(&t_order);
            out[39] = t_pos;
        }
        Matcher::StatStatCompare {
            source_stat,
            op,
            target_scope,
            target_stat,
        } => {
            out[2] = 1.0;
            out[12 + stat_type_index(source_stat)] = 1.0;
            out[15 + compare_op_index(op)] = 1.0;
            // second_stat at offset 40
            out[40 + stat_type_index(target_stat)] = 1.0;
            // second_scope at offset 43
            out[43 + scope_index(target_scope)] = 1.0;
        }
        Matcher::UnitCount { scope, op, value } => {
            out[3] = 1.0;
            out[5 + scope_index(scope)] = 1.0;
            out[15 + compare_op_index(op)] = 1.0;
            out[20] = norm(*value as f32, MAX_CONDITION_VALUE);
        }
        Matcher::IsPosition { scope, index } => {
            out[4] = 1.0;
            out[5 + scope_index(scope)] = 1.0;
            out[50] = norm(*index as f32, MAX_POSITION_INDEX);
        }
    }
}

// ── Shop Matcher Encoding (18 features) ──

fn encode_shop_matcher(matcher: &ShopMatcher, out: &mut [f32]) {
    // type_onehot: StatValueCompare(0), UnitCount(1), IsPosition(2)
    match matcher {
        ShopMatcher::StatValueCompare {
            scope,
            stat,
            op,
            value,
        } => {
            out[0] = 1.0;
            out[3 + shop_scope_index(scope)] = 1.0;
            out[8 + stat_type_index(stat)] = 1.0;
            out[11 + compare_op_index(op)] = 1.0;
            out[16] = norm(*value as f32, MAX_CONDITION_VALUE);
        }
        ShopMatcher::UnitCount { scope, op, value } => {
            out[1] = 1.0;
            out[3 + shop_scope_index(scope)] = 1.0;
            out[11 + compare_op_index(op)] = 1.0;
            out[16] = norm(*value as f32, MAX_CONDITION_VALUE);
        }
        ShopMatcher::IsPosition { scope, index } => {
            out[2] = 1.0;
            out[3 + shop_scope_index(scope)] = 1.0;
            out[17] = norm(*index as f32, MAX_POSITION_INDEX);
        }
    }
}

// ── Battle Ability Encoding ──

fn encode_battle_ability(
    ability: &Ability,
    card_pool: &std::collections::BTreeMap<CardId, UnitCard>,
    out: &mut [f32],
) {
    let mut idx = 0;

    // has_ability (1)
    out[idx] = 1.0;
    idx += 1;

    // trigger one-hot (13)
    let mut trigger = [0.0f32; 13];
    trigger[ability.trigger.index()] = 1.0;
    out[idx..idx + 13].copy_from_slice(&trigger);
    idx += 13;

    // effect type + all sub-fields
    let mut effect_type = [0.0f32; 6];
    let mut damage_amount = 0.0f32;
    let mut buff_atk = 0.0f32;
    let mut buff_hp = 0.0f32;
    let mut spawn_card_id = 0.0f32;
    let mut spawn_card_atk = 0.0f32;
    let mut spawn_card_hp = 0.0f32;
    let mut spawn_location = [0.0f32; 3];
    let mut gain_mana = 0.0f32;
    let mut target_scope = [0.0f32; 7];
    let mut target_mode = [0.0f32; 5];
    let mut target_count = 0.0f32;
    let mut target_stat = [0.0f32; 3];
    let mut target_order = [0.0f32; 2];
    let mut target_pos_index = 0.0f32;

    match &ability.effect {
        AbilityEffect::Damage { amount, target } => {
            effect_type[0] = 1.0;
            damage_amount = norm(amount.abs() as f32, MAX_EFFECT_VALUE);
            encode_battle_target_fields(target, &mut target_scope, &mut target_mode, &mut target_count, &mut target_stat, &mut target_order, &mut target_pos_index);
        }
        AbilityEffect::ModifyStats { health, attack, target } => {
            effect_type[1] = 1.0;
            buff_atk = norm(*attack as f32, MAX_EFFECT_VALUE);
            buff_hp = norm(*health as f32, MAX_EFFECT_VALUE);
            encode_battle_target_fields(target, &mut target_scope, &mut target_mode, &mut target_count, &mut target_stat, &mut target_order, &mut target_pos_index);
        }
        AbilityEffect::ModifyStatsPermanent { health, attack, target } => {
            effect_type[2] = 1.0;
            buff_atk = norm(*attack as f32, MAX_EFFECT_VALUE);
            buff_hp = norm(*health as f32, MAX_EFFECT_VALUE);
            encode_battle_target_fields(target, &mut target_scope, &mut target_mode, &mut target_count, &mut target_stat, &mut target_order, &mut target_pos_index);
        }
        AbilityEffect::SpawnUnit { card_id, spawn_location: loc } => {
            effect_type[3] = 1.0;
            spawn_card_id = norm(card_id.0 as f32, MAX_CARD_ID);
            if let Some(card) = card_pool.get(card_id) {
                spawn_card_atk = norm(card.stats.attack as f32, MAX_ATTACK);
                spawn_card_hp = norm(card.stats.health as f32, MAX_HEALTH);
            }
            match loc {
                SpawnLocation::Front => spawn_location[0] = 1.0,
                SpawnLocation::Back => spawn_location[1] = 1.0,
                SpawnLocation::DeathPosition => spawn_location[2] = 1.0,
            }
        }
        AbilityEffect::Destroy { target } => {
            effect_type[4] = 1.0;
            encode_battle_target_fields(target, &mut target_scope, &mut target_mode, &mut target_count, &mut target_stat, &mut target_order, &mut target_pos_index);
        }
        AbilityEffect::GainMana { amount } => {
            effect_type[5] = 1.0;
            gain_mana = norm(*amount as f32, MAX_EFFECT_VALUE);
        }
    }

    out[idx..idx + 6].copy_from_slice(&effect_type); idx += 6;
    out[idx] = damage_amount; idx += 1;
    out[idx] = buff_atk; idx += 1;
    out[idx] = buff_hp; idx += 1;
    out[idx] = spawn_card_id; idx += 1;
    out[idx] = spawn_card_atk; idx += 1;
    out[idx] = spawn_card_hp; idx += 1;
    out[idx..idx + 3].copy_from_slice(&spawn_location); idx += 3;
    out[idx] = gain_mana; idx += 1;
    out[idx..idx + 7].copy_from_slice(&target_scope); idx += 7;
    out[idx..idx + 5].copy_from_slice(&target_mode); idx += 5;
    out[idx] = target_count; idx += 1;
    out[idx..idx + 3].copy_from_slice(&target_stat); idx += 3;
    out[idx..idx + 2].copy_from_slice(&target_order); idx += 2;
    out[idx] = target_pos_index; idx += 1;

    // max_triggers (1)
    out[idx] = match ability.max_triggers {
        Some(n) => norm(n as f32, MAX_TRIGGERS_LIMIT),
        None => 0.0,
    };
    idx += 1;

    // num_conditions (1)
    out[idx] = norm(ability.conditions.len() as f32, MAX_CONDITIONS_PER_ABILITY as f32);
    idx += 1;

    // conditions (MAX_CONDITIONS_PER_ABILITY × BATTLE_MATCHER_FEATURES)
    for ci in 0..MAX_CONDITIONS_PER_ABILITY {
        if let Some(condition) = ability.conditions.get(ci) {
            let c_start = idx + ci * BATTLE_MATCHER_FEATURES;
            // Flatten Condition::Is / Condition::AnyOf into first matcher
            match condition {
                Condition::Is(matcher) => {
                    encode_battle_matcher(matcher, &mut out[c_start..c_start + BATTLE_MATCHER_FEATURES]);
                }
                Condition::AnyOf(matchers) => {
                    // Encode first matcher (primary condition)
                    if let Some(m) = matchers.first() {
                        encode_battle_matcher(m, &mut out[c_start..c_start + BATTLE_MATCHER_FEATURES]);
                    }
                }
            }
        }
    }

    debug_assert_eq!(idx + BATTLE_CONDITION_FEATURES, BATTLE_ABILITY_FEATURES);
}

// ── Shop Ability Encoding ──

fn encode_shop_ability(
    ability: &ShopAbility,
    card_pool: &std::collections::BTreeMap<CardId, UnitCard>,
    out: &mut [f32],
) {
    let mut idx = 0;

    out[idx] = 1.0; idx += 1; // has_ability

    // trigger one-hot (6)
    let mut trigger = [0.0f32; 6];
    trigger[shop_trigger_index(&ability.trigger)] = 1.0;
    out[idx..idx + 6].copy_from_slice(&trigger); idx += 6;

    let mut effect_type = [0.0f32; 4];
    let mut buff_atk = 0.0f32;
    let mut buff_hp = 0.0f32;
    let mut spawn_card_id = 0.0f32;
    let mut spawn_card_atk = 0.0f32;
    let mut spawn_card_hp = 0.0f32;
    let mut spawn_location = [0.0f32; 3];
    let mut gain_mana = 0.0f32;
    let mut target_scope = [0.0f32; 5];
    let mut target_mode = [0.0f32; 4];
    let mut target_count = 0.0f32;
    let mut target_stat = [0.0f32; 3];
    let mut target_order = [0.0f32; 2];
    let mut target_pos_index = 0.0f32;

    match &ability.effect {
        ShopEffect::ModifyStatsPermanent { health, attack, target } => {
            effect_type[0] = 1.0;
            buff_atk = norm(*attack as f32, MAX_EFFECT_VALUE);
            buff_hp = norm(*health as f32, MAX_EFFECT_VALUE);
            encode_shop_target_fields(target, &mut target_scope, &mut target_mode, &mut target_count, &mut target_stat, &mut target_order, &mut target_pos_index);
        }
        ShopEffect::SpawnUnit { card_id, spawn_location: loc } => {
            effect_type[1] = 1.0;
            spawn_card_id = norm(card_id.0 as f32, MAX_CARD_ID);
            if let Some(card) = card_pool.get(card_id) {
                spawn_card_atk = norm(card.stats.attack as f32, MAX_ATTACK);
                spawn_card_hp = norm(card.stats.health as f32, MAX_HEALTH);
            }
            match loc {
                SpawnLocation::Front => spawn_location[0] = 1.0,
                SpawnLocation::Back => spawn_location[1] = 1.0,
                SpawnLocation::DeathPosition => spawn_location[2] = 1.0,
            }
        }
        ShopEffect::Destroy { target } => {
            effect_type[2] = 1.0;
            encode_shop_target_fields(target, &mut target_scope, &mut target_mode, &mut target_count, &mut target_stat, &mut target_order, &mut target_pos_index);
        }
        ShopEffect::GainMana { amount } => {
            effect_type[3] = 1.0;
            gain_mana = norm(*amount as f32, MAX_EFFECT_VALUE);
        }
    }

    out[idx..idx + 4].copy_from_slice(&effect_type); idx += 4;
    out[idx] = buff_atk; idx += 1;
    out[idx] = buff_hp; idx += 1;
    out[idx] = spawn_card_id; idx += 1;
    out[idx] = spawn_card_atk; idx += 1;
    out[idx] = spawn_card_hp; idx += 1;
    out[idx..idx + 3].copy_from_slice(&spawn_location); idx += 3;
    out[idx] = gain_mana; idx += 1;
    out[idx..idx + 5].copy_from_slice(&target_scope); idx += 5;
    out[idx..idx + 4].copy_from_slice(&target_mode); idx += 4;
    out[idx] = target_count; idx += 1;
    out[idx..idx + 3].copy_from_slice(&target_stat); idx += 3;
    out[idx..idx + 2].copy_from_slice(&target_order); idx += 2;
    out[idx] = target_pos_index; idx += 1;

    // max_triggers (1)
    out[idx] = match ability.max_triggers {
        Some(n) => norm(n as f32, MAX_TRIGGERS_LIMIT),
        None => 0.0,
    };
    idx += 1;

    // num_conditions (1)
    out[idx] = norm(ability.conditions.len() as f32, MAX_CONDITIONS_PER_ABILITY as f32);
    idx += 1;

    // conditions
    for ci in 0..MAX_CONDITIONS_PER_ABILITY {
        if let Some(condition) = ability.conditions.get(ci) {
            let c_start = idx + ci * SHOP_MATCHER_FEATURES;
            match condition {
                ShopCondition::Is(matcher) => {
                    encode_shop_matcher(matcher, &mut out[c_start..c_start + SHOP_MATCHER_FEATURES]);
                }
                ShopCondition::AnyOf(matchers) => {
                    if let Some(m) = matchers.first() {
                        encode_shop_matcher(m, &mut out[c_start..c_start + SHOP_MATCHER_FEATURES]);
                    }
                }
            }
        }
    }

    debug_assert_eq!(idx + SHOP_CONDITION_FEATURES, SHOP_ABILITY_FEATURES);
}

// ── Target encoding helpers ──

fn encode_battle_target_fields(
    target: &AbilityTarget,
    scope: &mut [f32; 7],
    mode: &mut [f32; 5],
    count: &mut f32,
    stat: &mut [f32; 3],
    order: &mut [f32; 2],
    pos_index: &mut f32,
) {
    match target {
        AbilityTarget::Position { scope: s, index: i } => {
            scope[scope_index(s)] = 1.0;
            mode[0] = 1.0;
            *pos_index = norm(*i as f32, MAX_POSITION_INDEX);
        }
        AbilityTarget::Adjacent { scope: s } => {
            scope[scope_index(s)] = 1.0;
            mode[1] = 1.0;
        }
        AbilityTarget::Random { scope: s, count: c } => {
            scope[scope_index(s)] = 1.0;
            mode[2] = 1.0;
            *count = norm(*c as f32, MAX_TARGET_COUNT);
        }
        AbilityTarget::Standard { scope: s, stat: st, order: o, count: c } => {
            scope[scope_index(s)] = 1.0;
            mode[3] = 1.0;
            *count = norm(*c as f32, MAX_TARGET_COUNT);
            stat[stat_type_index(st)] = 1.0;
            order[sort_order_index(o)] = 1.0;
        }
        AbilityTarget::All { scope: s } => {
            scope[scope_index(s)] = 1.0;
            mode[4] = 1.0;
        }
    }
}

fn encode_shop_target_fields(
    target: &ShopTarget,
    scope: &mut [f32; 5],
    mode: &mut [f32; 4],
    count: &mut f32,
    stat: &mut [f32; 3],
    order: &mut [f32; 2],
    pos_index: &mut f32,
) {
    match target {
        ShopTarget::Position { scope: s, index: i } => {
            scope[shop_scope_index(s)] = 1.0;
            mode[0] = 1.0;
            *pos_index = norm(*i as f32, MAX_POSITION_INDEX);
        }
        ShopTarget::Random { scope: s, count: c } => {
            scope[shop_scope_index(s)] = 1.0;
            mode[1] = 1.0;
            *count = norm(*c as f32, MAX_TARGET_COUNT);
        }
        ShopTarget::Standard { scope: s, stat: st, order: o, count: c } => {
            scope[shop_scope_index(s)] = 1.0;
            mode[2] = 1.0;
            *count = norm(*c as f32, MAX_TARGET_COUNT);
            stat[stat_type_index(st)] = 1.0;
            order[sort_order_index(o)] = 1.0;
        }
        ShopTarget::All { scope: s } => {
            scope[shop_scope_index(s)] = 1.0;
            mode[3] = 1.0;
        }
    }
}

// ── Index helpers ──

fn scope_index(scope: &TargetScope) -> usize {
    match scope {
        TargetScope::SelfUnit => 0,
        TargetScope::Allies => 1,
        TargetScope::Enemies => 2,
        TargetScope::All => 3,
        TargetScope::AlliesOther => 4,
        TargetScope::TriggerSource => 5,
        TargetScope::Aggressor => 6,
    }
}

fn shop_scope_index(scope: &ShopScope) -> usize {
    match scope {
        ShopScope::SelfUnit => 0,
        ShopScope::Allies => 1,
        ShopScope::All => 2,
        ShopScope::AlliesOther => 3,
        ShopScope::TriggerSource => 4,
    }
}

fn shop_trigger_index(trigger: &ShopTrigger) -> usize {
    match trigger {
        ShopTrigger::OnBuy => 0,
        ShopTrigger::OnSell => 1,
        ShopTrigger::OnShopStart => 2,
        ShopTrigger::AfterLoss => 3,
        ShopTrigger::AfterWin => 4,
        ShopTrigger::AfterDraw => 5,
    }
}

fn stat_type_index(stat: &StatType) -> usize {
    match stat {
        StatType::Health => 0,
        StatType::Attack => 1,
        StatType::Mana => 2,
    }
}

fn sort_order_index(order: &SortOrder) -> usize {
    match order {
        SortOrder::Ascending => 0,
        SortOrder::Descending => 1,
    }
}

fn compare_op_index(op: &CompareOp) -> usize {
    match op {
        CompareOp::GreaterThan => 0,
        CompareOp::LessThan => 1,
        CompareOp::Equal => 2,
        CompareOp::GreaterThanOrEqual => 3,
        CompareOp::LessThanOrEqual => 4,
    }
}

// ── Full Card Ability Encoding ──

fn encode_card_abilities(
    card: &UnitCard,
    card_pool: &std::collections::BTreeMap<CardId, UnitCard>,
    out: &mut [f32],
) {
    debug_assert!(out.len() >= ABILITY_FEATURES);
    let mut idx = 0;

    for i in 0..MAX_BATTLE_ABILITIES {
        if let Some(ability) = card.battle_abilities.get(i) {
            encode_battle_ability(ability, card_pool, &mut out[idx..idx + BATTLE_ABILITY_FEATURES]);
        }
        idx += BATTLE_ABILITY_FEATURES;
    }

    for i in 0..MAX_SHOP_ABILITIES {
        if let Some(ability) = card.shop_abilities.get(i) {
            encode_shop_ability(ability, card_pool, &mut out[idx..idx + SHOP_ABILITY_FEATURES]);
        }
        idx += SHOP_ABILITY_FEATURES;
    }
}

// ── Full Observation Encoding ──

pub fn encode_observation(
    shadow_hand: &[Option<CardId>],
    shadow_board: &[Option<BoardUnit>],
    shadow_mana: i32,
    state: &oab_game::GameState,
) -> Vec<f32> {
    let mut obs = vec![0.0f32; OBS_DIM];
    let pool = &state.shop.card_pool;
    let mut idx = 0;

    // Hand cards
    for i in 0..HAND_SIZE {
        if let Some(Some(cid)) = shadow_hand.get(i) {
            if let Some(card) = pool.get(cid) {
                obs[idx + 0] = 1.0;
                obs[idx + 1] = norm(cid.0 as f32, MAX_CARD_ID);
                obs[idx + 2] = norm(card.stats.attack as f32, MAX_ATTACK);
                obs[idx + 3] = norm(card.stats.health as f32, MAX_HEALTH);
                obs[idx + 4] = norm(card.economy.play_cost as f32, MAX_COST);
                obs[idx + 5] = norm(card.economy.burn_value as f32, MAX_BURN);
                obs[idx + 6] = if shadow_mana >= card.economy.play_cost { 1.0 } else { 0.0 };
                encode_card_abilities(card, pool, &mut obs[idx + HAND_BASE..idx + HAND_FEATURES]);
            }
        }
        idx += HAND_FEATURES;
    }

    // Board units
    for i in 0..BOARD_SIZE {
        if let Some(Some(bu)) = shadow_board.get(i) {
            if let Some(card) = pool.get(&bu.card_id) {
                obs[idx + 0] = 1.0;
                obs[idx + 1] = norm(bu.card_id.0 as f32, MAX_CARD_ID);
                obs[idx + 2] = norm((card.stats.attack + bu.perm_attack) as f32, MAX_ATTACK);
                obs[idx + 3] = norm((card.stats.health + bu.perm_health) as f32, MAX_HEALTH);
                obs[idx + 4] = norm(card.economy.play_cost as f32, MAX_COST);
                obs[idx + 5] = norm(card.economy.burn_value as f32, MAX_BURN);
                obs[idx + 6] = norm(bu.perm_attack as f32, MAX_ATTACK);
                obs[idx + 7] = norm(bu.perm_health as f32, MAX_HEALTH);
                encode_card_abilities(card, pool, &mut obs[idx + BOARD_BASE..idx + BOARD_FEATURES]);
            }
        }
        idx += BOARD_FEATURES;
    }

    // Scalars
    obs[idx + 0] = norm(shadow_mana as f32, MAX_MANA);
    obs[idx + 1] = norm(state.shop.mana_limit as f32, MAX_MANA);
    obs[idx + 2] = norm(state.shop.round as f32, MAX_ROUND);
    obs[idx + 3] = norm(state.lives as f32, MAX_LIVES);
    obs[idx + 4] = norm(state.wins as f32, MAX_WINS);
    obs[idx + 5] = norm(state.bag.len() as f32, MAX_BAG);
    idx += SCALAR_FEATURES;

    // Bag: full card data per unique card type INCLUDING abilities
    let mut bag_counts: std::collections::BTreeMap<CardId, u32> = std::collections::BTreeMap::new();
    for card_id in &state.bag {
        *bag_counts.entry(*card_id).or_insert(0) += 1;
    }
    let total_bag = state.bag.len().max(1) as f32;
    for (ci, (card_id, count)) in bag_counts.iter().enumerate() {
        if ci >= MAX_BAG_CARD_TYPES {
            break;
        }
        let base = idx + ci * BAG_CARD_FEATURES;
        obs[base + 0] = *count as f32 / total_bag;
        obs[base + 1] = norm(card_id.0 as f32, MAX_CARD_ID);
        if let Some(card) = pool.get(card_id) {
            obs[base + 2] = norm(card.stats.attack as f32, MAX_ATTACK);
            obs[base + 3] = norm(card.stats.health as f32, MAX_HEALTH);
            obs[base + 4] = norm(card.economy.play_cost as f32, MAX_COST);
            obs[base + 5] = norm(card.economy.burn_value as f32, MAX_BURN);
            encode_card_abilities(card, pool, &mut obs[base + BAG_CARD_BASE..base + BAG_CARD_FEATURES]);
        }
    }

    obs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_battle_ability_features() {
        // 1 + 13 + 6 + 1+1+1 + 1+1+1+3 + 1 + 7+5+1 + 3+2+1 + 1 + 1 + conditions
        let base = 1 + 13 + 6 + 3 + 6 + 1 + 13 + 3 + 2 + 1 + 1 + 1;
        assert_eq!(base, 51);
        assert_eq!(BATTLE_ABILITY_FEATURES, base - 1 + 1 + BATTLE_CONDITION_FEATURES);
    }

    #[test]
    fn test_shop_ability_features() {
        let base = 1 + 6 + 4 + 2 + 6 + 1 + 5 + 4 + 1 + 3 + 2 + 1 + 1 + 1;
        assert_eq!(base, 38);
        assert_eq!(SHOP_ABILITY_FEATURES, base - 1 + 1 + SHOP_CONDITION_FEATURES);
    }

    #[test]
    fn test_obs_dim_computed_correctly() {
        let expected = HAND_SIZE * HAND_FEATURES
            + BOARD_SIZE * BOARD_FEATURES
            + SCALAR_FEATURES
            + BAG_FEATURES;
        assert_eq!(OBS_DIM, expected);
        // Bag includes full abilities now
        assert_eq!(BAG_CARD_FEATURES, BAG_CARD_BASE + ABILITY_FEATURES);
    }
}
