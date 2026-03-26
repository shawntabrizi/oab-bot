//! Action table and masking for the RL agent.
//!
//! 66 discrete actions mapped to game TurnActions.
//! Action masking computed directly from game state — no JSON.

use oab_battle::types::*;

// Hand and board sizes are fixed by the game rules (5 each).
// These match GameConfig defaults and the action table layout.
const HAND_SIZE: usize = 5;
const BOARD_SIZE: usize = 5;

pub const NUM_ACTIONS: usize = 66;

/// Decode an action index (0-65) into a TurnAction.
/// Returns None for EndTurn (action 0) since that's not a shop action.
pub fn action_to_turn_action(index: u32) -> Option<TurnAction> {
    let i = index as usize;
    match i {
        0 => None, // EndTurn
        1..=5 => Some(TurnAction::BurnFromHand {
            hand_index: (i - 1) as u32,
        }),
        6..=30 => {
            let offset = i - 6;
            Some(TurnAction::PlayFromHand {
                hand_index: (offset / BOARD_SIZE) as u32,
                board_slot: (offset % BOARD_SIZE) as u32,
            })
        }
        31..=35 => Some(TurnAction::BurnFromBoard {
            board_slot: (i - 31) as u32,
        }),
        36..=45 => {
            // Unordered pairs: (0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)
            let pair_index = i - 36;
            let (a, b) = SWAP_PAIRS[pair_index];
            Some(TurnAction::SwapBoard {
                slot_a: a as u32,
                slot_b: b as u32,
            })
        }
        46..=65 => {
            // Ordered pairs excluding f==t
            let pair_index = i - 46;
            let (f, t) = MOVE_PAIRS[pair_index];
            Some(TurnAction::MoveBoard {
                from_slot: f as u32,
                to_slot: t as u32,
            })
        }
        _ => None,
    }
}

// Pre-computed pair tables
const SWAP_PAIRS: [(usize, usize); 10] = [
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 4),
    (3, 4),
];

const MOVE_PAIRS: [(usize, usize); 20] = [
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 0),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 0),
    (2, 1),
    (2, 3),
    (2, 4),
    (3, 0),
    (3, 1),
    (3, 2),
    (3, 4),
    (4, 0),
    (4, 1),
    (4, 2),
    (4, 3),
];

/// Compute the 66-element action mask from shadow state.
pub fn compute_action_mask(
    shadow_hand: &[Option<CardId>],
    shadow_board: &[Option<BoardUnit>],
    shadow_mana: i32,
    pending_count: usize,
    max_actions_per_turn: usize,
    card_pool: &std::collections::BTreeMap<CardId, UnitCard>,
) -> Vec<bool> {
    let mut mask = vec![false; NUM_ACTIONS];

    // If at action limit, only EndTurn is valid
    if pending_count >= max_actions_per_turn {
        mask[0] = true;
        return mask;
    }

    // EndTurn: always valid
    mask[0] = true;

    // BurnFromHand(0..4): actions 1-5
    for i in 0..HAND_SIZE {
        mask[1 + i] = shadow_hand.get(i).map_or(false, |c| c.is_some());
    }

    // PlayFromHand(hi, bs): actions 6-30
    for hi in 0..HAND_SIZE {
        if let Some(Some(card_id)) = shadow_hand.get(hi) {
            if let Some(card) = card_pool.get(card_id) {
                if shadow_mana >= card.economy.play_cost {
                    for bs in 0..BOARD_SIZE {
                        let board_empty =
                            shadow_board.get(bs).map_or(false, |s| s.is_none());
                        mask[6 + hi * BOARD_SIZE + bs] = board_empty;
                    }
                }
            }
        }
    }

    // BurnFromBoard(0..4): actions 31-35
    for bs in 0..BOARD_SIZE {
        mask[31 + bs] = shadow_board.get(bs).map_or(false, |s| s.is_some());
    }

    // SwapBoard: actions 36-45 (10 unordered pairs)
    for (pi, &(a, b)) in SWAP_PAIRS.iter().enumerate() {
        let a_some = shadow_board.get(a).map_or(false, |s| s.is_some());
        let b_some = shadow_board.get(b).map_or(false, |s| s.is_some());
        mask[36 + pi] = a_some && b_some;
    }

    // MoveBoard: actions 46-65 (20 ordered pairs)
    for (pi, &(f, t)) in MOVE_PAIRS.iter().enumerate() {
        let f_some = shadow_board.get(f).map_or(false, |s| s.is_some());
        let t_some = shadow_board.get(t).map_or(false, |s| s.is_some());
        mask[46 + pi] = f_some && t_some;
    }

    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_count() {
        // Verify: 1 + 5 + 25 + 5 + 10 + 20 = 66
        assert_eq!(NUM_ACTIONS, 66);
    }

    #[test]
    fn test_endturn_is_none() {
        assert!(action_to_turn_action(0).is_none());
    }

    #[test]
    fn test_burn_from_hand() {
        for i in 0..5u32 {
            let action = action_to_turn_action(1 + i).unwrap();
            assert!(matches!(action, TurnAction::BurnFromHand { hand_index } if hand_index == i));
        }
    }

    #[test]
    fn test_play_from_hand() {
        // Action 6 = PlayFromHand(0, 0)
        let action = action_to_turn_action(6).unwrap();
        assert!(
            matches!(action, TurnAction::PlayFromHand { hand_index: 0, board_slot: 0 })
        );
        // Action 11 = PlayFromHand(1, 0)
        let action = action_to_turn_action(11).unwrap();
        assert!(
            matches!(action, TurnAction::PlayFromHand { hand_index: 1, board_slot: 0 })
        );
    }

    #[test]
    fn test_swap_pairs() {
        assert_eq!(SWAP_PAIRS.len(), 10);
        // First pair should be (0,1)
        let action = action_to_turn_action(36).unwrap();
        assert!(matches!(action, TurnAction::SwapBoard { slot_a: 0, slot_b: 1 }));
    }

    #[test]
    fn test_move_pairs() {
        assert_eq!(MOVE_PAIRS.len(), 20);
        // First pair should be (0,1)
        let action = action_to_turn_action(46).unwrap();
        assert!(matches!(action, TurnAction::MoveBoard { from_slot: 0, to_slot: 1 }));
    }
}
