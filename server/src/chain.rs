//! Contract game backend.
//!
//! Talks to the OAB PolkaVM smart contract over Ethereum JSON-RPC. Uses
//! `eth_sendTransaction` with a node-managed dev account for signing
//! (mirrors what the local revive-dev-node provides). Cards and sets are
//! baked in via `oab_assets` — we never read them from the chain.

#[cfg(feature = "chain")]
mod inner {
    use std::collections::BTreeMap;
    use std::path::PathBuf;
    use std::thread::sleep;
    use std::time::Duration;

    use parity_scale_codec::{Decode, Encode};
    use serde::Deserialize;
    use serde_json::{json, Value};
    use tiny_keccak::{Hasher, Keccak};

    use oab_battle::battle::{resolve_battle, BattleResult, CombatUnit};
    use oab_battle::rng::XorShiftRng;
    use oab_battle::types::*;
    use oab_game::view::GameView;
    use oab_game::{sealed, GamePhase, GameState, LocalGameState};

    use crate::types::{BattleReport, GameStateResponse, SetCardEntry, SetInfo, StepResponse};

    /// keccak256("BattleReported(uint8,uint8,uint8,uint8,uint64,bytes)")
    const BATTLE_REPORTED_TOPIC: &str =
        "0x96fd1736ea4fbef32e328d7005021b05c7ee31f32694ddef23dd55af68e089bd";

    // ── Public session ────────────────────────────────────────────────────

    /// Active arena session backed by the OAB PolkaVM contract.
    #[allow(dead_code)]
    pub struct ChainGameSession {
        rpc_url: String,
        contract_address: String,
        from_address: String,
        chain_id: u64,
        card_pool: BTreeMap<CardId, UnitCard>,
        state: Option<GameState>,
        set_id: SetIdValue,
    }

    impl ChainGameSession {
        /// Connect to a node and bind to a deployed OAB contract.
        ///
        /// `contract_address` falls back to `deps/open-auto-battler/contract/deployment.json`.
        /// `from_address` falls back to the first dev account on the node.
        pub fn new(
            rpc_url: &str,
            contract_address: Option<&str>,
            from_address: Option<&str>,
            set_id: SetIdValue,
        ) -> Result<Self, String> {
            let contract_address = match contract_address {
                Some(addr) => addr.to_string(),
                None => find_deployment_address()?,
            };

            let chain_id = rpc::eth_chain_id(rpc_url)?;
            let from_address = match from_address {
                Some(addr) => addr.to_string(),
                None => rpc::first_dev_account(rpc_url)?,
            };

            eprintln!("Connected to {} (chain id {})", rpc_url, chain_id);
            eprintln!("Contract: {}", contract_address);
            eprintln!("From:     {}", from_address);

            let card_pool = oab_assets::cards::build_pool();

            let mut session = Self {
                rpc_url: rpc_url.to_string(),
                contract_address,
                from_address,
                chain_id,
                card_pool,
                state: None,
                set_id,
            };

            session.sync_state_from_chain()?;
            if session.state.is_some() {
                let s = session.state.as_ref().unwrap();
                eprintln!(
                    "Resuming game (round={}, wins={}, lives={}).",
                    s.round, s.wins, s.lives
                );
            }

            Ok(session)
        }

        /// Reset the session: end/abandon any prior game and start a fresh one.
        ///
        /// `seed` is unused — the contract derives its own seed from caller +
        /// `seed_nonce`. We still accept it so the public interface mirrors
        /// the local backend's `reset(seed, set_id)`.
        pub fn reset(
            &mut self,
            seed: u64,
            set_id: Option<SetIdValue>,
        ) -> Result<GameStateResponse, String> {
            let set_id = set_id.unwrap_or(self.set_id);

            self.sync_state_from_chain()?;

            if let Some(state) = &self.state {
                if state.phase == GamePhase::Completed {
                    eprintln!("Ending completed game...");
                    let _ = self.send_call(&abi::encode_end_game());
                } else {
                    eprintln!("Abandoning active game...");
                    let _ = self.send_call(&abi::encode_abandon_game());
                }
                self.sync_state_from_chain()?;
            }

            if self.state.is_some() {
                return Err("Failed to clear active game on-chain. Try again.".into());
            }

            eprintln!("Starting new game (set_id={})...", set_id);
            let nonce = seed; // Use the seed as the nonce; contract mixes with caller.
            self.send_call(&abi::encode_start_game(set_id, nonce))?;

            self.set_id = set_id;
            self.sync_state_from_chain()?;

            match &self.state {
                Some(state) => {
                    eprintln!("Game started (round={}).", state.round);
                    Ok(self.get_state())
                }
                None => Err("Game not found after start_game".into()),
            }
        }

        /// Submit a turn (shop actions + battle), parse the BattleReported
        /// event, and replay the battle locally for the report.
        pub fn step(&mut self, action: &CommitTurnAction) -> Result<StepResponse, String> {
            let state = self
                .state
                .as_ref()
                .ok_or("No active game. Call POST /reset first.")?;

            if state.phase == GamePhase::Completed {
                return Err("Game is already over. Call POST /reset to start a new game.".into());
            }
            if state.phase != GamePhase::Shop {
                return Err(format!("Wrong phase: {:?}", state.phase));
            }

            // Local verification gives us the post-shop player board for replay.
            let mut verified_state = state.clone();
            oab_battle::commit::verify_and_apply_turn(&mut verified_state, action)
                .map_err(|e| format!("{:?}", e))?;

            let player_units: Vec<CombatUnit> = verified_state
                .board
                .iter()
                .filter_map(|slot| {
                    let u = slot.as_ref()?;
                    let card = self.card_pool.get(&u.card_id)?;
                    let mut cu = CombatUnit::from_card(card.clone());
                    cu.attack_buff = u.perm_attack;
                    cu.health_buff = u.perm_health;
                    cu.health = cu.health.saturating_add(u.perm_health).max(0);
                    Some(cu)
                })
                .collect();

            let prev_wins = state.wins;
            let prev_lives = state.lives;
            let completed_round = state.round;
            let board_size = state.config.board_size as usize;

            eprintln!("Submitting turn (round {})...", completed_round);
            let action_bytes = action.encode();
            let receipt = self.send_call_for_receipt(&abi::encode_submit_turn(&action_bytes))?;

            let battle_event = parse_battle_reported(&receipt)?;

            let battle_report = if let Some(event) = battle_event {
                let enemy_units: Vec<CombatUnit> = event
                    .ghost
                    .iter()
                    .filter_map(|gu| {
                        let card = self.card_pool.get(&gu.card_id)?;
                        let mut cu = CombatUnit::from_card(card.clone());
                        cu.attack_buff = gu.perm_attack;
                        cu.health_buff = gu.perm_health;
                        cu.health = cu.health.saturating_add(gu.perm_health).max(0);
                        Some(cu)
                    })
                    .collect();
                let enemy_count = enemy_units.len();

                let mut rng = XorShiftRng::seed_from_u64(event.battle_seed);
                let events = resolve_battle(
                    player_units,
                    enemy_units,
                    &mut rng,
                    &self.card_pool,
                    board_size,
                );

                BattleReport {
                    player_units_survived: 0,
                    enemy_units_faced: enemy_count,
                    events,
                }
            } else {
                BattleReport {
                    player_units_survived: 0,
                    enemy_units_faced: 0,
                    events: vec![],
                }
            };

            self.sync_state_from_chain()?;

            let state = self
                .state
                .as_ref()
                .ok_or("Game disappeared after submit_turn")?;

            let battle_result = if state.wins > prev_wins {
                BattleResult::Victory
            } else if state.lives < prev_lives {
                BattleResult::Defeat
            } else {
                BattleResult::Draw
            };

            let game_over = state.phase == GamePhase::Completed;
            let game_result = if game_over {
                let result = if state.wins >= state.config.wins_to_victory {
                    "victory"
                } else {
                    "defeat"
                };

                eprintln!("Game over ({}). Submitting end_game...", result);
                let _ = self.send_call(&abi::encode_end_game());

                Some(result.to_string())
            } else {
                None
            };

            let reward = match &battle_result {
                BattleResult::Victory => 1,
                BattleResult::Defeat => -1,
                BattleResult::Draw => 0,
            };

            let battle_result_str = match &battle_result {
                BattleResult::Victory => "Victory",
                BattleResult::Defeat => "Defeat",
                BattleResult::Draw => "Draw",
            };

            let player_units_survived = state.board.iter().filter(|s| s.is_some()).count();

            eprintln!(
                "Round {}: {} (survived={}, enemies={}, events={})",
                completed_round,
                battle_result_str,
                player_units_survived,
                battle_report.enemy_units_faced,
                battle_report.events.len()
            );

            Ok(StepResponse {
                completed_round,
                battle_result: battle_result_str.to_string(),
                game_over,
                game_result,
                reward,
                battle_report: BattleReport {
                    player_units_survived,
                    ..battle_report
                },
                state: self.get_state(),
            })
        }

        pub fn get_state(&self) -> GameStateResponse {
            match &self.state {
                Some(state) => {
                    let hand_used = vec![false; state.hand.len()];
                    let view = GameView::from_state(state, state.shop_mana, &hand_used, false);
                    view.into()
                }
                None => GameStateResponse {
                    round: 0,
                    lives: 0,
                    wins: 0,
                    mana: 0,
                    mana_limit: 0,
                    phase: "none".to_string(),
                    bag_count: 0,
                    hand: vec![],
                    board: vec![],
                    can_afford: vec![],
                    bag: vec![],
                },
            }
        }

        #[allow(dead_code)]
        pub fn get_cards(&self) -> Vec<oab_game::view::CardView> {
            self.card_pool
                .values()
                .map(oab_game::view::CardView::from)
                .collect()
        }

        #[allow(dead_code)]
        pub fn get_sets(&self) -> Vec<SetInfo> {
            let metas = oab_assets::sets::get_all_metas();
            let all_sets = oab_assets::sets::get_all();
            metas
                .into_iter()
                .zip(all_sets.into_iter())
                .map(|(meta, set)| SetInfo {
                    id: meta.id,
                    name: meta.name.to_string(),
                    card_count: set.cards.len(),
                    cards: set
                        .cards
                        .iter()
                        .map(|e| SetCardEntry {
                            card_id: e.card_id.0,
                            rarity: e.rarity,
                        })
                        .collect(),
                })
                .collect()
        }

        // ── Private helpers ──────────────────────────────────────────────

        fn sync_state_from_chain(&mut self) -> Result<(), String> {
            let raw = self.eth_call(&abi::encode_get_game_state())?;
            let bytes = match abi::decode_dynamic_bytes(&raw) {
                Some(b) if !b.is_empty() && !is_missing_session(&b) => b,
                _ => {
                    self.state = None;
                    return Ok(());
                }
            };

            let session = ContractSession::decode(&mut &bytes[..])
                .map_err(|e| format!("Failed to decode ArenaSession: {}", e))?;

            let local = LocalGameState {
                bag: session.bag,
                hand: session.hand,
                board: session.board,
                mana_limit: session.mana_limit,
                shop_mana: session.shop_mana,
                round: session.round,
                lives: session.lives,
                wins: session.wins,
                phase: session.phase,
                next_card_id: session.next_card_id,
                game_seed: session.game_seed,
            };

            self.state = Some(GameState::reconstruct(
                self.card_pool.clone(),
                session.set_id,
                sealed::default_config(),
                local,
            ));
            Ok(())
        }

        fn eth_call(&self, data: &str) -> Result<String, String> {
            rpc::eth_call(
                &self.rpc_url,
                &self.from_address,
                &self.contract_address,
                data,
            )
        }

        /// Send a state-changing call and wait for the receipt's success.
        fn send_call(&self, data: &str) -> Result<rpc::Receipt, String> {
            rpc::eth_send_transaction(
                &self.rpc_url,
                &self.from_address,
                &self.contract_address,
                data,
            )
        }

        fn send_call_for_receipt(&self, data: &str) -> Result<rpc::Receipt, String> {
            self.send_call(data)
        }
    }

    // ── ArenaSession mirror (matches contract field order) ──────────────

    #[derive(Debug, Decode)]
    struct ContractSession {
        bag: Vec<CardId>,
        hand: Vec<CardId>,
        board: Vec<Option<BoardUnit>>,
        mana_limit: ManaValue,
        shop_mana: ManaValue,
        round: RoundValue,
        lives: RoundValue,
        wins: RoundValue,
        phase: GamePhase,
        next_card_id: u16,
        game_seed: u64,
        set_id: SetIdValue,
    }

    /// revive's `Mapping` getter materializes a zeroed value for missing keys
    /// instead of returning empty bytes. Treat all-zero payloads as "no game".
    fn is_missing_session(data: &[u8]) -> bool {
        data.iter().all(|b| *b == 0)
    }

    // ── BattleReported event parsing ─────────────────────────────────────

    struct BattleReportedEvent {
        battle_seed: u64,
        ghost: Vec<GhostBoardUnit>,
    }

    fn parse_battle_reported(
        receipt: &rpc::Receipt,
    ) -> Result<Option<BattleReportedEvent>, String> {
        for log in &receipt.logs {
            let topic_match = log
                .topics
                .first()
                .map(|t| t.eq_ignore_ascii_case(BATTLE_REPORTED_TOPIC))
                .unwrap_or(false);
            if !topic_match {
                continue;
            }

            let bytes = abi::hex_to_bytes(&log.data)?;
            if bytes.len() < 12 {
                return Err(format!(
                    "BattleReported event truncated: got {} bytes, expected >= 12",
                    bytes.len()
                ));
            }

            let battle_seed = u64::from_be_bytes(
                bytes[4..12]
                    .try_into()
                    .map_err(|_| "Failed to slice battleSeed bytes".to_string())?,
            );
            let ghost_bytes = &bytes[12..];
            let ghost = if ghost_bytes.is_empty() {
                Vec::new()
            } else {
                Vec::<GhostBoardUnit>::decode(&mut &ghost_bytes[..])
                    .map_err(|e| format!("Failed to decode ghost SCALE: {}", e))?
            };

            return Ok(Some(BattleReportedEvent { battle_seed, ghost }));
        }
        Ok(None)
    }

    // ── Deployment helpers ───────────────────────────────────────────────

    #[derive(Debug, Deserialize)]
    struct Deployment {
        address: String,
    }

    fn find_deployment_address() -> Result<String, String> {
        // Prefer the submodule path first.
        let candidates = [
            "deps/open-auto-battler/contract/deployment.json",
            "../open-auto-battler/contract/deployment.json",
        ];
        for rel in &candidates {
            let path = PathBuf::from(rel);
            if path.exists() {
                let text = std::fs::read_to_string(&path)
                    .map_err(|e| format!("Failed to read {}: {}", rel, e))?;
                let dep: Deployment = serde_json::from_str(&text)
                    .map_err(|e| format!("Failed to parse {}: {}", rel, e))?;
                return Ok(dep.address);
            }
        }
        Err(
            "Contract address not specified and deployment.json not found. \
             Pass --contract or run open-auto-battler's deploy script first."
                .into(),
        )
    }

    // ── ABI helpers ──────────────────────────────────────────────────────

    pub mod abi {
        use super::*;

        fn keccak256_bytes(data: &[u8]) -> [u8; 32] {
            let mut hasher = Keccak::v256();
            hasher.update(data);
            let mut out = [0u8; 32];
            hasher.finalize(&mut out);
            out
        }

        fn selector(signature: &str) -> String {
            let hash = keccak256_bytes(signature.as_bytes());
            hash[..4].iter().map(|b| format!("{:02x}", b)).collect()
        }

        fn pad_uint(value: u64) -> String {
            format!("{:064x}", value)
        }

        /// Encode a `bytes` parameter (length + data, padded to 32 bytes).
        fn pad_bytes(data: &[u8]) -> String {
            let len = pad_uint(data.len() as u64);
            let body: String = data.iter().map(|b| format!("{:02x}", b)).collect();
            let padded = ((body.len() + 63) / 64) * 64;
            format!("{}{:0<width$}", len, body, width = padded)
        }

        pub fn encode_start_game(set_id: SetIdValue, seed_nonce: u64) -> String {
            format!(
                "0x{}{}{}",
                selector("startGame(uint16,uint64)"),
                pad_uint(set_id as u64),
                pad_uint(seed_nonce)
            )
        }

        pub fn encode_submit_turn(action_scale: &[u8]) -> String {
            // Single dynamic `bytes` arg → offset is 0x20.
            format!(
                "0x{}{}{}",
                selector("submitTurn(bytes)"),
                pad_uint(0x20),
                pad_bytes(action_scale)
            )
        }

        pub fn encode_get_game_state() -> String {
            format!("0x{}", selector("getGameState()"))
        }

        pub fn encode_end_game() -> String {
            format!("0x{}", selector("endGame()"))
        }

        pub fn encode_abandon_game() -> String {
            format!("0x{}", selector("abandonGame()"))
        }

        /// Decode a Solidity `bytes` return value: 32-byte offset, 32-byte
        /// length, then the raw bytes (32-byte aligned).
        pub fn decode_dynamic_bytes(hex_data: &str) -> Option<Vec<u8>> {
            let raw = hex_data.strip_prefix("0x").unwrap_or(hex_data);
            if raw.is_empty() {
                return None;
            }
            // Need at least offset (64 hex) + length (64 hex) = 128 hex chars.
            if raw.len() < 128 {
                return None;
            }
            // Length is the second 32-byte word.
            let len_hex = &raw[64..128];
            let len = u64::from_str_radix(len_hex, 16).ok()? as usize;
            if len == 0 {
                return Some(Vec::new());
            }
            let body_start: usize = 128;
            let body_end = body_start.checked_add(len.checked_mul(2)?)?;
            if raw.len() < body_end {
                return None;
            }
            let body_hex = &raw[body_start..body_end];
            hex_to_bytes(&format!("0x{}", body_hex)).ok()
        }

        pub fn hex_to_bytes(hex: &str) -> Result<Vec<u8>, String> {
            let raw = hex.strip_prefix("0x").unwrap_or(hex);
            if raw.len() % 2 != 0 {
                return Err(format!("Odd hex length: {}", raw.len()));
            }
            (0..raw.len())
                .step_by(2)
                .map(|i| {
                    u8::from_str_radix(&raw[i..i + 2], 16)
                        .map_err(|e| format!("Invalid hex byte at {}: {}", i, e))
                })
                .collect()
        }
    }

    // ── JSON-RPC helpers ─────────────────────────────────────────────────

    pub mod rpc {
        use super::*;

        #[derive(Debug)]
        pub struct LogEntry {
            pub topics: Vec<String>,
            pub data: String,
        }

        #[derive(Debug)]
        pub struct Receipt {
            pub logs: Vec<LogEntry>,
        }

        fn post(url: &str, body: &Value) -> Result<Value, String> {
            let resp = ureq::post(url)
                .set("Content-Type", "application/json")
                .send_json(body)
                .map_err(|e| format!("HTTP error to {}: {}", url, e))?;
            let json: Value = resp
                .into_json()
                .map_err(|e| format!("Failed to decode JSON from {}: {}", url, e))?;
            if let Some(err) = json.get("error") {
                return Err(format!("RPC error: {}", err));
            }
            Ok(json)
        }

        pub fn eth_chain_id(url: &str) -> Result<u64, String> {
            let body = json!({"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1});
            let resp = post(url, &body)?;
            let result = resp["result"].as_str().ok_or("Missing chainId result")?;
            u64::from_str_radix(result.strip_prefix("0x").unwrap_or(result), 16)
                .map_err(|e| format!("Invalid chainId: {}", e))
        }

        pub fn first_dev_account(url: &str) -> Result<String, String> {
            let body = json!({"jsonrpc":"2.0","method":"eth_accounts","params":[],"id":1});
            let resp = post(url, &body)?;
            resp["result"]
                .as_array()
                .and_then(|a| a.first())
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .ok_or_else(|| {
                    "No dev accounts available. Pass --from or unlock an account on the node."
                        .to_string()
                })
        }

        pub fn eth_call(url: &str, from: &str, to: &str, data: &str) -> Result<String, String> {
            let body = json!({
                "jsonrpc":"2.0","method":"eth_call",
                "params":[{"from":from,"to":to,"data":data},"latest"],
                "id":1
            });
            let resp = post(url, &body)?;
            Ok(resp["result"].as_str().unwrap_or("0x").to_string())
        }

        pub fn eth_send_transaction(
            url: &str,
            from: &str,
            to: &str,
            data: &str,
        ) -> Result<Receipt, String> {
            let body = json!({
                "jsonrpc":"2.0","method":"eth_sendTransaction",
                "params":[{"from":from,"to":to,"data":data,"gas":"0x10000000"}],
                "id":1
            });
            let resp = post(url, &body)?;
            let tx_hash = resp["result"]
                .as_str()
                .ok_or("Missing tx hash")?
                .to_string();
            wait_for_receipt(url, &tx_hash)
        }

        fn wait_for_receipt(url: &str, tx_hash: &str) -> Result<Receipt, String> {
            for _ in 0..120 {
                let body = json!({
                    "jsonrpc":"2.0","method":"eth_getTransactionReceipt",
                    "params":[tx_hash], "id":1
                });
                let resp = post(url, &body)?;
                if let Some(result) = resp.get("result").filter(|v| !v.is_null()) {
                    let status = result
                        .get("status")
                        .and_then(|v| v.as_str())
                        .unwrap_or("0x0");
                    let logs = result
                        .get("logs")
                        .and_then(|v| v.as_array())
                        .map(|arr| {
                            arr.iter()
                                .map(|log| LogEntry {
                                    topics: log
                                        .get("topics")
                                        .and_then(|v| v.as_array())
                                        .map(|arr| {
                                            arr.iter()
                                                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                                .collect()
                                        })
                                        .unwrap_or_default(),
                                    data: log
                                        .get("data")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("0x")
                                        .to_string(),
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    if status != "0x1" {
                        return Err(format!("Transaction reverted: {}", tx_hash));
                    }
                    return Ok(Receipt { logs });
                }
                sleep(Duration::from_millis(250));
            }
            Err(format!("Timed out waiting for receipt: {}", tx_hash))
        }
    }
}

#[cfg(feature = "chain")]
pub use inner::ChainGameSession;
