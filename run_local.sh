#!/usr/bin/env bash
# Run the trained OAB bot against a local blockchain node.
#
# Requires a local node running at ws://127.0.0.1:9944
#
# Usage:
#   ./run_local.sh                          # defaults: //Alice, 1 game
#   ./run_local.sh --games 5                # play 5 games
#   ./run_local.sh --model models/oab_v4_5m # different model
#   ./run_local.sh --set 1                  # use card set 1
#   ./run_local.sh --quiet                  # suppress per-round output

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER_DIR="$SCRIPT_DIR/server"
SERVER_BIN="$SERVER_DIR/target/release/oab-server"

# ── Defaults ──
RPC_URL="ws://127.0.0.1:9944"
KEY="//Alice"
MODEL="models/oab_agent"
GAMES=1
SET_ID=0
PORT=3030
QUIET=""

# ── Parse args ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --rpc)      RPC_URL="$2"; shift 2 ;;
        --key)      KEY="$2"; shift 2 ;;
        --model)    MODEL="$2"; shift 2 ;;
        --games)    GAMES="$2"; shift 2 ;;
        --set)      SET_ID="$2"; shift 2 ;;
        --port)     PORT="$2"; shift 2 ;;
        --quiet)    QUIET="--quiet"; shift ;;
        --help|-h)
            echo "Usage: ./run_local.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --rpc <URL>      Local node RPC endpoint (default: ws://127.0.0.1:9944)"
            echo "  --key <SURI>     Signing key (default: //Alice)"
            echo "  --model <PATH>   Model path (default: models/oab_agent)"
            echo "  --games <N>      Number of games (default: 1)"
            echo "  --set <N>        Card set ID (default: 0)"
            echo "  --port <N>       Server port (default: 3030)"
            echo "  --quiet          Suppress per-round output"
            echo "  --help           Show this help"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Validate model exists ──
if [[ ! -f "$SCRIPT_DIR/${MODEL}.zip" ]]; then
    echo "Error: Model not found at ${MODEL}.zip"
    echo "Train one first: make train"
    exit 1
fi

# ── Cleanup on exit ──
SERVER_PID=""
cleanup() {
    if [[ -n "$SERVER_PID" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ── Build server if needed ──
if [[ ! -f "$SERVER_BIN" ]]; then
    echo "Building oab-server (release)..."
    (cd "$SERVER_DIR" && cargo build --release)
fi

# ── Fund account ──
echo "Funding account ($KEY)..."
"$SERVER_BIN" --url "$RPC_URL" --key "$KEY" --fund "$KEY" 2>&1 | grep -v "^$"

# ── Start server ──
SERVER_LOG="$SCRIPT_DIR/.server.log"
echo ""
echo "Starting server (local chain, port $PORT)..."
"$SERVER_BIN" --url "$RPC_URL" --key "$KEY" --port "$PORT" --set "$SET_ID" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for server..."
for i in $(seq 1 30); do
    if curl -s "http://localhost:$PORT/cards" >/dev/null 2>&1; then
        echo "Server ready."
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Error: Server exited unexpectedly. Log:"
        cat "$SERVER_LOG"
        exit 1
    fi
    sleep 1
done

# ── Run bot ──
echo ""
source "$SCRIPT_DIR/.venv/bin/activate"
python "$SCRIPT_DIR/play.py" \
    --url "http://localhost:$PORT" \
    --model "$MODEL" \
    --set "$SET_ID" \
    --games "$GAMES" \
    $QUIET
