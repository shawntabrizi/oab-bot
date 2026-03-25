#!/usr/bin/env bash
# Run the trained OAB bot on-chain.
#
# Usage:
#   ./run.sh                                          # defaults: //Alice, 1 game
#   ./run.sh --games 5                                # play 5 games
#   ./run.sh --key //Bob --games 10                   # play as Bob
#   ./run.sh --rpc wss://other-node.example.com       # custom RPC
#   ./run.sh --model models/oab_agent_5m --games 3    # different model
#   ./run.sh --set 1                                  # use card set 1
#   ./run.sh --quiet                                  # suppress per-round output

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER_DIR="$(cd "$SCRIPT_DIR/../open-auto-battler" && pwd)"
SERVER_BIN="$SERVER_DIR/target/debug/oab-server"

# ── Defaults ──
RPC_URL="wss://oab-rpc.shawntabrizi.com"
KEY="//Alice"
MODEL="models/oab_agent"
GAMES=1
SET_ID=0
PORT=3030
QUIET=""
EXTRA_ARGS=""

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
            echo "Usage: ./run.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --rpc <URL>      Chain RPC endpoint (default: wss://oab-rpc.shawntabrizi.com)"
            echo "  --key <SURI>     Signing key (default: //Alice)"
            echo "  --model <PATH>   Model path (default: models/oab_agent)"
            echo "  --games <N>      Number of games (default: 1)"
            echo "  --set <N>        Card set ID (default: 0)"
            echo "  --port <N>       Local server port (default: 3030)"
            echo "  --quiet          Suppress per-round output"
            echo "  --help           Show this help"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

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
    echo "Building oab-server..."
    (cd "$SERVER_DIR" && cargo build -p oab-server)
fi

# ── Fund account ──
echo "Funding account ($KEY)..."
"$SERVER_BIN" --url "$RPC_URL" --key "$KEY" --fund "$KEY" 2>&1 | grep -v "^$"

# ── Start server ──
SERVER_LOG="$SCRIPT_DIR/.server.log"
echo ""
echo "Starting server (chain mode, port $PORT)..."
"$SERVER_BIN" --url "$RPC_URL" --key "$KEY" --port "$PORT" --set "$SET_ID" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for server..."
for i in $(seq 1 30); do
    if curl -s "http://localhost:$PORT/state" >/dev/null 2>&1; then
        echo "Server ready (log: $SERVER_LOG)."
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
    --games "$GAMES" \
    $QUIET
