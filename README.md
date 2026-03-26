# oab-bot

RL agent for [Open Auto Battler](https://github.com/shawntabrizi/open-auto-battler). Trains via self-play using MaskablePPO, then evaluates per-card balance statistics.

## Setup

```bash
git clone --recurse-submodules https://github.com/shawntabrizi/oab-bot.git
cd oab-bot
make setup    # creates venv, installs deps, builds Rust engine
```

If you have a local checkout of open-auto-battler:
```bash
git config submodule.deps/open-auto-battler.url /path/to/your/open-auto-battler
```

## Usage

```bash
make train                                  # train with defaults (500K steps)
make train ARGS="--config experiment.json"  # train with custom config
make train ARGS="--timesteps 5000000"       # train longer

make play-local ARGS="--games 5"            # test on local server (no blockchain)
make play-chain ARGS="--games 5"            # play on hosted blockchain

make evaluate ARGS="--num-games 100"        # evaluate model + card stats
make evaluate ARGS="--output results.json"  # export balance data

make test                                   # run tests
```

Or use the scripts directly:

```bash
python train.py --config experiment.json
./run_local.sh --games 5 --model models/oab_v4_5m
./run.sh --games 5 --rpc wss://custom-rpc.example.com --key //Alice
python evaluate.py --num-games 200 --output results.csv
```

Run `make help` for all available commands.
