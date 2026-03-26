# oab-bot

RL agent for [Open Auto Battler](https://github.com/shawntabrizi/open-auto-battler). Trains via self-play using MaskablePPO, then evaluates per-card balance statistics.

## Setup

```bash
# Clone with submodule
git clone --recurse-submodules https://github.com/shawntabrizi/oab-bot.git
cd oab-bot

# If already cloned without submodules
git submodule update --init

# Python dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Build PyO3 bindings
cd pylib
maturin develop --release
cd ..
```

### Local development override

If you have a local checkout of open-auto-battler and want to use it instead of the submodule:

```bash
git config submodule.deps/open-auto-battler.url /path/to/your/open-auto-battler
```

## Usage

```bash
# Train
python train.py
python train.py --config experiment.json --timesteps 5000000

# Evaluate
python evaluate.py --model-path models/oab_agent --num-games 100
python evaluate.py --output results.json

# Play on-chain
./run.sh --games 5
```

## Tests

```bash
pytest tests/ -v
```
