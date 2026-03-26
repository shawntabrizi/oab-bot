.PHONY: setup build-engine build-server train evaluate play-local play-chain test clean help

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Setup ──

setup: $(VENV)/.setup-done ## Install Python deps + build Rust engine
	@echo "Setup complete."

$(VENV)/.setup-done: requirements.txt
	@test -d $(VENV) || python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt
	$(MAKE) build-engine
	@touch $@

build-engine: ## Build PyO3 bindings (oab_py)
	@echo "Building oab_py engine..."
	cd oab-py && ../$(VENV)/bin/maturin develop --release
	@echo "oab_py ready."

build-server: ## Build the HTTP game server (release)
	cd server && cargo build --release

# ── Three main paths ──

train: setup ## Train the RL agent           (ARGS="--timesteps 1000000")
	$(PYTHON) train.py $(ARGS)

play-local: setup build-server ## Play on localhost server     (ARGS="--games 5")
	./run_local.sh $(ARGS)

play-chain: setup build-server ## Play on hosted blockchain    (ARGS="--games 5")
	./run.sh $(ARGS)

# ── Other ──

evaluate: setup ## Evaluate a trained model     (ARGS="--num-games 100")
	$(PYTHON) evaluate.py $(ARGS)

test: setup ## Run Python tests
	$(VENV)/bin/pytest tests/ -v

clean: ## Remove build artifacts
	rm -rf oab-py/target server/target $(VENV)/.setup-done
