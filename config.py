"""Training configuration for OAB RL agent.

Provides a dataclass for all tunable parameters and JSON load/save,
so each model can be reproduced from its saved config.
"""

import dataclasses
import json
from pathlib import Path


@dataclasses.dataclass
class TrainConfig:
    """All tunable parameters for training."""

    # PPO hyperparameters
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.02

    # Reward shaping
    action_cost: float = -0.01
    repeat_penalty: float = -0.1
    play_reward: float = 0.08
    reorder_penalty: float = -0.05
    board_unit_reward: float = 0.03
    empty_board_penalty: float = -0.2
    wasteful_burn_penalty: float = -0.03

    # Self-play
    lobby_size: int = 10
    max_boards_per_bucket: int = 50
    challenge_probability: float = 0.35
    seed_games_per_model: int = 1000
    seed_opponent_models: list[str] = dataclasses.field(default_factory=list)
    seed_script_bots: bool = True

    # Optional phase decomposition
    phase_decomposition: bool = False
    shop_action_limit: int = 10
    position_action_limit: int = 5
    max_rounds: int = 20

    # Training run
    timesteps: int = 500_000
    set_id: int = 0
    save_path: str = "models/oab_agent"
    log_dir: str = "logs"


def load_config(path=None):
    """Load config from a JSON file, merging over defaults.

    Missing keys use defaults. Extra keys are ignored.
    Returns TrainConfig with defaults if path is None.
    """
    if path is None:
        return TrainConfig()

    with open(path) as f:
        data = json.load(f)

    field_names = {field.name for field in dataclasses.fields(TrainConfig)}
    filtered = {k: v for k, v in data.items() if k in field_names}
    return TrainConfig(**filtered)


def save_config(config, path):
    """Save config as JSON alongside a trained model."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(dataclasses.asdict(config), f, indent=2)
        f.write("\n")


def load_saved_model_config(model_path):
    """Load the config saved next to a trained model, if present."""
    config_path = Path(f"{model_path}_config.json")
    if config_path.exists():
        return load_config(str(config_path))
    return TrainConfig(save_path=model_path)
