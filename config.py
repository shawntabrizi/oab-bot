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

    # Reward shaping
    action_cost: float = -0.01
    repeat_penalty: float = -0.1

    # Self-play
    lobby_size: int = 10
    max_boards_per_bucket: int = 50

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
