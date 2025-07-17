"""Utility functions for production ML."""
import numpy as np
import torch
import random
import json
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")

def save_metrics(metrics: Dict[str, float], path: str) -> None:
    """Save evaluation metrics to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {path}")

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_device() -> torch.device:
    """Get the best available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def format_number(n: int) -> str:
    """Format large numbers with K/M/B suffixes."""
    for unit in ['', 'K', 'M', 'B']:
        if abs(n) < 1000:
            return f"{n:.1f}{unit}"
        n /= 1000
    return f"{n:.1f}T"

# Update Faker-based synthetic data generator [2025-06-13T14:21:17]

# WIP: debugging mode collapse in generator [2025-06-16T17:52:18]

# Implement domain adaptation for transfer [2025-06-18T20:25:32]

# Implement domain adaptation for transfer [2025-06-20T15:28:31]

# Update Faker-based synthetic data generator [2025-06-25T14:23:40]

# Add evaluation with FID score metric [2025-07-01T19:23:10]

# Implement domain adaptation for transfer [2025-07-02T14:48:49]

# Implement domain adaptation for transfer [2025-07-06T16:13:57]

# Implement tabular GAN for synthetic records [2025-07-09T20:36:41]

# Implement tabular GAN for synthetic records [2025-07-10T19:32:12]

# Update documentation for generation API [2025-07-14T20:01:31]

# WIP: training larger models on cloud GPUs [2025-07-14T16:57:23]

# Add evaluation with FID score metric [2025-07-17T14:42:44]
