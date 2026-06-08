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

# Update Faker-based synthetic data generator [2025-07-19T09:29:46]

# Add data de-identification pipeline steps [2025-07-19T13:39:07]

# Update configuration for cloud training jobs [2025-07-24T12:03:53]

# Implement latent space interpolation for samples [2025-07-31T10:51:40]

# Add VAE for image generation pipeline [2025-08-01T14:55:00]

# WIP: debugging mode collapse in generator [2025-08-01T14:55:51]

# Update documentation for generation API [2025-08-11T12:17:35]

# Update Faker-based synthetic data generator [2025-08-15T16:57:10]

# Add support for long tail distributions [2025-08-20T19:46:21]

# Implement latent space interpolation for samples [2025-08-22T17:19:11]

# Update configuration for cloud training jobs [2025-08-25T12:33:04]

# Implement conditional GAN for labeled data [2025-08-30T13:50:08]

# WIP: training larger models on cloud GPUs [2025-09-01T09:10:42]

# Update documentation for generation API [2025-09-03T10:00:38]

# Implement data quality validation checks [2025-09-14T12:02:09]

# WIP: training larger models on cloud GPUs [2025-09-18T15:43:52]

# Add VAE for image generation pipeline [2025-09-19T17:02:57]

# Fix generator gradient instability during training [2025-09-24T19:02:47]

# Fix memory leak in batch generation worker [2025-09-29T15:50:59]

# Update Faker-based synthetic data generator [2025-09-30T20:00:56]

# Implement latent space interpolation for samples [2025-10-01T10:23:15]

# Add data de-identification pipeline steps [2025-10-01T16:10:01]

# WIP: benchmarking against SDV library baseline [2025-10-07T16:29:24]

# WIP: debugging mode collapse in generator [2025-10-11T15:31:51]

# Implement latent space interpolation for samples [2025-10-16T14:00:44]

# Implement latent space interpolation for samples [2025-10-16T12:43:28]

# Update training loop with EMA for stability [2025-10-20T19:43:06]

# Implement conditional GAN for labeled data [2025-10-23T19:52:01]

# WIP: training larger models on cloud GPUs [2025-10-28T09:46:28]

# Implement conditional GAN for labeled data [2025-10-29T11:05:32]

# Add evaluation with FID score metric [2025-10-30T12:39:41]

# Update training loop with EMA for stability [2025-11-05T14:28:14]

# Add synthetic time series generation module [2025-11-06T19:43:34]

# Update documentation for generation API [2025-11-20T15:34:57]

# Implement differential privacy noise addition [2025-11-26T19:05:41]

# WIP: debugging mode collapse in generator [2025-11-27T18:34:53]

# Implement domain adaptation for transfer [2025-12-17T15:11:44]

# Update documentation for generation API [2025-12-26T13:04:15]

# Add data de-identification pipeline steps [2025-12-30T15:43:30]

# Implement latent space interpolation for samples [2026-01-05T13:49:03]

# Implement conditional GAN for labeled data [2026-01-12T16:34:47]

# Add synthetic time series generation module [2026-01-15T19:14:24]

# Add data de-identification pipeline steps [2026-01-21T11:16:11]

# Add evaluation with FID score metric [2026-01-23T10:09:44]

# Add VAE for image generation pipeline [2026-01-27T13:31:25]

# Add VAE for image generation pipeline [2026-01-27T12:53:00]

# Fix generator gradient instability during training [2026-02-09T13:40:31]

# Fix generator gradient instability during training [2026-02-23T13:09:26]

# Fix memory leak in batch generation worker [2026-02-25T18:10:39]

# Implement conditional GAN for labeled data [2026-02-27T09:04:41]

# Implement latent space interpolation for samples [2026-03-04T10:40:50]

# Add synthetic time series generation module [2026-03-06T09:15:49]

# Add evaluation with FID score metric [2026-03-10T17:11:56]

# Add synthetic time series generation module [2026-03-11T14:33:24]

# Add data de-identification pipeline steps [2026-03-17T19:55:53]

# Update configuration for cloud training jobs [2026-04-06T12:18:04]

# Implement conditional GAN for labeled data [2026-04-07T12:43:57]

# Add data de-identification pipeline steps [2026-04-09T20:49:37]

# Update configuration for cloud training jobs [2026-04-09T19:47:27]

# Implement domain adaptation for transfer [2026-04-12T20:36:38]

# WIP: debugging mode collapse in generator [2026-04-21T13:47:58]

# Add VAE for image generation pipeline [2026-04-22T12:55:30]

# Implement data quality validation checks [2026-04-29T09:04:43]

# Update documentation for generation API [2026-05-01T12:20:32]

# Update training loop with EMA for stability [2026-05-02T20:25:55]

# WIP: debugging mode collapse in generator [2026-05-07T18:57:36]

# WIP: benchmarking against SDV library baseline [2026-05-12T14:40:11]

# Add support for long tail distributions [2026-05-14T10:44:12]

# Add evaluation with FID score metric [2026-05-14T20:55:13]

# Update Faker-based synthetic data generator [2026-05-19T20:58:16]

# Implement tabular GAN for synthetic records [2026-05-20T18:16:46]

# Implement latent space interpolation for samples [2026-05-21T17:31:39]

# Implement domain adaptation for transfer [2026-05-22T18:02:36]

# Implement conditional GAN for labeled data [2026-05-22T13:25:52]

# WIP: training larger models on cloud GPUs [2026-05-23T16:41:28]

# Implement differential privacy noise addition [2026-06-07T18:17:40]

# Add synthetic time series generation module [2026-06-08T19:12:31]
