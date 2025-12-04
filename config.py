from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class TrainConfig:
    """
    Training configuration for the two-stage audio autoencoder.
    """

    # Training hyperparameters
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    latent_dim_1: int = 256
    latent_dim_2: int = 64
    log_interval: int = 20

    # Audio settings
    sample_rate: int = 16000       # целевой sample rate
    max_duration_sec: float = 1.0  # длина аудио фрагмента в секундах
    audio_root: str = "audio_data" # корневая папка с аудио (train/test)

    # System / misc
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 2
    seed: int = 42
    output_dir: str = "checkpoints"

    def __str__(self) -> str:
        fields = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"TrainConfig({fields})"
