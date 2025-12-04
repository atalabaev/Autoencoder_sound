from __future__ import annotations

from typing import Tuple, Dict

import torch
from torch import nn


class AutoencoderStage1(nn.Module):
    """
    First-stage autoencoder for audio.

    Input: flattened waveform (num_samples)
    Latent: z1
    """

    def __init__(self, input_dim: int, latent_dim: int = 256) -> None:
        super().__init__()
        self.input_dim = input_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, input_dim),
            # без Sigmoid, т.к. аудио может быть в диапазоне [-1, 1]
            nn.Tanh(),  # предполагаем нормировку к [-1, 1]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z1 = self.encode(x)
        x_recon = self.decode(z1)
        return z1, x_recon


class AutoencoderStage2(nn.Module):
    """
    Second-stage autoencoder:
    Input: z1
    Latent: z2
    Output: z1_recon
    """

    def __init__(self, input_dim: int = 256, latent_dim: int = 64) -> None:
        super().__init__()
        self.input_dim = input_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, input_dim),
        )

    def encode(self, z1: torch.Tensor) -> torch.Tensor:
        return self.encoder(z1)

    def decode(self, z2: torch.Tensor) -> torch.Tensor:
        return self.decoder(z2)

    def forward(self, z1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z2 = self.encode(z1)
        z1_recon = self.decode(z2)
        return z2, z1_recon


class TwoStageAudioAutoencoder(nn.Module):
    """
    Полный двуступенчатый автоэнкодер для аудио:

        x -> AE1.encoder -> z1
        z1 -> AE2.encoder -> z2
        z2 -> AE2.decoder -> z1_hat
        z1_hat -> AE1.decoder -> x_hat_stage2

    Параллельно:
        z1 -> AE1.decoder -> x_hat_stage1
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim_1: int,
        latent_dim_2: int,
    ) -> None:
        super().__init__()
        self.stage1 = AutoencoderStage1(
            input_dim=input_dim, latent_dim=latent_dim_1
        )
        self.stage2 = AutoencoderStage2(
            input_dim=latent_dim_1, latent_dim=latent_dim_2
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Stage 1
        z1, x_recon_stage1 = self.stage1(x)

        # Stage 2
        z2, z1_recon = self.stage2(z1)

        # Decode reconstructed z1
        x_recon_stage2 = self.stage1.decode(z1_recon)

        return {
            "z1": z1,
            "z2": z2,
            "x_recon_stage1": x_recon_stage1,
            "x_recon_stage2": x_recon_stage2,
            "z1_recon": z1_recon,
        }
