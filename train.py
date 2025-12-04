from __future__ import annotations

from typing import Dict

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchaudio

from config import TrainConfig
from data import get_dataloaders
from models import TwoStageAudioAutoencoder
from utils import set_seed, save_checkpoint, load_model_from_checkpoint


def train_epoch(
    model: TwoStageAudioAutoencoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    log_interval: int = 20,
    alpha_stage2: float = 1.0,
) -> float:
    model.train()
    mse = nn.MSELoss()
    running_loss = 0.0

    for batch_idx, (waveform, _) in enumerate(loader):
        waveform = waveform.to(device)
        batch_size, channels, num_samples = waveform.shape
        x = waveform.view(batch_size, -1)

        optimizer.zero_grad()
        outputs: Dict[str, torch.Tensor] = model(x)

        loss_stage1 = mse(outputs["x_recon_stage1"], x)
        loss_stage2 = mse(outputs["x_recon_stage2"], x)
        loss = loss_stage1 + alpha_stage2 * loss_stage2

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % log_interval == 0:
            print(
                f"Train step [{batch_idx + 1}/{len(loader)}] "
                f"loss={loss.item():.6f} "
                f"(stage1={loss_stage1.item():.6f}, stage2={loss_stage2.item():.6f})"
            )

    return running_loss / len(loader)


def eval_epoch(
    model: TwoStageAudioAutoencoder,
    loader: DataLoader,
    device: str,
    alpha_stage2: float = 1.0,
) -> float:
    model.eval()
    mse = nn.MSELoss()
    total_loss = 0.0

    with torch.no_grad():
        for waveform, _ in loader:
            waveform = waveform.to(device)
            batch_size, channels, num_samples = waveform.shape
            x = waveform.view(batch_size, -1)

            outputs = model(x)

            loss_stage1 = mse(outputs["x_recon_stage1"], x)
            loss_stage2 = mse(outputs["x_recon_stage2"], x)
            loss = loss_stage1 + alpha_stage2 * loss_stage2
            total_loss += loss.item()

    return total_loss / len(loader)


def run_training(cfg: TrainConfig) -> None:
    print("Config:", cfg)
    set_seed(cfg.seed)

    train_loader, test_loader = get_dataloaders(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        sample_rate=cfg.sample_rate,
        max_duration_sec=cfg.max_duration_sec,
        audio_root=cfg.audio_root,
    )

    num_samples = int(cfg.sample_rate * cfg.max_duration_sec)
    model = TwoStageAudioAutoencoder(
        input_dim=num_samples,
        latent_dim_1=cfg.latent_dim_1,
        latent_dim_2=cfg.latent_dim_2,
    ).to(cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val_loss = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        print(f"\n=== Epoch {epoch}/{cfg.epochs} ===")
        train_loss = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=cfg.device,
            log_interval=cfg.log_interval,
        )

        val_loss = eval_epoch(
            model=model,
            loader=test_loader,
            device=cfg.device,
        )

        print(
            f"Epoch {epoch} summary: "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            path = save_checkpoint(model, optimizer, epoch, cfg)
            print(f"New best model saved to: {path}")


def preprocess_single_audio(
    path: str,
    sample_rate: int,
    max_duration_sec: float,
    device: str,
) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)

    num_samples = int(sample_rate * max_duration_sec)
    if waveform.shape[1] < num_samples:
        pad_size = num_samples - waveform.shape[1]
        waveform = torch.nn.functional.pad(
            waveform, (0, pad_size), mode="constant", value=0.0
        )
    elif waveform.shape[1] > num_samples:
        waveform = waveform[:, :num_samples]

    waveform = waveform.to(device)
    return waveform


def reconstruct_audio(args, cfg: TrainConfig) -> None:
    if not args.checkpoint or not os.path.isfile(args.checkpoint):
        raise FileNotFoundError("You must provide a valid --checkpoint path")

    if not args.input_wav or not os.path.isfile(args.input_wav):
        raise FileNotFoundError("You must provide a valid --input-wav file")

    num_samples = int(cfg.sample_rate * cfg.max_duration_sec)
    model = TwoStageAudioAutoencoder(
        input_dim=num_samples,
        latent_dim_1=cfg.latent_dim_1,
        latent_dim_2=cfg.latent_dim_2,
    )

    last_epoch = load_model_from_checkpoint(model, args.checkpoint, cfg.device)
    print(f"Loaded checkpoint from epoch {last_epoch}")

    model.eval()

    waveform = preprocess_single_audio(
        path=args.input_wav,
        sample_rate=cfg.sample_rate,
        max_duration_sec=cfg.max_duration_sec,
        device=cfg.device,
    )

    with torch.no_grad():
        x = waveform.view(1, -1)
        outputs = model(x)
        x_recon = outputs["x_recon_stage2"]
        x_recon = x_recon.view(1, -1).cpu()

    torchaudio.save(
        args.output_wav,
        x_recon,
        sample_rate=cfg.sample_rate,
    )
    print(f"Reconstructed audio saved to {args.output_wav}")
