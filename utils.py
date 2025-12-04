from __future__ import annotations

import os

import torch

from config import TrainConfig


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg: TrainConfig,
) -> str:
    ensure_dir(cfg.output_dir)
    path = os.path.join(cfg.output_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg.__dict__,
        },
        path,
    )
    return path


def load_model_from_checkpoint(
    model: torch.nn.Module, checkpoint_path: str, device: str
) -> int:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    epoch = checkpoint.get("epoch", -1)
    return epoch
