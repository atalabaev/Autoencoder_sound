from __future__ import annotations

import argparse

from config import TrainConfig
from train import run_training, reconstruct_audio
print(">>> MAIN.PY STARTED")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-stage audio autoencoder")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "reconstruct"],
        default="train",
        help="train: обучение модели; reconstruct: прогнать один файл через автоэнкодер",
    )

    # Общие параметры
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim-1", type=int, default=256)
    parser.add_argument("--latent-dim-2", type=int, default=64)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-duration-sec", type=float, default=1.0)
    parser.add_argument("--audio-root", type=str, default="audio_data")

    # Для reconstruct
    parser.add_argument("--checkpoint", type=str, help="path to .pth checkpoint")
    parser.add_argument("--input-wav", type=str, help="input wav path")
    parser.add_argument("--output-wav", type=str, default="reconstructed.wav")

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim_1=args.latent_dim_1,
        latent_dim_2=args.latent_dim_2,
        log_interval=args.log_interval,
        output_dir=args.output_dir,
        seed=args.seed,
        sample_rate=args.sample_rate,
        max_duration_sec=args.max_duration_sec,
        audio_root=args.audio_root,
        device=TrainConfig().device,
    )


def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    print("=== Parsed config ===")
    print(cfg)

    if args.mode == "train":
        print(">>> Starting TRAIN mode")
        run_training(cfg)
    elif args.mode == "reconstruct":
        print(">>> Starting RECONSTRUCT mode")
        reconstruct_audio(args, cfg)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
