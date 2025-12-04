from __future__ import annotations

import glob
import os
from typing import Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio

torchaudio.set_audio_backend("soundfile")



class AudioFolderDataset(Dataset):
    """
    Простая датасет-обёртка: обходит все .wav (или .flac) файлы в указанной папке,
    приводит их к фиксированному sample_rate и длине.

    Каждой записи сопоставляется фиктивный label=0 (нам он не нужен для автоэнкодера).
    """

    def __init__(
        self,
        root: str,
        split: str,
        sample_rate: int,
        max_duration_sec: float,
    ) -> None:
        """
        :param root: корневая папка с подкаталогами train/test
        :param split: 'train' или 'test'
        :param sample_rate: целевой sample rate
        :param max_duration_sec: длина фрагмента в секундах
        """
        assert split in ("train", "test"), "split must be 'train' or 'test'"
        self.sample_rate = sample_rate
        self.max_duration_sec = max_duration_sec

        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Directory not found: {split_dir}")

        self.file_paths: List[str] = []
        for ext in ("wav", "flac", "mp3"):
            self.file_paths.extend(
                glob.glob(os.path.join(split_dir, f"**/*.{ext}"), recursive=True)
            )

        if len(self.file_paths) == 0:
            raise RuntimeError(f"No audio files found in {split_dir}")

        self.num_samples = int(self.sample_rate * self.max_duration_sec)
        self._resamplers = {}  # cache resamplers by original sr

    def __len__(self) -> int:
        return len(self.file_paths)

    def _get_resampler(self, orig_sr: int) -> torchaudio.transforms.Resample:
        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = torchaudio.transforms.Resample(
                orig_freq=orig_sr,
                new_freq=self.sample_rate,
            )
        return self._resamplers[orig_sr]

    def _load_and_preprocess(self, path: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(path)  # (channels, samples)

        # mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # resample if needed
        if sr != self.sample_rate:
            resampler = self._get_resampler(sr)
            waveform = resampler(waveform)

        # pad or crop
        if waveform.shape[1] < self.num_samples:
            pad_size = self.num_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(
                waveform, (0, pad_size), mode="constant", value=0.0
            )
        elif waveform.shape[1] > self.num_samples:
            waveform = waveform[:, : self.num_samples]

        # теперь waveform: (1, num_samples)
        return waveform

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.file_paths[idx]
        waveform = self._load_and_preprocess(path)
        return waveform, 0  # label не нужен, но DataLoader требует 2 значения


def get_dataloaders(
    batch_size: int,
    num_workers: int,
    sample_rate: int,
    max_duration_sec: float,
    audio_root: str,
) -> Tuple[DataLoader, DataLoader]:
    """
    Создаёт train/test DataLoader из папки с аудиофайлами.
    """
    train_dataset = AudioFolderDataset(
        root=audio_root,
        split="train",
        sample_rate=sample_rate,
        max_duration_sec=max_duration_sec,
    )

    test_dataset = AudioFolderDataset(
        root=audio_root,
        split="test",
        sample_rate=sample_rate,
        max_duration_sec=max_duration_sec,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader
