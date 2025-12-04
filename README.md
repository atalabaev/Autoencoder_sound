# Two-Stage Audio Autoencoder

This project implements a **two-stage autoencoder for audio signal reconstruction** using PyTorch.  
The model compresses short audio signals into a latent space and reconstructs them in two sequential stages.

The project was developed as a **university machine learning coursework**.

---

 What Is a Two-Stage Autoencoder?

The architecture consists of two autoencoders:

1. **Stage 1**  
   Encodes and reconstructs the raw audio waveform.

2. **Stage 2**  
   Encodes and reconstructs the latent representation of the first autoencoder.

This allows deeper compression and more flexible representation of audio signals
Project Structure
two_stage_autoencoder_project/
├── main.py # Entry point (train / reconstruct)
├── train.py # Training and reconstruction logic
├── models.py # Neural network architectures
├── data.py # Dataset and DataLoader
├── utils.py # Utility functions
├── config.py # Training configuration
├── audio_data/
│ ├── train/ # Training WAV files
│ └── test/ # Test WAV files
├── requirements.txt
├── README.md
└── .gitignore

---

##
️ Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt


Training the Model
Basic training:
python3 main.py --mode train --epochs 10
High-quality training example:
python3 main.py --mode train \
  --epochs 50 \
  --latent-dim-1 512 \
  --latent-dim-2 128 \
  --max-duration-sec 2.0 \
  --batch-size 64
 Audio Reconstruction
After training, you can reconstruct any WAV file:
python3 main.py \
  --mode reconstruct \
  --checkpoint checkpoints/checkpoint_epoch_50.pth \
  --input-wav audio_data/test/sample1.wav \
  --output-wav reconstructed.wav \
  --latent-dim-1 512 \
  --latent-dim-2 128 \
  --max-duration-sec 2.0
The reconstructed file will be saved as:
reconstructed.wav
 Loss Function
The total loss is the sum of two Mean Squared Errors:
L = MSE(stage1) + MSE(stage2)
Stage 1 compares original waveform and its reconstruction
Stage 2 compares original waveform and reconstruction from the second stage
 Technologies Used
Python 3.9+
PyTorch
TorchAudio
NumPy
SoundFile
