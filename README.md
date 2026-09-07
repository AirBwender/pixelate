# Pixelate

SDXL LoRA fine-tuning project for anime pixel-art generation.

## Goal

Fine-tune Stable Diffusion XL using LoRA to generate high-quality
anime-style pixel art, with a particular focus on anime-girl sprites and faces.

## Dataset

- Source: `nullHawk/anime-pixel-art-2`
- Training subset: 2,000 images
- Selection seed: 42

## Training

- Base model: Stable Diffusion XL 1.0
- Method: LoRA
- Resolution: 256×256
- Batch size: 1
- Planned training steps: 20,000
- LoRA rank: 32
- Learning rate: 1e-4
- Precision: FP16
- Training environment: Kaggle

## Structure

- `preprocessing/` — Dataset preparation
- `training.py` — LoRA training
- `inference/` — Model inference
- `postprocessing/` — Pixel-art postprocessing
- `notebooks/` — Kaggle notebooks