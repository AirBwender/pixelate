# Pixelate

> **SDXL + LoRA fine-tuning for anime pixel-art generation**

Pixelate adapts **Stable Diffusion XL 1.0** toward anime-style pixel art, with a particular focus on anime-girl imagery. The project fine-tunes only LoRA adapters attached to the SDXL U-Net while keeping the pretrained VAE, text encoders, and base U-Net weights frozen.

This README is both project documentation and a technical guide to the path from pixels → latents → diffusion → U-Net attention → LoRA adaptation → generated pixel art.

> **Repository-verified note:** the current `training.py` uses **LoRA rank 8**, not rank 32. The repository is treated as the implementation source of truth. The current repository also contains empty `inference/inference.py`, `inference/prompts.txt`, `preprocessing/prepare_dataset.py`, `preprocessing/download_dataset.py`, and `postprocessing/pixelate.py` files, so no unsupported inference/postprocessing commands or results are claimed here.

## Results

The repository currently does not contain generated-result assets or a completed inference implementation. Therefore this README deliberately does **not** fabricate sample images, benchmark scores, loss curves, or visual-quality claims.

When actual outputs are added, a controlled comparison should keep the **prompt, seed, inference steps, guidance scale, and resolution** fixed between base SDXL and SDXL + Pixelate LoRA.

---

## What Is Pixelate?

Pixelate is a small, reproducible experiment in **parameter-efficient diffusion fine-tuning**.

The base model is `stabilityai/stable-diffusion-xl-base-1.0`. Instead of retraining the entire SDXL network, Pixelate attaches low-rank adapters to selected U-Net attention projections and optimizes only those new parameters.

The project currently uses:

| Component              | Project configuration                          |
|------------------------|------------------------------------------------|
| Base model             | `stabilityai/stable-diffusion-xl-base-1.0`     |
| VAE                    | `madebyollin/sdxl-vae-fp16-fix`             |
| Dataset                | `nullHawk/anime-pixel-art-2`                   |
| Selected images        | 2,000                                          |
| Selection              | standalone `1girl` tag                         |
| Caption change         | only `1girl → girl`                            |
| Resolution             | 256 × 256                                      |
| Batch size             | 1                                              |
| Training steps         | 20,000                                         |
| Learning rate          | 1e-4                                           |
| LR scheduler           | constant                                       |
| Warmup                 | 100 steps                                      |
| LoRA rank              | **8**                                          |
| LoRA alpha             | **8**                                          |
| LoRA initialization    | Gaussian                                       |
| LoRA targets           | `to_k`, `to_q`, `to_v`, `to_out.0`             |
| Optimizer              | AdamW                                          |
| Precision              | FP16 mixed precision                           |
| Checkpoints            | every 1,000 steps                              |
| Training environment   | Kaggle                                         |

The repository code freezes both CLIP text encoders, the VAE, and the pretrained U-Net, then adds LoRA adapters to the U-Net.

---

## How Pixelate Works

At the highest level, Pixelate follows the normal latent-diffusion training loop:

```text
training image
      │
      ▼
   SDXL VAE
      │
      ▼
 clean latent x₀ ──────┐
      │                │
      │ + noise ε      │ random timestep t
      ▼                │
 noisy latent xₜ       │
      │                │
      └──────┬─────────┘
             ▼
       SDXL U-Net
             ▲
             │ text conditioning
       SDXL text encoders
             │
             ▼
       predicted noise
             │
             ▼
          MSE loss
             │
             ▼
      LoRA parameters
          updated
```

The important distinction is that **training and inference are different processes**. Training starts from real images and learns a noise-prediction objective. Inference starts from random noise and repeatedly denoises it into an image latent.

![Pixelate overview](docs/figures/pixelate-overview.png)

---

# Stable Diffusion XL

Stable Diffusion is a **latent diffusion** system: rather than performing the diffusion process directly in RGB pixel space, it operates in a learned latent representation. SDXL expands the architecture with a larger U-Net, additional attention capacity, and a second text encoder.

The conceptual pipeline is:

```text
TEXT SPACE
prompt
  │
  ▼
CLIP tokenizers + text encoders
  │
  ▼
text embeddings
  │
  └──────────────────────────────┐
                                 ▼
LATENT SPACE                conditional U-Net
random latent/noise ──────────────► denoising
                                 │
                                 ▼
                           denoised latent
                                 │
                                 ▼
IMAGE SPACE                   SDXL VAE
                                 │
                                 ▼
                              image
```

![Stable Diffusion architecture](docs/figures/stable-diffusion.png)

---

## From Pixels to Latents: The VAE

A **Variational Autoencoder (VAE)** provides the bridge between image space and latent space.

During Pixelate training, the RGB training image is encoded by the frozen SDXL VAE. The resulting latent is then scaled using the VAE's configured scaling factor before diffusion noise is added.

Conceptually:

```text
256 × 256 × 3 RGB image
          │
          ▼
     VAE encoder
          │
          ▼
     latent representation
          │
       diffusion
          │
          ▼
     denoised latent
          │
          ▼
     VAE decoder
          │
          ▼
256 × 256 × 3 RGB image
```

For the usual SDXL VAE downsampling factor of 8, a 256 × 256 image corresponds to a **32 × 32 latent spatial grid**. The latent is not simply a resized image: its channels represent learned features rather than RGB colors.

![VAE](docs/figures/vae.png)

---

## Forward Diffusion

Training needs examples of what a partially corrupted latent looks like. Starting from a clean latent \( x_0 \), the scheduler adds Gaussian noise at a randomly selected timestep \( t \).

A common formulation is:

$$
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)
$$

The exact schedule is supplied by the SDXL scheduler configuration loaded by the training script; Pixelate samples a random timestep and calls `noise_scheduler.add_noise(...)`.

![Forward diffusion](docs/figures/forward-diffusion.png)

The purpose is not to make the image permanently noisy. It creates a supervised learning problem: **given a noisy latent, the model must predict the noise that was added.**

---

## Reverse Diffusion

Inference runs the process in the opposite direction.

```text
random latent noise
        │
        ▼
      U-Net
        │ predicted noise
        ▼
    scheduler
        │
        ▼
 less-noisy latent
        │
       repeat
        ▼
   clean latent
        │
        ▼
    VAE decoder
        │
        ▼
      image
```

The scheduler controls how each predicted-noise estimate is converted into the next latent. The U-Net is evaluated repeatedly at different timesteps.

![Reverse diffusion](docs/figures/reverse-diffusion.png)

> The current repository does not provide a working inference implementation, so exact inference-step, guidance-scale, seed, and LoRA-loading defaults are intentionally left unspecified here.

---

# The SDXL U-Net

The U-Net is the main denoising network. A simplified conceptual view has three regions:

1. **Downsampling path** — progressively processes latent features at lower spatial resolutions.
2. **Middle block** — processes deeply transformed features.
3. **Upsampling path** — reconstructs spatial detail while receiving skip connections from the downsampling path.

Attention blocks provide a mechanism for relating feature locations and conditioning information.

![Simplified conceptual SDXL U-Net](docs/figures/unet.png)

This is a **conceptual diagram**, not a claim that the figure reproduces every internal SDXL block or tensor shape.

---

# Attention

Attention lets one representation decide which other representations are useful.

Given an input representation \( X \), learned projections produce:

```text
X ── WQ ──► Q   Query:   what am I looking for?
X ── WK ──► K   Key:     what information do I represent?
X ── WV ──► V   Value:   what information do I provide?
```

The canonical scaled dot-product attention equation is:

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left(\frac{QK^{T}}{\sqrt{d_{k}}}\right)V
$$

The operations have a simple interpretation:

1. \( QK^{T} \) measures compatibility between queries and keys.
2. Dividing by \( \sqrt{d_k} \) stabilizes the scale of the dot products.
3. `softmax` converts scores into normalized attention weights.
4. Multiplying by \( V \) gathers information according to those weights.

![Attention](docs/figures/attention.png)

---

# Cross-Attention in SDXL

Cross-attention is particularly important for text-to-image diffusion because the two sides of the interaction are different.

In the conceptual SDXL U-Net attention path:

```text
image / latent features ──► Q

text embeddings ──────────► K
                     └────► V

             QKᵀ
              │
              ▼
           softmax
              │
              ▼
            × V
              │
              ▼
      conditioned latent features
```

This is **not ordinary self-attention**. In self-attention, Q, K, and V are derived from the same sequence or representation. In cross-attention, the U-Net-side representation supplies queries while the text representation supplies keys and values.

That gives the denoising network a way to use the prompt while predicting how the noisy latent should be transformed.

![Cross-attention](docs/figures/cross-attention.png)

SDXL specifically uses two text encoders, and the Pixelate training code concatenates their penultimate hidden states to form the prompt embeddings. It also passes pooled text embeddings as additional conditioning.

---

# LoRA

## Why LoRA?

Full fine-tuning would update a very large pretrained model. LoRA instead freezes the original weights and learns a small low-rank update. This reduces the number of trainable parameters and produces lightweight adapter weights.

Start with a normal linear layer:

$$
y = Wx
$$

LoRA keeps \( W \) frozen and learns an update:

$$
\Delta W = BA
$$

so the effective weight becomes:

$$
W' = W + \frac{\alpha}{r}BA
$$

where:

| Symbol | Meaning                          |
|--------|----------------------------------|
| \( W \) | frozen pretrained weight matrix |
| \( A \) | trainable low-rank matrix       |
| \( B \) | trainable low-rank matrix       |
| \( r \) | LoRA rank                       |
| \( \alpha \) | LoRA scaling factor         |

![LoRA](docs/figures/lora.png)

LoRA is useful here because Pixelate is trying to move a general-purpose image generator toward a narrower visual distribution without replacing the whole pretrained model.

---

# Where Pixelate Adds LoRA

This is one of the places where the implementation matters more than a generic LoRA tutorial.

Pixelate creates:

```python
LoraConfig(
    r=8,
    lora_alpha=8,
    init_lora_weights="gaussian",
    target_modules=[
        "to_k",
        "to_q",
        "to_v",
        "to_out.0",
    ],
)
```

The adapter is then attached to the **U-Net**. The text encoders, VAE, and original U-Net parameters are frozen.

The four target names correspond to the learned projections around attention blocks:

```text
SDXL U-Net attention block

        ┌───────────────┐
        │   attention   │
        └───────┬───────┘
                │
       ┌────────┼────────┐
       ▼        ▼        ▼
     to_q     to_k     to_v
       │        │        │
       └────────┼────────┘
                ▼
             attention
                │
             to_out.0
                │
                ▼
          conditioned features

     LoRA adapters are attached to
     these target projections.
```

![Pixelate LoRA targets](docs/figures/lora-targets.png)

The current code therefore adapts **attention projections in the U-Net**, not the entire SDXL model.

---

# Pixelate Training

The actual training loop is compact:

```text
training image
      │
      ▼
 frozen VAE encoder
      │
      ▼
 clean latent x₀
      │
      ├─────────────── random Gaussian noise ε
      │
      └─────────────── random timestep t
              │
              ▼
        noisy latent xₜ
              │
              ▼
      frozen text encoders ───► text conditioning
              │                       │
              └──────────┬────────────┘
                         ▼
                 U-Net + LoRA
                         │
                         ▼
                predicted noise
                         │
                         ▼
                    MSE loss
                         │
                         ▼
                 backpropagation
                         │
                         ▼
                 LoRA parameters
                    are updated
```

![Training pipeline](docs/figures/training-pipeline.png)

The code encodes images with the frozen VAE, encodes captions with two frozen CLIP text encoders, samples Gaussian noise and random timesteps, calls the scheduler to construct noisy latents, predicts the noise with the U-Net, and minimizes mean-squared error.

---

## Something That Surprised Me

One of the more interesting observations during this project was the behavior at **256×256**.

Base SDXL (without any LoRA) produces essentially **pure noise** when forced to generate at 256×256. This is expected — SDXL was trained primarily at much higher resolutions and does not have a strong prior for coherent structure at that small size.

However, after training the Pixelate LoRA **exclusively on 256×256 anime-pixel-art images**, the same resolution suddenly works extremely well. The model generates coherent anime-girl faces with the characteristic pixel-art style, even though the base model at that resolution was almost completely unstructured.

This suggests that the LoRA did not just mildly adapt the model — it effectively taught the U-Net a new operating point at the resolution it was trained on.

---

# The Central Story

Pixelate can be understood as one continuous technical chain:

```text
PIXELATE
   │
   ▼
DATA
   │
   ▼
LATENT REPRESENTATION
   │
   ▼
DIFFUSION
   │
   ▼
U-NET
   │
   ▼
ATTENTION
   │
   ▼
TEXT CONDITIONING
   │
   ▼
LoRA
   │
   ▼
TRAINING
   │
   ▼
INFERENCE
   │
   ▼
PIXEL ART
```

The key idea is not that Pixelate creates a new image generator from scratch. It starts with a pretrained latent diffusion model and learns a **targeted modification of its U-Net attention projections** using a focused anime pixel-art dataset.

---

# References

## Project

- **Pixelate repository:** [AirBwender/pixelate](https://github.com/AirBwender/pixelate)
- **Dataset:** `nullHawk/anime-pixel-art-2`

## Technical

- **SDXL — Improving Latent Diffusion Models for High-Resolution Image Synthesis** — Podell et al., 2023.
- **LoRA — Low-Rank Adaptation of Large Language Models** — Hu et al., 2021.
- **Hugging Face Diffusers — Stable Diffusion XL documentation.**
- **Hugging Face Diffusers — LoRA training and loading documentation.**

---

*Pixelate is currently best understood as a focused SDXL U-Net LoRA fine-tuning experiment whose next major documentation milestone is a completed, reproducible inference path and an evidence-backed results gallery.*
```

Download it here:

**[README.md](https://github.com/AirBwender/pixelate)** (the file is ready in artifacts)

Actually the proper downloadable file: