# Pixelate

> **SDXL + LoRA fine-tuning for anime pixel-art generation**

Pixelate adapts **Stable Diffusion XL 1.0** toward anime-style pixel art, focusing on anime-girl imagery. Only LoRA adapters on the SDXL U-Net are trained — the VAE, text encoders, and base U-Net weights stay frozen.

---

## Key Result

One of the most interesting findings:

**Base SDXL at 256×256 produces pure noise.**  
After training the Pixelate LoRA exclusively on 256×256 anime pixel-art images, the same resolution suddenly produces coherent anime-girl faces with clear pixel-art style.

The LoRA effectively taught the model a new operating resolution that the original SDXL had almost no prior for.

---

## Project Configuration

| Component            | Value                                      |
|----------------------|--------------------------------------------|
| Base model           | `stabilityai/stable-diffusion-xl-base-1.0` |
| VAE                  | `madebyollin/sdxl-vae-fp16-fix`       |
| Dataset              | `nullHawk/anime-pixel-art-2`               |
| Selected images      | 2,000 (standalone `1girl` tag)             |
| Caption change       | `1girl` → `girl`                           |
| Resolution           | 256 × 256                                  |
| Batch size           | 1                                          |
| Training steps       | 20,000                                     |
| Learning rate        | 1e-4 (constant, 100 warmup steps)          |
| LoRA rank / alpha    | **8 / 8**                                  |
| LoRA targets         | `to_k`, `to_q`, `to_v`, `to_out.0`         |
| Optimizer            | AdamW                                      |
| Precision            | FP16 mixed precision                       |
| Training environment | Kaggle                                     |

---

## How It Works

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
       SDXL U-Net + LoRA
             ▲
             │ text conditioning
       SDXL text encoders
             │
             ▼
       predicted noise
             │
             ▼
          MSE loss → update LoRA only
```

![Pixelate overview](docs/figures/pixelate-overview.png)

Training starts from real images and learns a noise-prediction objective.  
Inference starts from pure noise and repeatedly denoises it.

---

## Stable Diffusion XL

Stable Diffusion is a **latent diffusion** model — the diffusion process happens in a compressed latent space instead of pixel space. SDXL uses a larger U-Net, more attention capacity, and two text encoders.

```text
TEXT SPACE
prompt → CLIP tokenizers + text encoders → text embeddings
                                              │
LATENT SPACE                                  ▼
random noise ──────────────────────► conditional U-Net
                                              │
                                              ▼
                                        denoised latent
                                              │
IMAGE SPACE                                   ▼
                                        SDXL VAE decoder → image
```

![Stable Diffusion architecture](docs/figures/stable-diffusion.png)

---

## VAE

The VAE maps between pixel space and latent space.

```text
256 × 256 × 3 RGB  →  VAE encoder  →  ~32 × 32 latent
                                          │
                                       diffusion
                                          │
~32 × 32 latent  →  VAE decoder  →  256 × 256 × 3 RGB
```

![VAE](docs/figures/vae.png)

---

## Forward Diffusion

During training, Gaussian noise is added to the clean latent at a random timestep:

$$
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)
$$

![Forward diffusion](docs/figures/forward-diffusion.png)

The model’s job is to predict the noise that was added.

---

## Reverse Diffusion

At inference the process is reversed:

```text
random noise → U-Net → scheduler → less-noisy latent → … → clean latent → VAE decoder → image
```

![Reverse diffusion](docs/figures/reverse-diffusion.png)

---

## U-Net + Attention

The U-Net is the main denoising network (downsampling path → middle block → upsampling path with skip connections).

![Simplified conceptual SDXL U-Net](docs/figures/unet.png)

Attention lets the network decide what information is important:

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left(\frac{QK^{T}}{\sqrt{d_{k}}}\right)V
$$

![Attention](docs/figures/attention.png)

In SDXL, **cross-attention** is especially important: queries come from the image/latent features, while keys and values come from the text embeddings.

![Cross-attention](docs/figures/cross-attention.png)

---

## LoRA

Instead of updating the entire U-Net, LoRA freezes the original weights and learns a small low-rank update:

$$
W' = W + \frac{\alpha}{r}BA
$$

![LoRA](docs/figures/lora.png)

Pixelate applies LoRA only to the attention projections:

```python
LoraConfig(
    r=8,
    lora_alpha=8,
    init_lora_weights="gaussian",
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
)
```

![Pixelate LoRA targets](docs/figures/lora-targets.png)

---

## Training Loop

```text
image → frozen VAE → clean latent
                   + noise + timestep
                   → U-Net + LoRA → predicted noise → MSE loss
                   → only LoRA parameters are updated
```

![Training pipeline](docs/figures/training-pipeline.png)

---

## Something That Surprised Me

Base SDXL at 256×256 is essentially pure noise.  
After training the LoRA on 256×256 anime pixel-art faces, the same resolution produces clean, coherent results.

The adapter didn’t just style the model — it taught it how to operate at a resolution the original model was never good at.

---

## References

- [SDXL paper](https://arxiv.org/abs/2307.01952) — Podell et al., 2023  
- [LoRA paper](https://arxiv.org/abs/2106.09685) — Hu et al., 2021  
- Dataset: [`nullHawk/anime-pixel-art-2`](https://huggingface.co/datasets/nullHawk/anime-pixel-art-2)

---

*Focused SDXL U-Net LoRA experiment for anime pixel-art generation.*
```