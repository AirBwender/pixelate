import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator
from accelerate.utils import set_seed

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers

from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)


# config

MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_NAME = "madebyollin/sdxl-vae-fp16-fix"

DATASET_DIR = "data/anime-pixel-art-2k"
OUTPUT_DIR = "outputs/pixelate-lora"

RESOLUTION = 256
BATCH_SIZE = 1

MAX_TRAIN_STEPS = 20_000

LEARNING_RATE = 1e-4
LR_SCHEDULER = "constant"
LR_WARMUP_STEPS = 100

LORA_RANK = 8

SEED = 42

CHECKPOINTING_STEPS = 1_000

MIXED_PRECISION = "fp16"


# dataset

class PixelArtDataset(Dataset):

    def __init__(self, dataset_dir, tokenizer_one, tokenizer_two):
        self.dataset_dir = Path(dataset_dir)
        self.image_dir = self.dataset_dir / "images"
        self.metadata_file = self.dataset_dir / "metadata.jsonl"

        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two

        self.samples = []

        with self.metadata_file.open("r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image_path = self.dataset_dir / sample["file_name"]

        image = Image.open(image_path).convert("RGB")

        image = torch.from_numpy(
            np.array(image)
        ).float() / 127.5 - 1.0

        image = image.permute(2, 0, 1)

        caption = sample["text"]

        input_ids_one = self.tokenizer_one(
            caption,
            padding="max_length",
            max_length=self.tokenizer_one.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]

        input_ids_two = self.tokenizer_two(
            caption,
            padding="max_length",
            max_length=self.tokenizer_two.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]

        return {
            "pixel_values": image,
            "input_ids_one": input_ids_one,
            "input_ids_two": input_ids_two,
        }


def collate_fn(examples):
    return {
        "pixel_values": torch.stack(
            [x["pixel_values"] for x in examples]
        ),
        "input_ids_one": torch.stack(
            [x["input_ids_one"] for x in examples]
        ),
        "input_ids_two": torch.stack(
            [x["input_ids_two"] for x in examples]
        ),
    }


# text encoder

def encode_text(
    text_encoder_one,
    text_encoder_two,
    input_ids_one,
    input_ids_two,
):
    output_one = text_encoder_one(
        input_ids_one,
        output_hidden_states=True,
    )

    output_two = text_encoder_two(
        input_ids_two,
        output_hidden_states=True,
    )

    prompt_embeds_one = output_one.hidden_states[-2]
    prompt_embeds_two = output_two.hidden_states[-2]

    prompt_embeds = torch.cat(
        [
            prompt_embeds_one,
            prompt_embeds_two,
        ],
        dim=-1,
    )

    pooled_prompt_embeds = output_two[0]

    return prompt_embeds, pooled_prompt_embeds


# checkpoint

def save_lora(unet, accelerator, output_dir):
    unwrapped_unet = accelerator.unwrap_model(unet)

    lora_state_dict = get_peft_model_state_dict(
        unwrapped_unet
    )

    lora_state_dict = convert_state_dict_to_diffusers(
        lora_state_dict
    )

    StableDiffusionXLPipeline.save_lora_weights(
        save_directory=output_dir,
        unet_lora_layers=lora_state_dict,
        safe_serialization=True,
    )


# main

def main():

    set_seed(SEED)

    accelerator = Accelerator(
        mixed_precision=MIXED_PRECISION,
        gradient_accumulation_steps=1,
    )

    device = accelerator.device

    if accelerator.is_main_process:
        print("=" * 60)
        print("PIXELATE")
        print("=" * 60)
        print(f"model:          {MODEL_NAME}")
        print(f"vae:            {VAE_NAME}")
        print(f"resolution:     {RESOLUTION}")
        print(f"batch size:     {BATCH_SIZE}")
        print(f"steps:          {MAX_TRAIN_STEPS}")
        print(f"learning rate:  {LEARNING_RATE}")
        print(f"lora rank:      {LORA_RANK}")
        print(f"precision:      {MIXED_PRECISION}")
        print("=" * 60)

    Path(OUTPUT_DIR).mkdir(
        parents=True,
        exist_ok=True,
    )

    # tokenizer

    tokenizer_one = CLIPTokenizer.from_pretrained(
        MODEL_NAME,
        subfolder="tokenizer",
    )

    tokenizer_two = CLIPTokenizer.from_pretrained(
        MODEL_NAME,
        subfolder="tokenizer_2",
    )

    # text encoder

    text_encoder_one = CLIPTextModel.from_pretrained(
        MODEL_NAME,
        subfolder="text_encoder",
    )

    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        MODEL_NAME,
        subfolder="text_encoder_2",
    )

    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    # vae

    vae = AutoencoderKL.from_pretrained(
        VAE_NAME,
    )

    vae.requires_grad_(False)

    # scheduler

    noise_scheduler = DDPMScheduler.from_pretrained(
        MODEL_NAME,
        subfolder="scheduler",
    )

    # unet

    unet = UNet2DConditionModel.from_pretrained(
        MODEL_NAME,
        subfolder="unet",
    )

    unet.requires_grad_(False)

    # lora

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK,
        init_lora_weights="gaussian",
        target_modules=[
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
        ],
    )

    unet.add_adapter(lora_config)

    trainable_parameters = [
        parameter
        for parameter in unet.parameters()
        if parameter.requires_grad
    ]

    if accelerator.is_main_process:
        trainable_count = sum(
            parameter.numel()
            for parameter in trainable_parameters
        )

        print(f"trainable parameters: {trainable_count:,}")

    # optimizer

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    # data

    train_dataset = PixelArtDataset(
        DATASET_DIR,
        tokenizer_one,
        tokenizer_two,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    steps_per_epoch = math.ceil(
        len(train_dataloader) / 1
    )

    num_train_epochs = math.ceil(
        MAX_TRAIN_STEPS / steps_per_epoch
    )

    # scheduler

    lr_scheduler = get_scheduler(
        LR_SCHEDULER,
        optimizer=optimizer,
        num_warmup_steps=LR_WARMUP_STEPS,
        num_training_steps=MAX_TRAIN_STEPS,
    )

    # prepare

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    weight_dtype = torch.float32

    if MIXED_PRECISION == "fp16":
        weight_dtype = torch.float16

    vae.to(device, dtype=weight_dtype)

    text_encoder_one.to(
        device,
        dtype=weight_dtype,
    )

    text_encoder_two.to(
        device,
        dtype=weight_dtype,
    )

    # training

    global_step = 0

    if accelerator.is_main_process:
        print("training started")

    for epoch in range(num_train_epochs):

        unet.train()

        for batch in train_dataloader:

            with accelerator.accumulate(unet):

                pixel_values = batch["pixel_values"].to(
                    device=device,
                    dtype=weight_dtype,
                )

                input_ids_one = batch["input_ids_one"].to(device)
                input_ids_two = batch["input_ids_two"].to(device)

                # latents

                with torch.no_grad():

                    latents = vae.encode(
                        pixel_values
                    ).latent_dist.sample()

                    latents = (
                        latents
                        * vae.config.scaling_factor
                    )

                # text

                with torch.no_grad():

                    prompt_embeds, pooled_prompt_embeds = encode_text(
                        text_encoder_one,
                        text_encoder_two,
                        input_ids_one,
                        input_ids_two,
                    )

                # noise

                noise = torch.randn_like(latents)

                batch_size = latents.shape[0]

                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=device,
                ).long()

                noisy_latents = noise_scheduler.add_noise(
                    latents,
                    noise,
                    timesteps,
                )

                # time ids

                add_time_ids = torch.tensor(
                    [
                        RESOLUTION,
                        RESOLUTION,
                        0,
                        0,
                        RESOLUTION,
                        RESOLUTION,
                    ],
                    device=device,
                    dtype=prompt_embeds.dtype,
                )

                add_time_ids = add_time_ids.unsqueeze(0).repeat(
                    batch_size,
                    1,
                )

                added_cond_kwargs = {
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": add_time_ids,
                }

                # unet

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample

                # loss

                target = noise

                loss = F.mse_loss(
                    model_pred.float(),
                    target.float(),
                    reduction="mean",
                )

                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:

                global_step += 1

                if (
                    accelerator.is_main_process
                    and global_step % 50 == 0
                ):
                    print(
                        f"step {global_step}/{MAX_TRAIN_STEPS} "
                        f"loss {loss.item():.4f} "
                        f"lr {lr_scheduler.get_last_lr()[0]:.2e}"
                    )

                if (
                    global_step % CHECKPOINTING_STEPS == 0
                    and accelerator.is_main_process
                ):
                    checkpoint_dir = (
                        Path(OUTPUT_DIR)
                        / f"checkpoint-{global_step}"
                    )

                    checkpoint_dir.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    save_lora(
                        unet,
                        accelerator,
                        checkpoint_dir,
                    )

                    print(
                        f"checkpoint saved: {checkpoint_dir}"
                    )

            if global_step >= MAX_TRAIN_STEPS:
                break

        if global_step >= MAX_TRAIN_STEPS:
            break

    # final

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:

        save_lora(
            unet,
            accelerator,
            OUTPUT_DIR,
        )

        print("=" * 60)
        print("training complete")
        print(f"output: {OUTPUT_DIR}")
        print("=" * 60)


if __name__ == "__main__":
    main()