# -*- coding: utf-8 -*-
# @file demo_base_refiner_v2.py
# @author zhangshilong
# @date 2024/9/1

import torch

from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers import StableDiffusionXLPipeline

# Pipeline都可以直接使用DiffusionPipeline，明确写更方便进入源码
base_path = "/mnt/workspace/model/stable-diffusion-xl-base-1.0"
base_pipe = StableDiffusionXLPipeline.from_pretrained(
    base_path,
    torch_dtype=torch.float16, variant="fp16",
    device_map="balanced"
)

refiner_path = "/mnt/workspace/model/stable-diffusion-xl-refiner-1.0"
refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    refiner_path,
    torch_dtype=torch.float16, variant="fp16",
    device_map="balanced"
)

# ==============================================================================================================================


# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 40
high_noise_frac = 0.8
prompt = "A majestic lion jumping from a big stone at night"

image = base_pipe(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_end=high_noise_frac,
    output_type="latent",
).images

image = refiner_pipe(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_start=high_noise_frac,
    image=image,
).images[0]
