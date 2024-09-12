# -*- coding: utf-8 -*-
# @file demo_base_refiner.py
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


prompt = "Watercolor painting of a desert landscape, with sand dunes, mountains, and a blazing sun, soft and delicate brushstrokes, warm and vibrant colors"
negative_prompt = "(EasyNegative),(watermark), (signature), (sketch by bad-artist), (signature), (worst quality), (low quality), (bad anatomy), NSFW, nude, (normal quality)"
seed = torch.Generator("cuda").manual_seed(42)

latent = base_pipe(prompt=prompt, negative_prompt=negative_prompt, generator=seed, output_type="latent").images
image = refiner_pipe(prompt=prompt, negative_prompt=negative_prompt, generator=seed, image=latent).images[0]
