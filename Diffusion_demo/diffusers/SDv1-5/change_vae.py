# -*- coding: utf-8 -*-
# @file change_vae.py
# @author zhangshilong
# @date 2024/9/2

import torch

from diffusers import AutoencoderKL
from diffusers import StableDiffusionPipeline

vae_path = "/mnt/workspace/model/sd-vae-ft-mse"
vae = AutoencoderKL.from_pretrained(vae_path,
                                    torch_dtype=torch.float16, device_map="auto")

model_path = "/mnt/workspace/model/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_path,
                                               vae=vae,
                                               torch_dtype=torch.float16, device_map="balanced")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]
