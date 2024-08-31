# -*- coding: utf-8 -*-
# @file demo.py
# @author zhangshilong
# @date 2024/8/31

import torch

from diffusers import StableDiffusionPipeline

# vae_path = "/mnt/workspace/sd-vae-ft-mse"
# vae = AutoencoderKL.from_pretrained(vae_path,
#                                     torch_dtype=torch.float16, device_map="auto")

model_path = "/mnt/workspace/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_path,
                                               # vae=vae,
                                               torch_dtype=torch.float16, device_map="balanced")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]
