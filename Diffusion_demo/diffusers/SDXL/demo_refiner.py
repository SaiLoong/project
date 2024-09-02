# -*- coding: utf-8 -*-
# @file demo_refiner.py
# @author zhangshilong
# @date 2024/9/1

import torch

from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

# Pipeline可以直接使用DiffusionPipeline，明确写更方便进入源码
refiner_path = "/mnt/workspace/stable-diffusion-xl-refiner-1.0"
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    refiner_path,
    torch_dtype=torch.float16, variant="fp16",
    device_map="balanced"
)

# https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/aa_xl/000000009.png
init_image = load_image("000000009.png")
prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, image=init_image).images[0]
