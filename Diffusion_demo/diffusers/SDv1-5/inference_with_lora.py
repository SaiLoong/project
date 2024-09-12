# -*- coding: utf-8 -*-
# @file inference_with_lora.py
# @author zhangshilong
# @date 2024/9/2

import torch

from diffusers import StableDiffusionPipeline

model_path = "/mnt/workspace/model/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_path,
                                               torch_dtype=torch.float16, device_map="balanced")

prompt = "a blue and white dragon with its mouth open"

# 原模型效果
image = pipe(prompt).images[0]

# 加载lora权重
lora_path = "/mnt/workspace/LoRA_model/sd-v1-5-pokemon"
pipe.unet.load_attn_procs(lora_path)
# lora效果
image2 = pipe(prompt).images[0]