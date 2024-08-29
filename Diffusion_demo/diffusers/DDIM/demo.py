# -*- coding: utf-8 -*-
# @file demo.py
# @author zhangshilong
# @date 2024/8/29

import torch

from diffusers import DDIMPipeline

model_path = "/mnt/workspace/ddpm-ema-celebahq-256"

pipe = DDIMPipeline.from_pretrained(model_path,
                                    torch_dtype=torch.float16).to("cuda")

image = pipe().images[0]
