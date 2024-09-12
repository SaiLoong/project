# -*- coding: utf-8 -*-
# @file demo.py
# @author zhangshilong
# @date 2024/8/31

import torch

from diffusers import StableDiffusionPipeline

# 从https://civitai.com/models/61170下载后，转化为diffusers的格式，重点是--half --scheduler_type="dpm"
# 然后把scheduler配置的use_karras_sigmas改为true，这样就是DPM++ 2M Karras了
model_path = "/mnt/workspace/model/awportrait_v14"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.safety_checker = None
pipe = pipe.to("cuda")


# ======================================================================================================

def parse_and_generate(info, seed=-1, test=False):
    infos = info.strip().split("\n")
    prompt = infos[0]
    negative_prompt = infos[1].split(": ", 1)[1]
    args = dict(text.split(": ", 1) for text in infos[2].split(", "))
    print(f"{args=}\n")

    num_inference_steps = int(args["Steps"])
    guidance_scale = int(args["CFG scale"])

    if seed is None:
        generator = None
    else:
        if seed == -1:
            seed = int(args["Seed"])
        generator = torch.Generator("cuda").manual_seed(seed)

    width, height = map(int, args["Size"].split("x"))
    # clip_skip = int(args["Clip skip"])
    clip_skip = None

    if test:
        negative_prompt = None

    print(f"{prompt=}\n")
    print(f"{negative_prompt=}\n")

    image = pipe(
        prompt, negative_prompt=negative_prompt, generator=generator,
        num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
        height=height, width=width, clip_skip=clip_skip
    ).images[0]

    return image


raw_info = """

A woman striking a poised stance,with a traditional East Asian influence. She's wearing a floral-patterned qipao in rich red and green hues,complementing her classic hairstyle,set against a plain backdrop with delicate flowers.,
Negative prompt: ng_deepnegative_v1_75t,(badhandv4:1.2),(worst quality:2),(low quality:2),(normal quality:2),lowres,bad anatomy,bad hands,((monochrome)),((grayscale)) watermark,moles,large breast,big breast,long fingers:1 bad hand:1,many legs,many shoes,
Steps: 40, CFG scale: 7, Sampler: DPM++ 2M Karras, Seed: 3817915034, Size: 512x768, Model: AWPortrait_1.4, Version: v1.7.0, TI hashes: [object Object], Model hash: 1d66b36bd7, Hires steps: 40, Hires upscale: 1.5, Hires upscaler: Lanczos, ADetailer model: face_yolov8n.pt, ADetailer version: 23.11.1, Denoising strength: 0.4, ADetailer mask blur: 4, ADetailer confidence: 0.3, ADetailer dilate erode: 4, ADetailer inpaint padding: 32, ADetailer denoising strength: 0.3, ADetailer inpaint only masked: True

"""

parse_and_generate(raw_info, test=False)

# ==============================================================


StableDiffusionPipeline().__call__()
