# -*- coding: utf-8 -*-
# @file inference_with_lora.py
# @author zhangshilong
# @date 2024/9/3

import torch

from diffusers import AutoencoderKL
from diffusers import StableDiffusionXLPipeline

base_path = "/mnt/workspace/model/stable-diffusion-xl-base-1.0"
vae_path = "/mnt/workspace/model/sdxl-vae-fp16-fix"

# SDXL自带的VAE容易出现数值不稳定，解码时容易变成全黑图
# 详见https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16, device_map="auto")
# Pipeline可以直接使用DiffusionPipeline，明确写更方便进入源码
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_path, vae=vae,
    torch_dtype=torch.float16, variant="fp16",
    device_map="balanced"
)

attn1 = pipe.unet.down_blocks[1].attentions[0].transformer_blocks[0].attn1
assert isinstance(attn1.to_q, torch.nn.Linear), "pipe已被污染"
raw_weight = attn1.to_q.weight.detach().clone()
print(f"{raw_weight=}")


def generate_image(pipe_, seed=1024):
    gen = torch.Generator("cuda").manual_seed(seed)
    prompt = "A photo of a young woman with light-colored hair, wearing a pink crop top and beige shorts, posing on a metallic surface with her arms extended. She has a confident expression and is wearing a black jacket with pink fur cuffs."
    image = pipe_(prompt, seed=gen).images[0]
    return image


generate_image(pipe)

# ================================================================================================


# 加载lora权重和层
lora_path = "/mnt/workspace/LoRA_model/sdxl-beauty"
pipe.load_lora_weights(lora_path)

generate_image(pipe)


# ================================================================================================


def almost_equal(first, second, delta=0.0002):
    # 不论执行safe_lora多少次，始终有0.0001220703125的误差，可能是fp16导致的，反正误差不会累积就行了
    diff = (first - second).abs().max().item()
    return diff < delta


# TODO 可以在不融合的前提下动态设置scale
#  https://huggingface.co/docs/diffusers/using-diffusers/loading_adapters#adjust-lora-weight-scale

# TODO 融合与解除融合lora很危险，来回操作会导致权值不断发生轻微修改。可能这就是即使指定了seed也会得到不同结果的原因
def generate_image_with_lora(lora_scale=1.0):
    base_weight = attn1.to_q.base_layer.weight
    assert almost_equal(base_weight, raw_weight), "A1"
    assert pipe.num_fused_loras == 0, "A2"

    pipe.fuse_lora(lora_scale=lora_scale)
    assert not almost_equal(base_weight, raw_weight), "B1"
    assert pipe.num_fused_loras == 1, "B2"

    image = generate_image(pipe)

    pipe.unfuse_lora()
    assert almost_equal(base_weight, raw_weight), "C1"
    assert pipe.num_fused_loras == 0, "C2"

    # print(f"{raw_weight=}")
    # print(f"{base_weight=}")
    # diff = (raw_weight - base_weight).abs().max().item()
    # print(f"{diff=}")
    return image


generate_image_with_lora(1.0)
