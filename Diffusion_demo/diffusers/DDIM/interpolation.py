# -*- coding: utf-8 -*-
# @file interpolation.py
# @author zhangshilong
# @date 2024/8/29

import math

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from PIL import Image
from tqdm import tqdm

from diffusers import DDIMScheduler
from diffusers import UNet2DModel
from diffusers.utils import pt_to_pil

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 正确显示中文
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号“-”


def plot_images(images, n_rows=None, n_cols=None, suptitle=None, titles=None, scale=2):
    if isinstance(images, Image.Image):
        images = [images]
    elif isinstance(images, torch.Tensor):
        images = pt_to_pil(images)

    n = len(images)
    if n_rows is None and n_cols is None:
        n_cols = math.ceil(math.sqrt(n))
        n_rows = math.ceil(n / n_cols)
    elif n_cols is None:
        n_cols = math.ceil(n / n_rows)
    elif n_rows is None:
        n_rows = math.ceil(n / n_cols)

    titles = titles or [None] * n
    assert len(titles) == n, f"titles({len(titles)})应和images({n})长度一致"

    # 等价于 fig = plt.figure(figsize=..., layout=...)    axes = fig.subplots(n_rows, n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * scale, n_rows * scale), layout="constrained")
    fig.suptitle(suptitle)
    axes = [axes] if isinstance(axes, Axes) else axes.flatten()

    # axes长度可能会大于images
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)

    # 空白图片也有坐标轴，也需要取消
    for ax in axes:
        ax.axis("off")
    plt.show()


# =============================================================================================


model_path = "/mnt/workspace/ddpm-ema-celebahq-256"
dtype = torch.float16
unet = UNet2DModel.from_pretrained(model_path,
                                   torch_dtype=dtype, device_map="cuda")
scheduler = DDIMScheduler.from_pretrained(model_path)

# =============================================================================================


generator = torch.Generator("cuda").manual_seed(1024)


def gen_sample(batch_size=1):
    return torch.randn(
        batch_size, unet.config.in_channels, unet.config.sample_size, unet.config.sample_size,
        device="cuda", dtype=dtype, generator=generator
    )


def step(sample, eta=0.0, num_inference_steps=50):
    scheduler.set_timesteps(num_inference_steps)

    for t in tqdm(scheduler.timesteps):
        with torch.no_grad():
            pred_noise = unet(sample, t).sample

        sample = scheduler.step(pred_noise, t, sample, eta=eta, generator=generator).prev_sample

    return sample


def linear_interpolation(img1, img2, lambdas):
    # (B, 1, 1, 1)
    lambdas = torch.tensor(lambdas, device="cuda", dtype=dtype).reshape(-1, 1, 1, 1)

    # (B, C, H, W)
    return lambdas * img1 + (1 - lambdas) * img2


def spherical_interpolation(img1, img2, lambdas):
    # (B, 1, 1, 1)
    lambdas = torch.tensor(lambdas, device="cuda", dtype=dtype).reshape(-1, 1, 1, 1)

    # (B, C, H, W)
    return torch.sin(lambdas * torch.pi / 2) * img1 + torch.cos(lambdas * torch.pi / 2) * img2


latent1 = gen_sample()
latent2 = gen_sample()
lambdas = [0.9, 0.7, 0.5, 0.3, 0.1]

tuples = [
    ("线性插值", linear_interpolation),
    ("球面插值", spherical_interpolation)
]
for suptitle, interpolation_func in tuples:
    fused_latents = interpolation_func(latent1, latent2, lambdas)
    latents = torch.concat([latent1, fused_latents, latent2])
    images = step(latents)

    titles = ["image1"] + [f"lamda={lamb}" for lamb in lambdas] + ["image2"]
    plot_images(images, n_rows=1, suptitle=suptitle, titles=titles)
