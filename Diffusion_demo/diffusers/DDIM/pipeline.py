# -*- coding: utf-8 -*-
# @file ddim.py
# @author zhangshilong
# @date 2024/8/29

import math

import matplotlib.pyplot as plt
import numpy as np
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


def plot_denoise_progress(scheduler_outputs, show=6):
    indices = np.linspace(0, len(scheduler_outputs) - 1, show, dtype=int)
    timestamps = np.array(sorted(scheduler_outputs.keys(), reverse=True))[indices]

    # current_sample_coeff、pred_original_sample_coeff、noise_coeff属性都是修改源码添加的
    tuples = [
        ("生成图片", "prev_sample", "current_sample_coeff"),
        ("预测原图", "pred_original_sample", "pred_original_sample_coeff")]

    for suptitle, image_key, coeff_key in tuples:
        images = torch.stack([scheduler_outputs[t][image_key] for t in timestamps])
        images = images.permute(1, 0, 2, 3, 4).reshape(-1, *images.shape[-3:])

        titles = list()
        for t in timestamps:
            title = f"{t=}"
            if (coeff := scheduler_outputs[t].get(coeff_key)) is not None:
                title += f"\n{coeff=:.2f}"

            if (noise_coeff := scheduler_outputs[t].get("noise_coeff")) is not None:
                title += f"\n{noise_coeff=:.2f}"

            titles.append(title)
        titles += [None] * (len(images) - show)
        plot_images(images, n_cols=show, suptitle=suptitle, titles=titles)


# =============================================================================================


model_path = "/mnt/workspace/ddpm-ema-celebahq-256"
dtype = torch.float16
unet = UNet2DModel.from_pretrained(model_path,
                                   torch_dtype=dtype, device_map="cuda")
scheduler = DDIMScheduler.from_pretrained(model_path)

# =============================================================================================


generator = torch.Generator("cuda").manual_seed(1024)


def step(sample, eta=0.0, num_inference_steps=50):
    scheduler.set_timesteps(num_inference_steps)

    outputs = dict()
    for t in tqdm(scheduler.timesteps):
        with torch.no_grad():
            pred_noise = unet(sample, t).sample

        output = scheduler.step(pred_noise, t, sample, eta=eta, generator=generator)
        sample = output.prev_sample
        outputs[t.item()] = output

    return outputs


sample = torch.randn(
    2, unet.config.in_channels, unet.config.sample_size, unet.config.sample_size,
    device="cuda", dtype=dtype, generator=generator
)

# eta须<=1
outputs = step(sample, eta=0, num_inference_steps=50)
plot_denoise_progress(outputs, show=11)
