# -*- coding: utf-8 -*-
# @file inference_without_noise.py
# @author zhangshilong
# @date 2024/8/25

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from tqdm import trange

from diffusers import DDPMScheduler
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


model_path = "/mnt/workspace/model/ddpm-ema-celebahq-256"
unet = UNet2DModel.from_pretrained(model_path).cuda()
scheduler = DDPMScheduler.from_pretrained(model_path)

generator = torch.Generator("cuda").manual_seed(1024)

sample = torch.randn(
    1, unet.config.in_channels, unet.config.sample_size, unet.config.sample_size,
    device="cuda", generator=generator
)

outputs = dict()
for t in tqdm(scheduler.timesteps):
    with torch.no_grad():
        pred_noise = unet(sample, t).sample

    output = scheduler.step(pred_noise, t, sample, generator=generator)
    sample = output.prev_sample
    outputs[t.item()] = output

plot_denoise_progress(outputs, show=11)

# =============================================================================================


# 重写系数的代码
"""
current_sample_coeff: Optional[torch.Tensor] = None
pred_original_sample_coeff: Optional[torch.Tensor] = None
noise_coeff: Optional[torch.Tensor] = None


# self._get_variance里面的clamp记得屏蔽

raw_variance = self._get_variance(t, predicted_variance=predicted_variance)
sigma = eta * raw_variance ** 0.5
c1 = (1 - alpha_prod_t_prev - sigma**2) ** 0.5
c2 = beta_prod_t ** 0.5

pred_original_sample_coeff = alpha_prod_t_prev ** 0.5 - c1 / c2 * alpha_prod_t ** 0.5
current_sample_coeff = c1 / c2

noise_coeff = sigma
variance = noise_coeff * variance_noise
"""

# =============================================================================================


# Sheldon
img_path = "/mnt/workspace/dataset/CelebA/data256x256/02122.jpg"
raw_image = Image.open(img_path)
plot_images(raw_image, suptitle="原图")
transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
image = transform(raw_image).unsqueeze(0).cuda()


# 这里的t指论文记号，即x_0表示原图，x_1、x_2、...、x_1000表示加噪图片
def reconstruct_from_xt(x_0, t):
    assert t > 0
    noise = torch.randn_like(x_0)
    # 第0个系数对应x_1
    x_t = scheduler.add_noise(x_0, noise, torch.tensor(t - 1))

    outputs = dict()
    x_rec = x_t
    for s in trange(t - 1, -1, -1):
        with torch.no_grad():
            pred_noise = unet(x_rec, s).sample

        output = scheduler.step(pred_noise, s, x_rec)
        x_rec = output.prev_sample
        outputs[s] = output

    return x_t, x_rec, outputs


noise_image, rec_image, outputs = reconstruct_from_xt(image, t=5)
plot_denoise_progress(outputs, show=6)
