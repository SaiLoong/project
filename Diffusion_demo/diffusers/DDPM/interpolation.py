# -*- coding: utf-8 -*-
# @file interpolation.py
# @author zhangshilong
# @date 2024/8/25

import math

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from PIL import Image
from torchvision import transforms
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


# =============================================================================================


def _load_image(number, transform):
    image_filename = f"{number}.jpg"
    image_path = f"/mnt/workspace/dataset/CelebA/data256x256/{image_filename}"
    raw_image = Image.open(image_path)
    image = transform(raw_image).unsqueeze(0).cuda()
    return image_filename, raw_image, image


def load_images(*numbers):
    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    image_filenames, raw_images, images = zip(*[_load_image(number, transform) for number in numbers])
    plot_images(raw_images, n_rows=1, suptitle="原图", titles=image_filenames)
    return images


img1, img2, _ = load_images("02122", "22644", "20813")

# ================================================================================================

model_path = "/mnt/workspace/model/ddpm-ema-celebahq-256"
unet = UNet2DModel.from_pretrained(model_path).cuda()
scheduler = DDPMScheduler.from_pretrained(model_path)


# 这里的t指论文记号，即x_0表示原图，x_1、x_2、...、x_1000表示加噪图片
def forward(x_0, t):
    assert t > 0
    noise = torch.randn_like(x_0)
    # 第0个系数对应x_1
    return scheduler.add_noise(x_0, noise, torch.tensor(t - 1))


def backward(x_t, t):
    x_rec = x_t
    for t in trange(t - 1, -1, -1):
        with torch.no_grad():
            pred_noise = unet(x_rec, t).sample
        x_rec = scheduler.step(pred_noise, t, x_rec).prev_sample
    return x_rec


def interpolate(x_0a, x_0b, t, lambdas):
    # (1, C, H, W)
    x_ta = forward(x_0a, t)
    x_tb = forward(x_0b, t)

    # (B, 1, 1, 1)
    lambdas = torch.tensor(lambdas, device="cuda").reshape(-1, 1, 1, 1)
    # (B, C, H, W)
    x_t = lambdas * x_ta + (1 - lambdas) * x_tb
    return backward(x_t, t)


# ================================================================================================


timestamps = [100, 300, 500, 700]
lambs = [0.7, 0.5, 0.3]

interpolated_images = list()
titles = list()
for t in timestamps:
    interpolated_images.append(interpolate(img1, img2, t, lambs))
    for lamb in lambs:
        titles.append(f"{t=} lambda={lamb}")
interpolated_images = torch.concat(interpolated_images)

plot_images(interpolated_images, n_rows=len(timestamps), suptitle="图片插值", titles=titles)
