# -*- coding: utf-8 -*-
# @file reconstruction.py
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


model_path = "/mnt/workspace/ddpm-ema-celebahq-256"
unet = UNet2DModel.from_pretrained(model_path).cuda()
scheduler = DDPMScheduler.from_pretrained(model_path)

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

    x_rec = x_t
    for s in trange(t - 1, -1, -1):
        with torch.no_grad():
            pred_noise = unet(x_rec, s).sample
        x_rec = scheduler.step(pred_noise, s, x_rec).prev_sample

    return x_t, x_rec


timestamps = [100, 300, 500, 800, 1000]
noise_images, rec_images = zip(*[reconstruct_from_xt(image, t) for t in timestamps])

tuples = [
    ("噪声图片", noise_images),
    ("重建图片", rec_images)
]
for suptitle, images in tuples:
    images = torch.concat(images)
    titles = [f"{t=}" for t in timestamps]
    plot_images(images, n_rows=1, suptitle=suptitle, titles=titles, scale=3)
