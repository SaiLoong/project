# -*- coding: utf-8 -*-
# @file verify_caption.py
# @author zhangshilong
# @date 2024/9/4

import math
import os

import jsonlines
import matplotlib.pyplot as plt
import torch
from diffusers import AutoencoderKL
from diffusers import StableDiffusionXLPipeline
from diffusers.utils import pt_to_pil
from matplotlib.axes import Axes
from PIL import Image
from tqdm import tqdm

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 正确显示中文
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号“-”


def plot_images(images, n_rows=None, n_cols=None, suptitle=None, titles=None, scale=1.):
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

    width, height = images[0].size
    dpi = 96
    width_figsize = int(n_cols * width / dpi * scale)
    height_figsize = int(n_rows * height / dpi * scale)

    # 等价于 fig = plt.figure(figsize=..., layout=...)    axes = fig.subplots(n_rows, n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width_figsize, height_figsize), layout="constrained")
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


# ================================================================================================================


# Pipeline可以直接使用DiffusionPipeline，明确写更方便进入源码
base_path = "/mnt/workspace/model/stable-diffusion-xl-base-1.0"
vae_path = "/mnt/workspace/model/sdxl-vae-fp16-fix"

# SDXL自带的VAE容易出现数值不稳定，解码时容易变成全黑图
# 详见https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16, device_map="auto")
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_path, vae=vae,
    torch_dtype=torch.float16, variant="fp16",
    device_map="balanced"
)


# ================================================================================================================


def read_jsonl(path):
    with jsonlines.open(path, "r") as f:
        return list(f.iter(type=dict, skip_invalid=True))


image_dir = "/mnt/workspace/dataset/beauty/train"
metadata_path = os.path.join(image_dir, "metadata.jsonl")
metadata = read_jsonl(metadata_path)
N = len(metadata)

# ================================================================================================================


generate_dir = "/mnt/workspace/output/beauty_sdxl_generate"
os.makedirs(generate_dir, exist_ok=True)
images = list()
titles = list()

# 耗时37:52，平均每个15.46s
# 66、115超过77个词元，进行截断
for meta in tqdm(metadata):
    file_name = meta["file_name"]
    text = meta["text"]
    title = file_name
    seed = torch.Generator("cuda").manual_seed(1024)

    image = pipe(text, generator=seed).images[0]
    images.append(image)
    titles.append(title)

    generate_path = os.path.join(generate_dir, file_name)
    image.save(generate_path)

plot_images(images, n_cols=10, suptitle=f"原生SDXL生成结果（共{N}张）", titles=titles, scale=0.15)
