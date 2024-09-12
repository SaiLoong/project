# -*- coding: utf-8 -*-
# @file reconstruction.py
# @author zhangshilong
# @date 2024/9/1

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.axes import Axes
from PIL import Image
from tqdm import trange

from diffusers import AutoencoderKL
from diffusers import DDPMScheduler
from diffusers import UNet2DConditionModel
from diffusers.utils import pt_to_pil
from transformers import CLIPTextModel
from transformers import CLIPTokenizer

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 正确显示中文
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号“-”

device = "cuda"
dtype = torch.float16


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


# 参考自VaeImageProcessor
def pil_to_pt(images):
    if not isinstance(images, list):
        images = [images]
    images = [np.array(image).astype(np.float32) / 255.0 for image in images]
    images = np.stack(images, axis=0)

    images = torch.from_numpy(images.transpose(0, 3, 1, 2)).cuda().to(dtype)
    images = 2 * images - 1
    return images


def load_image(number, hw=512):
    image_filename = f"{number}.jpg"
    image_path = f"/mnt/workspace/dataset/CelebA/data{hw}x{hw}/{image_filename}"
    return Image.open(image_path)


# ========================================================================================================


model_path = "/mnt/workspace/model/stable-diffusion-v1-5"
generator = torch.Generator(device).manual_seed(1024)

vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae",
                                    torch_dtype=dtype, device_map="auto")
unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet",
                                            torch_dtype=dtype, device_map="auto")
# 用回DDPM，比较容易控制step
scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder",
                                             torch_dtype=dtype, device_map=device)


@torch.inference_mode()
def vae_encode_image(x):
    posterior = vae.encode(x).latent_dist
    z = posterior.mode()
    # 不能用 *= ，因为z是encoder输出的一半视图
    return z * vae.config.scaling_factor


@torch.inference_mode()
def vae_decode_latent(z):
    # 不要用 /= ，latent可能还有用，不要修改
    return vae.decode(z / vae.config.scaling_factor).sample


@torch.inference_mode()
def get_text_embedding(prompt=""):
    text_input = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
    text_embedding = text_encoder(text_input)[0]
    return text_embedding


# 这里的t指论文记号，即z_0表示原图，z_1、z_2、...、z_1000表示加噪图片
def add_noise(z_0, t):
    assert t > 0
    # rand_like不能加generator
    noise = torch.randn(z_0.shape, generator=generator, dtype=dtype, device=device)
    # 第0个系数对应z_1
    z_t = scheduler.add_noise(z_0, noise, torch.tensor(t - 1))
    return z_t


# 这里的t指论文记号，即z_0表示原图，z_1、z_2、...、z_1000表示加噪图片
@torch.inference_mode()
def denoise(z_t, t, prompt=""):
    text_embedding = get_text_embedding(prompt)
    z_recon = z_t
    for s in trange(t - 1, -1, -1):
        # encoder_hidden_states必须传入
        pred_noise = unet(z_recon, s, encoder_hidden_states=text_embedding).sample
        z_recon = scheduler.step(pred_noise, s, z_recon).prev_sample
    return z_recon


# 这里的t指论文记号，即z_0表示原图，z_1、z_2、...、z_1000表示加噪图片
def reconstruct(z_0, t, size=None, scale_factor=None, prompt=""):
    z_t = add_noise(z_0, t)
    if size is not None or scale_factor is not None:
        z_t = F.interpolate(z_t, size=size, scale_factor=scale_factor)

    z_recon = denoise(z_t, t, prompt=prompt)
    return z_recon, z_t


def latent_to_pil(z):
    image = vae_decode_latent(z)
    return pt_to_pil(image)


# ========================================================================================================


# Sheldon
hw = 512
raw_image = load_image("02122", hw=hw)
image = pil_to_pt(raw_image)

# VAE重建
latent = vae_encode_image(image)
vae_recon_image = latent_to_pil(latent)[0]

suptitle1 = f"原图分辨率：{hw}"
show_images1 = [raw_image, vae_recon_image]
titles1 = ["原图", "VAE重建"]
plot_images(show_images1, n_rows=1, suptitle=suptitle1, titles=titles1, scale=1.1)

# ========================================================================================================


scale_factor = 1.
resize_hw = int(hw * scale_factor)
t = 10

# noise_latent = torch.randn(
#     (1, unet.config.in_channels, hw // 8, hw // 8),
#     generator=generator, dtype=dtype, device=device
# )
# recon_latent = denoise(noise_latent, t=1000)
recon_latent, noise_latent = reconstruct(latent, t, scale_factor=scale_factor)
latents = torch.cat([recon_latent, noise_latent])
sd_recon_image, noise_image = latent_to_pil(latents)

# 直接插值
interpolate_image = raw_image.resize((resize_hw, resize_hw))

suptitle2 = f"原图分辨率：{hw} 修改分辨率：{resize_hw}"
show_images2 = [interpolate_image, sd_recon_image]
titles2 = ["插值", f"SD-VAE重建({t=})"]
plot_images(show_images2, n_rows=1, suptitle=suptitle2, titles=titles2, scale=1.1)

# ========================================================================================================

plot_images(noise_image, suptitle=f"噪声-VAE重建({t=})")
