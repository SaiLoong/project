# -*- coding: utf-8 -*-
# @file train_cifar10.py
# @author zhangshilong
# @date 2024/8/22

import os

from torchvision import datasets
from tqdm import tqdm

from denoising_diffusion_pytorch import GaussianDiffusion
from denoising_diffusion_pytorch import Trainer
from denoising_diffusion_pytorch import Unet

DATA_DIR = "/mnt/workspace/dataset"


def save_cifar10():
    dataset = datasets.CIFAR10(DATA_DIR, download=True)

    folder = f"{DATA_DIR}/CIFAR-10"
    os.makedirs(folder, exist_ok=True)
    for idx, (img, label) in enumerate(tqdm(dataset), start=1):
        img.save(f"{folder}/{idx:05}.jpg")


save_cifar10()

# =================================================================================================


model = Unet(
    dim=128,
    dim_mults=(1, 2, 2, 2),
    flash_attn=False
)

diffusion = GaussianDiffusion(
    model,
    image_size=32,
    timesteps=1000,  # number of steps
    sampling_timesteps=250
    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)

trainer = Trainer(
    diffusion,
    "/mnt/workspace/dataset/CIFAR-10",
    train_batch_size=128,
    train_lr=2e-4,
    train_num_steps=70000,  # total training steps
    save_and_sample_every=1000,
    results_folder="results/cifar10",
    gradient_accumulate_every=1,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    amp=True,  # turn on mixed precision
    calculate_fid=False,  # whether to calculate fid during training
    num_fid_samples=640  # 默认是50000，大约需要2h
)

trainer.train()
# 5000step 已经看着差不多了，只是分辨率太低不好判断像不像
