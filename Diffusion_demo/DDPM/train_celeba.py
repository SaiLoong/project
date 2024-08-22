# -*- coding: utf-8 -*-
# @file train_celeba.py
# @author zhangshilong
# @date 2024/8/22

from denoising_diffusion_pytorch import GaussianDiffusion
from denoising_diffusion_pytorch import Trainer
from denoising_diffusion_pytorch import Unet

# 256 batch_size只能设8，太慢了
IMAGE_SIZE = 128
DATA_DIR = f"/mnt/workspace/dataset/CelebA/data{IMAGE_SIZE}x{IMAGE_SIZE}"

model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 4),
    flash_attn=False
)

diffusion = GaussianDiffusion(
    model,
    image_size=IMAGE_SIZE,
    timesteps=1000,  # number of steps
    sampling_timesteps=250
    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)

trainer = Trainer(
    diffusion,
    DATA_DIR,
    train_batch_size=32,
    train_lr=8e-5,
    train_num_steps=50000,  # total training steps
    save_and_sample_every=500,
    num_samples=64,
    results_folder="results/celeba",
    gradient_accumulate_every=2,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    amp=True,  # turn on mixed precision
    calculate_fid=False,  # whether to calculate fid during training
    num_fid_samples=640  # 默认是50000，大约需要2h
)

trainer.train()
