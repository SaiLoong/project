# -*- coding: utf-8 -*-
# @file average_photo.py
# @author zhangshilong
# @date 2024/8/25

import numpy as np
import torch
from PIL import Image
from tqdm import trange

# =================================================================================


image_dir = "/mnt/workspace/dataset/CelebA/data256x256"

N = 30000
image_sum = torch.zeros(256, 256, 3, dtype=torch.int, device="cuda")
# image_sum = np.zeros((256, 256, 3), dtype=int)
for i in trange(1, N + 1):
    raw_image = Image.open(f"{image_dir}/{i:05}.jpg")

    # image_sum += np.array(raw_image)
    image_sum += torch.as_tensor(np.array(raw_image), device="cuda")

average_image = (image_sum / N).cpu().numpy().astype("uint8")
average_image = Image.fromarray(average_image)
# numpy 26s, tensor+cuda 18s
