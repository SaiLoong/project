# -*- coding: utf-8 -*-
# @file preprocess_image.py
# @author zhangshilong
# @date 2024/9/4

import os

from PIL import Image


def resize(image, size):
    if isinstance(size, int):
        width, height = image.size
        factor = size / min(width, height)
        size = (round(width * factor), round(height * factor))
    return image.resize(size)


# 加载、过滤图片
image_dir = "/mnt/workspace/dataset/raw_beauty"
image_filenames = os.listdir(image_dir)

images = list()
for filename in image_filenames:
    path = os.path.join(image_dir, filename)
    try:
        image = Image.open(path)
    except IsADirectoryError:  # .ipynb_checkpoints
        continue

    if min(image.size) < 1024:
        continue

    image = resize(image.convert("RGB"), 1024)
    images.append(image)

print(f"{len(images)=}")

# =============================================================================================

# 编号、保存
save_dir = "/mnt/workspace/dataset/beauty/train"
os.makedirs(save_dir, exist_ok=True)
for idx, image in enumerate(images, start=1):
    path = os.path.join(save_dir, f"{idx:03}.jpg")
    image.save(path)
