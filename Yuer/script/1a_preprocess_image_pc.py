# -*- coding: utf-8 -*-
# @file 1a_preprocess_image_pc.py
# @author zhangshilong
# @date 2024/9/5

import os

from PIL import Image
from tqdm import tqdm


def resize(img, size):
    if isinstance(size, int):
        width, height = img.size
        factor = size / min(width, height)
        size = (round(width * factor), round(height * factor))
    return img.resize(size)


# 将img放缩到其中一边等于目标，另一边大于目标
def close_resize(img, new_width, new_height):
    width, height = img.size
    factor = height / width
    if factor > new_height / new_width:
        img = img.resize((new_width, round(factor * new_width)))
    else:
        img = img.resize((round(new_height / factor), new_height))
    return img


def center_crop(img, new_width, new_height):
    width, height = img.size
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    return img.crop((left, top, right, bottom))


# ============================================================================================


download_dir = r"D:\BaiduNetdiskDownload"
output_dir = os.path.join(download_dir, "yuer_unfiltered")
# 虽然训练、推理尺寸是512*768，但是直接缩到这么小容易丢失图像细节
scale = 2
target_width = 512 * scale
target_height = 768 * scale

os.makedirs(output_dir, exist_ok=True)
image_dirnames = [name for name in os.listdir(download_dir) if name.isdigit()]
target_factor = target_height / target_width

for idx, image_dirname in enumerate(image_dirnames, start=1):
    image_dir = os.path.join(download_dir, image_dirname)
    image_filenames = [name for name in os.listdir(image_dir) if name.endswith(".jpg") and int(name.split(".")[0]) > 1]

    for image_filename in tqdm(image_filenames, desc=f"{image_dirname}({idx}/{len(image_dirnames)})"):
        output_path = os.path.join(output_dir, f"{image_dirname}_{image_filename}")
        if os.path.isfile(output_path):
            continue

        image = Image.open(os.path.join(image_dir, image_filename)).convert("RGB")
        image = close_resize(image, target_width, target_height)
        image = center_crop(image, target_width, target_height)

        assert image.size == (target_width, target_height)
        assert image.mode == "RGB"

        # 双保险，尽可能地减少质量损耗
        # https://pillow.readthedocs.io/en/latest/handbook/image-file-formats.html#jpeg-saving
        image.save(output_path, quality=95, subsampling=0)
