# -*- coding: utf-8 -*-
# @file 1b_preprocess_image_pc.py
# @author zhangshilong
# @date 2024/9/9

import os
import shutil

from tqdm import tqdm

# 将原本放在一个文件夹的图片复制到独立文件夹中，两种存储方式并存，按需使用

download_dir = r"D:\BaiduNetdiskDownload"
image_dir = os.path.join(download_dir, "yuer_unfiltered")
filenames = os.listdir(image_dir)

for filename in tqdm(filenames):
    dirname = filename.split("_")[0]
    new_output_dir = os.path.join(download_dir, "yuer_unfiltered_v2", dirname)
    os.makedirs(new_output_dir, exist_ok=True)

    image_path = os.path.join(image_dir, filename)
    new_output_path = os.path.join(new_output_dir, filename)
    shutil.copy(image_path, new_output_path)
