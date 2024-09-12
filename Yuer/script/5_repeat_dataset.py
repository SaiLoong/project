# -*- coding: utf-8 -*-
# @file 5_repeat_dataset.py
# @author zhangshilong
# @date 2024/9/9

import os
import shutil

from tools import *

# 小数据集频繁切换epoch期间会降速

dataset = "yuer100"
dataset_dir = f"/mnt/workspace/dataset/{dataset}"
image_dir = os.path.join(dataset_dir, "train")
metadata_path = os.path.join(image_dir, "metadata.jsonl")
metadata = read_jsonl(metadata_path)

repeat_num = 10
new_dataset_dir = f"{dataset_dir}r{repeat_num}"
new_image_dir = os.path.join(new_dataset_dir, "train")
os.makedirs(new_image_dir, exist_ok=True)
new_metadata_path = os.path.join(new_image_dir, "metadata.jsonl")

new_metadata = list()
for meta in metadata:
    filename = meta["file_name"]
    caption = meta["text"]
    image_path = os.path.join(image_dir, filename)

    for idx in range(repeat_num):
        new_filename = f"r{idx:02}_{filename}"
        new_image_path = os.path.join(new_image_dir, new_filename)
        shutil.copy(image_path, new_image_path)

        new_metadata.append({"file_name": new_filename, "text": caption})

write_jsonl(new_metadata_path, new_metadata)
