# -*- coding: utf-8 -*-
# @file 3c_default_metadata.py
# @author zhangshilong
# @date 2024/9/11

import os

from tools import *

# 将metadata与生成图片拷贝到数据集内，方便后续统一管理

dataset = "yuer48"
image_dir = f"/mnt/workspace/dataset/{dataset}/default"
os.makedirs(image_dir, exist_ok=True)

metadata_path = os.path.join(image_dir, "metadata.jsonl")
write_jsonl(metadata_path, default_metadata)
