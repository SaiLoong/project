# -*- coding: utf-8 -*-
# @file 3a_generate_metadata.py
# @author zhangshilong
# @date 2024/9/7

import os
import random

from tqdm import tqdm

from tools import *
from transformers import CLIPTokenizer

dataset = "yuer859"
split = "train"
image_dir = f"/mnt/workspace/dataset/{dataset}/{split}"
filenames = [filename for filename in os.listdir(image_dir) if filename.endswith(".jpg")]
filenames.sort()

raw_caption_path = "/mnt/workspace/dataset/yuer_raw_caption.json"
raw_caption = read_json(raw_caption_path)


def plot_image_with_caption(filename, caption):
    image = Image.open(os.path.join(image_dir, filename))
    plot_image_with_text(image, caption)


# =======================================================================================


# 抽查raw_caption有没问题
random_filenames = sorted(random.sample(filenames, k=5))
for filename in random_filenames:
    caption = raw_caption[filename]
    image = Image.open(os.path.join(image_dir, filename))
    plot_image_with_text(image, caption)

# =======================================================================================


# 后处理生成metadata
sd_model_path = "/mnt/workspace/model/stable-diffusion-v1-5"
clip_tokenizer = CLIPTokenizer.from_pretrained(sd_model_path, subfolder="tokenizer")

insert_tags = ["Yuer", "1girl", "solo", "portrait", "realistic"]
n = len(insert_tags)

metadata = list()
for filename in tqdm(filenames):
    caption = raw_caption[filename]
    tags = [tag.strip() for tag in caption.split(",")[:-1]]  # 最后的tag可能不完整

    max_pos = min(15, len(tags))
    # 将固定tag随机插到前15的位置
    for idx, tag in enumerate(insert_tags):
        if tag in tags:
            continue

        index = random.randrange(max_pos - n + idx)
        tags.insert(index, tag)

    caption = ", ".join(tags)
    assert len(clip_tokenizer(caption).input_ids) <= clip_tokenizer.model_max_length
    metadata.append({"file_name": filename, "text": caption})

    if filename in random_filenames:
        image = Image.open(os.path.join(image_dir, filename))
        plot_image_with_text(image, caption)

metadata_path = os.path.join(image_dir, "metadata.jsonl")
write_jsonl(metadata_path, metadata)
