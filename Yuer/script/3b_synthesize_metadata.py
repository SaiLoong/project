# -*- coding: utf-8 -*-
# @file 3b_synthesize_metadata.py
# @author zhangshilong
# @date 2024/9/10

import os
import random

from tools import *
from transformers import CLIPTokenizer

raw_caption_path = "/mnt/workspace/dataset/yuer_raw_caption.json"
raw_caption = read_json(raw_caption_path)

all_tags = list()
for filename, caption in raw_caption.items():
    all_tags += [tag.strip() for tag in caption.split(",")[:-1]]  # 最后的tag可能不完整

# ==================================================================================================


sd_model_path = "/mnt/workspace/model/stable-diffusion-v1-5"
clip_tokenizer = CLIPTokenizer.from_pretrained(sd_model_path, subfolder="tokenizer")

insert_tags = ["Yuer", "1girl", "solo", "portrait", "realistic"]
n = len(insert_tags)


def check_and_save(metadata, tags, filename):
    caption = ", ".join(tags)
    assert len(clip_tokenizer(caption).input_ids) <= clip_tokenizer.model_max_length
    metadata.append({"file_name": filename, "text": caption})


def synthesize_metadata(all_tags, sample_fuc, k, i):
    metadata = list()
    tags = sample_fuc(all_tags, k)
    check_and_save(metadata, tags, f"s{i:02}_a.jpg")

    max_pos = min(15, len(tags))
    # 将固定tag随机插到前15的位置
    for idx, tag in enumerate(insert_tags):
        if tag in tags:
            continue

        index = random.randrange(max_pos - n + idx)
        tags.insert(index, tag)
    check_and_save(metadata, tags, f"s{i:02}_b.jpg")

    return metadata


def weighted_sample(all_tags, k):
    return list(set(random.sample(all_tags, k=2 * k)))[:k]


def unweighted_sample(all_tags, k):
    return random.sample(set(all_tags), k=k)


# ==================================================================================================


num = 5
k = 17  # raw_caption的平均tag数

metadata = list()
i = 0
for func in [weighted_sample, unweighted_sample]:
    for _ in range(num):
        i += 1
        metadata += synthesize_metadata(all_tags, func, k, i)

dataset = "yuer100"
image_dir = f"/mnt/workspace/dataset/{dataset}/synthesis"
os.makedirs(image_dir, exist_ok=True)

metadata_path = os.path.join(image_dir, "metadata.jsonl")
write_jsonl(metadata_path, metadata)
