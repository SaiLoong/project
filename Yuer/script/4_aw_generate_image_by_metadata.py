# -*- coding: utf-8 -*-
# @file 4_aw_generate_image_by_metadata.py
# @author zhangshilong
# @date 2024/9/4

import os

from tqdm import tqdm

from tools import *

pipe = load_aw14()
pipe.set_progress_bar_config(disable=True)

# ================================================================================================================


dataset = "yuer859"
splits = [
    "train",
    # "test", "synthesis", "default"
]

image_dirs = dict()
metadatas = dict()
generate_dirs = dict()
for split in splits:
    image_dir = f"/mnt/workspace/dataset/{dataset}/{split}"
    image_dirs[split] = image_dir

    metadata_path = os.path.join(image_dir, "metadata.jsonl")
    metadata = read_jsonl(metadata_path)
    metadatas[split] = metadata

    generate_dir = f"/mnt/workspace/output/aw14_{dataset}/raw/{split}"
    os.makedirs(generate_dir, exist_ok=True)
    generate_dirs[split] = generate_dir

save_dir = f"/mnt/workspace/output/aw14_{dataset}/raw/savefig"

buckets = [
    "3409", "3410", "3464", "3474", "4521", "4890",
    "5060", "5549", "5581", "5638", "5971", "6211",
    "6243", "6438", "6701", "6747", "6768", "7162",
    "7187", "7350", "7358", "7557", "7882", "8205",
    "8245", "8311", "8315", "8321", "8325", "8330",
    "8439", "8457", "8490"
]


# ================================================================================================================


def generate_image(prompt, generate_path):
    if os.path.isfile(generate_path):
        image = Image.open(generate_path)
    else:
        image = generate(pipe, prompt)
        image.save(generate_path, quality=95, subsampling=0)
    return image


def collect_split(split):
    metadata = metadatas[split]

    images = list()
    titles = list()
    # 3.15s/it
    for meta in tqdm(metadata, desc=split):
        filename = meta["file_name"]
        prompt = meta["text"]

        images.append(generate_image(prompt, os.path.join(generate_dirs[split], filename)))
        titles.append(filename)

    return images, titles


def collect_bucket(bucket):
    split = "train"
    metadata = metadatas[split]

    images = list()
    titles = list()
    for meta in metadata:
        filename = meta["file_name"]
        prompt = meta["text"]
        if not filename.startswith(bucket):
            continue

        images += [
            Image.open(os.path.join(image_dirs[split], filename)),
            generate_image(prompt, os.path.join(generate_dirs[split], filename))
        ]
        titles += [f"{filename} {split}原图", "生成图"]

    return images, titles


# ================================================================================================================


# 单个split
save_split_dir = os.path.join(save_dir, "split")
os.makedirs(save_split_dir, exist_ok=True)
for split in splits:
    save_path = os.path.join(save_split_dir, f"{split}.jpg")
    if not os.path.isfile(save_path):
        images, titles = collect_split(split)

        suptitle = f"模型: aw14\n\n 数据集: {dataset}-{split}\n\n 生成结果\n"
        plot_images(images, n_cols=6, suptitle=suptitle, titles=titles,
                    scale=0.5, save_path=save_path, show=False)

# ================================================================================================================


# 如果train太大，按单个bucket拆开，并与原图对比
save_bucket_dir = os.path.join(save_dir, "bucket")
os.makedirs(save_bucket_dir, exist_ok=True)
for bucket in tqdm(buckets):
    save_path = os.path.join(save_bucket_dir, f"{bucket}.jpg")
    if not os.path.isfile(save_path):
        images, titles = collect_bucket(bucket)

        suptitle = f"模型: aw14\n\n 数据集: {dataset}\n\n bucket: {bucket}\n\n 生成结果\n"
        plot_images(images, n_cols=6, suptitle=suptitle, titles=titles,
                    scale=0.25, save_path=save_path, show=False)
