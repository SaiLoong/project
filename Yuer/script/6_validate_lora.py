# -*- coding: utf-8 -*-
# @file 6_validate_lora.py
# @author zhangshilong
# @date 2024/9/8

import os

from tqdm import tqdm

from tools import *

pipe = load_aw14()

# 加载lora权重和层
dataset = "yuer100"
trainset = "yuer100r10"
version = "v3"
lora_path = f"/mnt/workspace/LoRA_model/aw14_{trainset}_{version}"
# 要和训练时保持一致
lora_r = 128
lora_alpha = 128
pipe.load_lora_weights(lora_path)
pipe.set_progress_bar_config(disable=True)

# ================================================================================================================


splits = ["train", "test", "synthesis", "default"]
lora_weights = [0.3, 0.5, 0.7, 1.]

image_dirs = dict()
metadatas = dict()
raw_generate_dirs = dict()
lora_generate_dirs = dict()
for split in splits:
    image_dir = f"/mnt/workspace/dataset/{dataset}/{split}"
    image_dirs[split] = image_dir

    metadata_path = os.path.join(image_dir, "metadata.jsonl")
    metadata = read_jsonl(metadata_path)
    metadatas[split] = metadata

    raw_generate_dir = f"/mnt/workspace/output/aw14_{dataset}/raw/{split}"
    raw_generate_dirs[split] = raw_generate_dir

    lora_generate_dir = f"/mnt/workspace/output/aw14_{dataset}/{trainset}-{version}/{split}"
    os.makedirs(lora_generate_dir, exist_ok=True)
    lora_generate_dirs[split] = lora_generate_dir

save_dir = f"/mnt/workspace/output/aw14_{dataset}/{trainset}-{version}/savefig"

model_name = f"aw14-{trainset}-{version}"

buckets = [
    "3409", "3410", "3464", "3474", "4521", "4890",
    "5060", "5549", "5581", "5638", "5971", "6211",
    "6243", "6438", "6701", "6747", "6768", "7162",
    "7187", "7350", "7358", "7557", "7882", "8205",
    "8245", "8311", "8315", "8321", "8325", "8330",
    "8439", "8457", "8490"
]


# ================================================================================================================


def generate_with_lora_weight(prompt, lora_weight=0.):
    # 没有加载lora时不受影响，也不会报错
    pipe.set_adapters("default_0", lora_weight * lora_alpha / lora_r)
    return generate(pipe, prompt)


def collect_row(split, meta):
    filename = meta["file_name"]
    prompt = meta["text"]

    images = list()
    titles = list()

    if split in ["train", "test"]:
        images.append(Image.open(os.path.join(image_dirs[split], filename)))
        titles.append(f"{filename} {split}原图")

    images.append(Image.open(os.path.join(raw_generate_dirs[split], filename)))
    titles.append(f"{filename} lora_weight=0.0")

    for lora_weight in lora_weights:
        w = f"_w{lora_weight}".replace(".", "") + "."
        new_filename = filename.replace(".", w)
        lora_generate_path = os.path.join(lora_generate_dirs[split], new_filename)

        if os.path.isfile(lora_generate_path):
            image = Image.open(lora_generate_path)
        else:
            image = generate_with_lora_weight(prompt, lora_weight)
            image.save(lora_generate_path, quality=95, subsampling=0)

        images.append(image)
        titles.append(f"{lora_weight=}")

    return images, titles


def collect_bucket(bucket):
    images = list()
    titles = list()
    for split in ["train", "test"]:
        metadata = metadatas[split]
        for meta in metadata:
            filename = meta["file_name"]
            if not filename.startswith(bucket):
                continue

            images2, titles2 = collect_row(split, meta)
            images += images2
            titles += titles2
    return images, titles


def filter_metadata(metadata):
    new_metadata = list()
    buckets = set()
    for meta in metadata:
        filename = meta["file_name"]
        bucket = filename.split("_")[0]
        if bucket in buckets:
            continue

        buckets.add(bucket)
        new_metadata.append(meta)

    return new_metadata


def collect_split(split, filter=True):
    metadata = metadatas[split]
    if filter and split == "train":
        metadata = filter_metadata(metadata)

    images = list()
    titles = list()
    for meta in tqdm(metadata, desc=split):
        images2, titles2 = collect_row(split, meta)
        images += images2
        titles += titles2
    return images, titles


# ================================================================================================================


# 单个split
save_split_dir = os.path.join(save_dir, "split")
os.makedirs(save_split_dir, exist_ok=True)
for split in splits:
    save_path = os.path.join(save_split_dir, f"{split}.jpg")
    if not os.path.isfile(save_path):
        images, titles = collect_split(split, filter=False)

        n_cols = len(lora_weights) + (2 if split in ["train", "test"] else 1)
        suptitle = f"模型: {model_name}\n\n 数据集: {dataset}-{split}\n\n 生成结果\n"
        scale = 0.25 if split in ["train", "test"] else 0.5
        plot_images(images, n_cols=n_cols, suptitle=suptitle, titles=titles,
                    scale=scale, save_path=save_path, show=False)

# ================================================================================================================


# 单个bucket
save_bucket_dir = os.path.join(save_dir, "bucket")
os.makedirs(save_bucket_dir, exist_ok=True)
for bucket in tqdm(buckets):
    save_path = os.path.join(save_bucket_dir, f"{bucket}.jpg")
    if not os.path.isfile(save_path):
        images, titles = collect_bucket(bucket)

        n_cols = len(lora_weights) + 2
        suptitle = f"模型: {model_name}\n\n 数据集: {dataset}\n\n bucket: {bucket}\n\n train/test生成结果\n"
        plot_images(images, n_cols=n_cols, suptitle=suptitle, titles=titles,
                    scale=0.25, save_path=save_path, show=False)


# ================================================================================================================


def break_prompt(prompt):
    mid_idx = len(prompt) // 2
    l = prompt[:mid_idx]

    if "," in l:
        sep = ","
    elif "." in l:
        sep = "."
    else:
        sep = " "

    l1, l2 = l.rsplit(sep, 1)
    return l1 + f"{sep}\n" + l2 + prompt[mid_idx:]


# 单条prompt
for split in splits:
    save_prompt_dir = os.path.join(save_dir, "prompt", split)
    os.makedirs(save_prompt_dir, exist_ok=True)

    for meta in tqdm(metadatas[split], desc=split):
        filename = meta["file_name"]
        prompt = meta["text"]

        save_path = os.path.join(save_prompt_dir, filename)
        if not os.path.isfile(save_path):
            images, titles = collect_row(split, meta)
            suptitle = f"模型: {model_name}\n\n 数据集: {dataset}-{split}\n\n prompt: {break_prompt(prompt)}\n"
            scale = 0.25 if split in ["train", "test"] else 0.5
            plot_images(images, n_rows=1, suptitle=suptitle, titles=titles,
                        scale=scale, save_path=save_path, show=False)
