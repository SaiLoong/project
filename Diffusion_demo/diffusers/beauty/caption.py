# -*- coding: utf-8 -*-
# @file caption.py
# @author zhangshilong
# @date 2024/9/4

import os

import jsonlines
import torch
from PIL import Image
from tqdm import trange

from transformers import Qwen2VLForConditionalGeneration
from transformers import Qwen2VLProcessor

model_path = "/mnt/workspace/model/Qwen2-VL-7B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="cuda",
    # 默认"sdpa"，即使用F.scaled_dot_product_attention。不加这个单张图片推理都会爆显存，加上后就ok了
    attn_implementation="flash_attention_2"
)
processor = Qwen2VLProcessor.from_pretrained(model_path)


# ================================================================================================================


def caption(image):
    prompt = "用英文描述图中的女人，包括身材、样貌、年龄、动作、表情等特征。以'A photo of a young woman with'开头，不超过60个单词。"

    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(
        text=[text_prompt], images=[image], padding=True, return_tensors="pt"
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    output_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
    return output_text


image_dir = "/mnt/workspace/dataset/beauty/train"
N = 147

# 耗时8:40，平均每张3.54s
metadata = list()
for idx in trange(1, N + 1):
    file_name = f"{idx:03}.jpg"
    image = Image.open(os.path.join(image_dir, file_name))
    text = caption(image)

    print(f"[{file_name}] {text}")
    metadata.append({"file_name": file_name, "text": text})


# =======================================================================================


def write_jsonl(path, data):
    with jsonlines.open(path, "w") as f:
        f.write_all(data)


metadata_path = os.path.join(image_dir, "metadata.jsonl")
write_jsonl(metadata_path, metadata)
