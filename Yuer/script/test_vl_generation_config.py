# -*- coding: utf-8 -*-
# @file test_vl_generation_config.py
# @author zhangshilong
# @date 2024/9/7

import os
import random

import matplotlib.pyplot as plt
import torch
from PIL import Image

from transformers import Qwen2VLForConditionalGeneration
from transformers import Qwen2VLProcessor

vl_model_path = "/mnt/workspace/model/Qwen2-VL-7B-Instruct"
vl_model = Qwen2VLForConditionalGeneration.from_pretrained(
    vl_model_path, torch_dtype=torch.bfloat16, device_map="cuda",
    # 默认"sdpa"，即使用F.scaled_dot_product_attention。不加这个单张图片推理都会爆显存，加上后就ok了
    attn_implementation="flash_attention_2"
)
vl_processor = Qwen2VLProcessor.from_pretrained(vl_model_path)


# =======================================================================================


def caption_image(image):
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
    text_prompt = vl_processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = vl_processor(
        text=[text_prompt], images=[image], padding=True, return_tensors="pt"
    ).to("cuda")

    # Inference: Generation of the output
    output_ids = vl_model.generate(**inputs)
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = vl_processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
    return output_text


# =======================================================================================

image_dir = "/mnt/workspace/dataset/yuer/train2"
filenames = [filename for filename in os.listdir(image_dir) if filename.endswith(".jpg")]
filenames2 = random.sample(filenames, k=5)

# =======================================================================================


# 参考自https://github.com/jiayev/GPT4V-Image-Captioner/blob/main/saved_prompts.csv
prompt = (
    "As an AI image tagging expert, please provide precise tags for these images of people to enhance CLIP model's understanding of the content. "
    "Employ succinct keywords or phrases, steering clear of elaborate sentences and extraneous conjunctions. "
    "Prioritize the tags by relevance. "
    "Your tags should first capture key elements of people, include specific details like gender, nationality, attire, actions, pose, expressions, accessories, makeup, composition type, age, etc. "
    "And then capture other elements such as the main subject, setting, artistic style, composition, image quality, color tone, filter, and camera specifications, and any other tags crucial for the image. "
    # "For other image categories, apply appropriate and common descriptive tags as well. "
    # "Recognize and tag any celebrities, well-known landmark or IPs if clearly featured in the image. "
    "Your tags should be accurate, non-duplicative, and within a 20-75 word count range. "
    "These tags will use for image re-creation, so the closer the resemblance to the original image, the better the tag quality. "
    "Tags should be comma-separated. "
    "Exceptional tagging will be rewarded with $10 per image."
)

vl_model.generation_config.update(
    # SD的CLIP只接受77个token，留一些给人为指定标签
    max_new_tokens=60,
    # 在准确性和多样性之间权衡
    top_k=50,
    top_p=1.,
    temperature=0.5
)
print(vl_model.generation_config)

for filename in filenames2:
    image = Image.open(os.path.join(image_dir, filename))
    caption = caption_image(image)

    plt.imshow(image)
    plt.axis("off")
    plt.show()
    print(caption, "\n\n")
