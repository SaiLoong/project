# -*- coding: utf-8 -*-
# @file demo.py
# @author zhangshilong
# @date 2024/9/4

import torch
from PIL import Image

from transformers import AutoProcessor
from transformers import Qwen2VLForConditionalGeneration

model_path = "/mnt/workspace/model/Qwen2-VL-7B-Instruct"
# Load the model in half-precision on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="cuda",
    # 默认"sdpa"，即使用F.scaled_dot_product_attention。不加这个单张图片推理都会爆显存，加上后就ok了
    attn_implementation="flash_attention_2"
)
processor = AutoProcessor.from_pretrained(model_path)

# Image
image_path = "/mnt/workspace/dataset/woman_dog.jpg"
image = Image.open(image_path)

conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
            },
            {
                "type": "text",
                "text": "Describe this image."
            }
        ]
    }
]

# Preprocess the inputs
text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
# Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'

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
)
print(output_text)
