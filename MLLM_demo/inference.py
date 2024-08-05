# -*- coding: utf-8 -*-
# @file inference.py
# @author zhangshilong
# @date 2024/8/1

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import set_seed

set_seed(1024)

model_path = "/mnt/workspace/Qwen-VL-Chat-Int4"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, device_map="cuda"
).eval()

# 第一轮对话
query = tokenizer.from_list_format([
    {"image": "demo.jpeg"},  # 还可以直接填URL
    {"text": "这是什么?"},
])
print(query, "\n")
"""
Picture 1: <img>demo.jpeg</img>
这是什么?
"""

response, history = model.chat(tokenizer, query=query, history=None)
print(response)
# 图中是一名年轻女子在沙滩上和她的狗玩耍，狗的品种是拉布拉多犬。


# 第二轮对话
response, history = model.chat(tokenizer, "框出图中击掌的位置", history=history)
print(response)
# <ref>击掌</ref><box>(515,507),(589,611)</box>

image = tokenizer.draw_bbox_on_latest_picture(response, history)
if image:
    image.save("demo_out.jpg")
else:
    print("no box")
