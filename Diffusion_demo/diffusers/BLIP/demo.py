# -*- coding: utf-8 -*-
# @file demo.py
# @author zhangshilong
# @date 2024/9/4

from PIL import Image

from transformers import BlipForConditionalGeneration
from transformers import BlipProcessor

model_path = "/mnt/workspace/model/blip-image-captioning-large"
processor = BlipProcessor.from_pretrained(model_path)
model = BlipForConditionalGeneration.from_pretrained(model_path, device_map="cuda")

image_path = "/mnt/workspace/dataset/woman_dog.jpg"
raw_image = Image.open(image_path).convert('RGB')

# conditional image captioning
text = "a photography of"
inputs = processor(raw_image, text, return_tensors="pt").to("cuda")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))

# unconditional image captioning
inputs = processor(raw_image, return_tensors="pt").to("cuda")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))
