import os
os.environ['HF_HOME'] = './cache/hf_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import gradio as gr
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import AutoPipelineForText2Image
from PIL import Image
import numpy as np

# blip model
print("init blip model")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base",resume_download=False)
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float16,resume_download=False).to("cuda")

# diffusers model
print("init diffusers model")
diffuser_pipeline = AutoPipelineForText2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16")
diffuser_pipeline.to("cuda")

# qwen model
print("init qwen model")
os.environ['MODELSCOPE_CACHE'] = './cache/qwen_cache'
from modelscope import AutoTokenizer, AutoModelForCausalLM
qwen_tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen-1_8B-Chat-Int4",trust_remote_code=True)
qwen_model = AutoModelForCausalLM.from_pretrained(
    "qwen/Qwen-1_8B-Chat-Int4", 
    device_map="auto",
    trust_remote_code=True
).eval()

def mypipeline(input_image):
    # caption
    inputs = blip_processor(input_image, return_tensors="pt").to("cuda", torch.float16)
    out = blip_model.generate(**inputs)
    blip_result = blip_processor.decode(out[0], skip_special_tokens=True)
    print(blip_result)

    # story
    response, _ = qwen_model.chat(qwen_tokenizer, f"你好！请帮我写一个故事，关于{blip_result}", history=None)
    return response

def change_image(input_image,input_text):
    init_image = Image.fromarray(np.array(input_image)).resize((512,512))
    image = diffuser_pipeline(input_text, image=init_image).images[0]
    return image

with gr.Blocks() as demo:
    gr.Markdown("# 天池小课堂——看图写故事")
    with gr.Column():
        with gr.Row():
            input_image = gr.Image()
            with gr.Column():
                change_text = gr.Textbox(label="引导提示",lines=3)
                change_button = gr.Button("修改图片")
        run_button = gr.Button("讲故事！")
        result_text = gr.Textbox(lines=5,label="生成故事")

    change_button.click(fn=change_image, inputs=[input_image,change_text], outputs=input_image)
    run_button.click(fn=mypipeline, inputs=input_image, outputs=result_text)
demo.launch()

# 本地无显卡可尝试在DSW-A10环境下使用如下代码 替换 demo.lauch() 运行
# name = os.environ['JUPYTER_NAME']​
# region = os.environ['dsw_region']​
# ​
# host = "dsw-gateway-{region}.data/aliyun.com".format(region=region)​
# ​
# port = 7860​
# root_path = f'/{name}/proxy/{port}'​
# demo.launch(root_path=root_path, server_port=port)