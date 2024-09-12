# -*- coding: utf-8 -*-
# @file tools.py
# @author zhangshilong
# @date 2024/9/8

import json
import math

import jsonlines
import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from PIL import Image

from diffusers import DPMSolverMultistepScheduler
from diffusers import StableDiffusionPipeline
from diffusers.utils import pt_to_pil

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 正确显示中文
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号“-”


# ================================================================================================================


def read_json(path, *args, **kwargs):
    with open(path, "r") as file:
        return json.load(file, *args, **kwargs)


def write_json(path, data, ensure_ascii=False, indent=4, *args, **kwargs):
    with open(path, "w") as file:
        json.dump(data, file, ensure_ascii=ensure_ascii, indent=indent, *args, **kwargs)


def read_jsonl(path):
    with jsonlines.open(path, "r") as f:
        return list(f.iter(type=dict, skip_invalid=True))


def write_jsonl(path, data):
    with jsonlines.open(path, "w") as f:
        f.write_all(data)


# ================================================================================================================


def plot_images(images, n_rows=None, n_cols=None, suptitle=None, titles=None, scale=1., save_path=None, show=True):
    if isinstance(images, Image.Image):
        images = [images]
    elif isinstance(images, torch.Tensor):
        images = pt_to_pil(images)

    n = len(images)
    if n_rows is None and n_cols is None:
        n_cols = math.ceil(math.sqrt(n))
        n_rows = math.ceil(n / n_cols)
    elif n_cols is None:
        n_cols = math.ceil(n / n_rows)
    elif n_rows is None:
        n_rows = math.ceil(n / n_cols)

    titles = titles or [None] * n
    assert len(titles) == n, f"titles({len(titles)})应和images({n})长度一致"

    width, height = images[0].size
    dpi = 96
    width_figsize = int(n_cols * width / dpi * scale)
    height_figsize = int(n_rows * height / dpi * scale)

    # 等价于 fig = plt.figure(figsize=..., layout=...)    axes = fig.subplots(n_rows, n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width_figsize, height_figsize), layout="constrained")
    fig.suptitle(suptitle)
    axes = [axes] if isinstance(axes, Axes) else axes.flatten()

    # axes长度可能会大于images
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)

    # 空白图片也有坐标轴，也需要取消
    for ax in axes:
        ax.axis("off")

    if save_path:
        plt.savefig(save_path, pil_kwargs=dict(quality=95, subsampling=0))
    if show:
        plt.show()

    plt.cla()
    plt.close()


def plot_image_with_text(image, text):
    plt.imshow(image)
    plt.axis("off")
    plt.show()
    print(text, "\n\n")


# ================================================================================================================


default_metadata = [
    {'file_name': 'd01.jpg',
     'text': 'upper body portrait of 1girl wear a black color turtleneck sweater,A proud and confident expression,long hair,look at viewers,studio fashion portrait,studio light,pure white background,'},
    {'file_name': 'd02.jpg',
     'text': 'upper body portrait of 1girl wear (red color turtleneck sweater:1),A proud and confident smile expression,long hair,look at viewers,studio fashion portrait,studio light,pure white background,'},
    {'file_name': 'd03.jpg',
     'text': 'upper body portrait of 1girl wear suit with tie,A proud and confident smile expression,long hair,look at viewers,studio fashion portrait,studio light,pure white background,'},
    # {'file_name': 'd04.jpg',
    #  'text': "A woman striking a poised stance,with a traditional East Asian influence. She's wearing a floral-patterned qipao in rich red and green hues,complementing her classic hairstyle,set against a plain backdrop with delicate flowers.,"},
    # {'file_name': 'd05.jpg',
    #  'text': 'upper body portrait of 1girl,sling dress,standing,wearing a floral-patterned qipao in rich red and green hues,(outdoors:1)  look at view,'},
    {'file_name': 'd06.jpg',
     'text': "A black and white portrait of a young woman with a captivating gaze. She's bundled up in a cozy black sweater,hands gently cupped near her face. The monochromatic tones highlight her delicate features and the contemplative mood of the image.,"},
    {'file_name': 'd07.jpg',
     'text': 'Fashion photography portrait,((upper body:1)),(1girl is surrounded by Peacock feather from head to neck , ((long hair:1)), (wear Peacock feather outfit fashion with ruffled layers:1)'},
    # {'file_name': 'd08.jpg',
    #  'text': 'a 10-year-old little girl,happy,bokeh,motion blur,facula,documentary photography,in winter,major snow,(snow floating in the air:1.2),high_heels,white pantyhose,white dress,feature article,face close-up,(masterpiece:1.2, best quality)'},
    # {'file_name': 'd09.jpg',
    #  'text': 'Asian female relax,Thin cardigan sweater,Capris,sports shoes,OOTD,Outfit of the Day,full_body,solo,simple_background,light green background,'},
    {'file_name': 'd10.jpg',
     'text': 'upper body portrait of 1girl,sling dress,standing,(Best quality:1.2),(masterpiece:1.2),(ultra high res:1.2),8k,Photography,super detailed,Depth of field,Bright color,Korean style photography,fresh and natural,girlish,bright colors,exquisite makeup,elegant posture,fashionable elements,light and shadow interweaving,atmosphere creation,detail processing,visual appeal,'},
    {'file_name': 'd11.jpg',
     'text': 'A woman dressed in a traditional East Asian outfit poses against an earth-toned background. Her elegant attire and the intricate hair accessories set against her long,flowing hair evoke a sense of historical beauty and poise.,'},
    # {'file_name': 'd12.jpg',
    #  'text': 'upper body portrait of asian female 80 years old,Wearing luxurious clothing,standing in front of a simple background,studio,'},
    # {'file_name': 'd13.jpg',
    #  'text': 'upper body of asian female 60 years old,Wearing luxurious clothing,standing in front of a simple background,studio,'},
    {'file_name': 'd14.jpg',
     'text': 'realistic 1girl,solo,looking at viewer,realistic,upper body,long hair,earrings,blurry background,brown hair,blurry,jewelry,plant,bangs,green sweater,smile,shirt,lips,leaf,signature,black eyes,closed mouth,black hair,'},
    {'file_name': 'd15.jpg',
     'text': 'A pensive schoolgirl with a bow in her hair and wearing a dark blazer over a white shirt with a red ribbon tie looks away thoughtfully. The soft daylight accentuates her youthful innocence.,eyes look down,'},
    {'file_name': 'd16.jpg',
     'text': 'a 16-year-old girl,n,bokeh,motion blur,(facula),feature article,face close-up,teeth,smile,an adoring expression,be shy,blush,model,female model,beautiful face,upper_body,chignon,clothes of diamonds,stage,diamonds floating in the air,(masterpiece:1.2, best quality),(a hand waving to the audience:1),'},
    {'file_name': 'd17.jpg',
     'text': 'A close-up portrait of a young woman with a delicate bow in her hair. Her features are highlighted by subtle makeup,accentuating her youthful appearance and innocent expression. The background is simple,focusing attention on her detailed features and the reflective quality in her eyes.,'}
]


def load_aw14():
    model_path = "/mnt/workspace/model/awportrait_v14"
    pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipe.safety_checker = None
    pipe = pipe.to("cuda")

    # DPM++ 2M Karras
    assert isinstance(pipe.scheduler, DPMSolverMultistepScheduler)
    assert pipe.scheduler.config["use_karras_sigmas"] is True

    return pipe


def generate(pipe, prompt):
    negative_prompt = "ng_deepnegative_v1_75t,(badhandv4:1.2),(worst quality:2),(low quality:2),(normal quality:2),lowres,bad anatomy,bad hands,((monochrome)),((grayscale)) watermark,moles,large breast,big breast,long fingers:1 bad hand:1,many legs,many shoes,"
    generator = torch.Generator("cuda").manual_seed(1024)

    image = pipe(
        prompt, negative_prompt=negative_prompt, generator=generator,
        num_inference_steps=40, guidance_scale=7,
        height=768, width=512
    ).images[0]
    return image
