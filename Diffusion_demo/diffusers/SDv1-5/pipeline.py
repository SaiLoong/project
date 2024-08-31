# -*- coding: utf-8 -*-
# @file SDv1-5_pipeline.py
# @author zhangshilong
# @date 2024/8/25

import torch
from tqdm import tqdm

from diffusers import AutoencoderKL
from diffusers import PNDMScheduler
from diffusers import UNet2DConditionModel
from diffusers.utils import pt_to_pil
from transformers import CLIPTextModel
from transformers import CLIPTokenizer

model_path = "/mnt/workspace/stable-diffusion-v1-5"
device = "cuda"
dtype = torch.float16

# 1. Load the autoencoder model which will be used to decode the latents into image space.
vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae",
                                    torch_dtype=dtype, device_map="auto")

# 2. Load the tokenizer and text encoder to tokenize and encode the text.
tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder",
                                             torch_dtype=dtype, device_map=device)

# 3. The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet",
                                            torch_dtype=dtype, device_map="auto")

scheduler = PNDMScheduler.from_pretrained(model_path, subfolder="scheduler")
# 虽然换了scheduler，但是参数和原版的完全一样
# scheduler = LMSDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
# scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
#                                  num_train_timesteps=1000)

# =================================================================================================


prompt = [
    # "a photograph of an astronaut riding a horse",
    "a photograph of a cute cat"
]

height = 512  # default height of Stable Diffusion
width = 512  # default width of Stable Diffusion

num_inference_steps = 50  # Number of denoising steps

guidance_scale = 7.5  # Scale for classifier-free guidance

generator = torch.Generator(device).manual_seed(1024)  # Seed generator to create the inital latent noise

# B
batch_size = len(prompt)

# =================================================================================================

# 2B
pos_neg_prompts = [""] * batch_size + prompt

# 2B*77
text_input = tokenizer(pos_neg_prompts, padding="max_length", truncation=True, return_tensors="pt")

with torch.inference_mode():
    # 2B*77*768
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

# =================================================================================================


# B*4*64*64, 其中64=512/8、512是指定的宽高
latents = torch.randn(
    (batch_size, unet.config.in_channels, height // 8, width // 8),
    generator=generator, dtype=dtype, device=device
)
# init_noise_sigma = 1.
latents = latents * scheduler.init_noise_sigma

# =================================================================================================


# scheduler.timesteps长度101, 第一步视作warm_up（除了pipeline进度条不显示没有任何区别）
scheduler.set_timesteps(num_inference_steps)

for t in tqdm(scheduler.timesteps):
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    # 2B*4*64*64
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

    # predict the noise residual
    with torch.inference_mode():
        # 2B*4*64*64
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

    # perform guidance
    # 都是B*4*64*64
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    # B*4*64*64
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # compute the previous noisy sample x_t -> x_t-1
    # B*4*64*64
    latents = scheduler.step(noise_pred, t, latents).prev_sample

# =================================================================================================


# scale and decode the image latents with vae
# scaling_factor = 0.18215
latents /= vae.config.scaling_factor
with torch.inference_mode():
    # B*3*512*512, 其中512是指定的宽高
    image = vae.decode(latents).sample

# =================================================================================================


pil_images = pt_to_pil(image)
pil_images[0]
