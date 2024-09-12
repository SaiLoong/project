export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"  # 减少显存碎片

MODEL_PATH="/mnt/workspace/model/stable-diffusion-xl-base-1.0"
VAE_PATH="/mnt/workspace/model/sdxl-vae-fp16-fix"
DATASET_PATH="/mnt/workspace/dataset/beauty"
OUTPUT_PATH="/mnt/workspace/LoRA_model/sdxl-beauty"

PROMPT="A photo of a young woman with medium-length wavy hair, seated with one leg extended and the other bent, hand resting on her thigh. She faces the camera, wearing a white strapless crop top and light blue distressed jeans, paired with white sneakers. She gazes directly at the viewer with a composed expression."

# TODO 待尝试--snr_gamma=5
# 结束lr是原来的9.5%
accelerate launch train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_PATH \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --variant="fp16" \
  --train_data_dir=$DATASET_PATH \
  --validation_prompt="$PROMPT" \
  --num_validation_images=4 \
  --validation_epochs=10 \
  --output_dir=$OUTPUT_PATH \
  --seed=1024 \
  --resolution=1024 \
  --random_flip \
  --train_batch_size=1 \
  --max_train_steps=15000 \
  --checkpointing_steps=500 \
  --checkpoints_total_limit=3 \
  --resume_from_checkpoint="latest" \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --lr_scheduler="cosine_with_restarts" \
  --lr_warmup_steps=500 \
  --lr_num_cycles=0.8 \
  --dataloader_num_workers=8 \
  --max_grad_norm=1 \
  --mixed_precision="fp16" \
  --report_to="wandb" \
  --debug_loss
