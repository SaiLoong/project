MODEL_PATH="/mnt/workspace/model/stable-diffusion-v1-5"
DATASET_PATH="/mnt/workspace/dataset/pokemon-blip-captions"
OUTPUT_PATH="/mnt/workspace/LoRA_model/sd-v1-5-pokemon"

python train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_PATH \
  --variant="fp16" \
  --dataset_name=$DATASET_PATH \
  --validation_prompt="a cartoon dragon" \
  --num_validation_images=4 \
  --validation_epochs 1 \
  --output_dir=$OUTPUT_PATH \
  --seed=1024 \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --train_batch_size=4 \
  --max_train_steps=15000 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-04 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --dataloader_num_workers=8 \
  --max_grad_norm=1 \
  --mixed_precision="fp16" \
  --report_to="wandb" \
  --checkpointing_steps=500 \
  --checkpoints_total_limit=3
