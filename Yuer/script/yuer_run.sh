export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"  # 减少显存碎片

MODEL_PATH="/mnt/workspace/model/awportrait_v14"
# 数据集太小时，频繁切换epoch期间会降速，重复训练集
DATASET_PATH="/mnt/workspace/dataset/yuer100r10"
OUTPUT_PATH="/mnt/workspace/LoRA_model/aw14_yuer100r10_v3"

# train, 4890_03.jpg
PROMPT1="female, young woman, Asian, long hair, shoulder-length, blue dress, realistic, 1girl, portrait, Yuer, long sleeves, casual, outdoor, solo, nature, park, fence, stone path, autumn, serene, peaceful, calm, light makeup, simple accessories, soft focus"

# test, 8330_12.jpg
PROMPT2="female, Asian, young adult, white dress, long hair, natural light, 1girl, outdoor, greenery, solo, realistic, trees, Yuer, summer, casual, hat, sunlight, serene, nature, beauty, fashion, portrait, park, daylight, relaxed, smiling"

# synthesis, s06_b.jpg
PROMPT3="portrait, 1girl, green tank top, Yuer, city life, realistic, solo, bookshelf, long brown hair, neutral colors, long-sleeved dress, floral pattern, table setting, luxury, simple setting, light green wall, long-sleeve top, holding bowl, floral bouquet, Japanese ethnicity, bending over, smiling"

# default, d11.jpg
PROMPT4="A woman dressed in a traditional East Asian outfit poses against an earth-toned background. Her elegant attire and the intricate hair accessories set against her long,flowing hair evoke a sense of historical beauty and poise.,"


# --validation_prompts后面必须用空格，不能用等号
accelerate launch yuer_train_lora.py \
  --pretrained_model_name_or_path=$MODEL_PATH \
  --train_data_dir=$DATASET_PATH \
  --validation_prompts "$PROMPT1" "$PROMPT2" "$PROMPT3" "$PROMPT4" \
  --num_validation_images=2 \
  --validation_epochs=5 \
  --output_dir=$OUTPUT_PATH \
  --seed=1024 \
  --random_flip \
  --train_batch_size=4 \
  --max_train_steps=105000 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-5 \
  --lr_scheduler="cosine_with_restarts" \
  --lr_warmup_steps=0 \
  --lr_num_cycles=14 \
  --snr_gamma=5 \
  --dataloader_num_workers=8 \
  --max_grad_norm=1 \
  --mixed_precision="fp16" \
  --report_to="wandb" \
  --checkpointing_steps=1000 \
  --checkpoints_total_limit=2 \
  --resume_from_checkpoint="latest" \
  --lora_rank=128 \
  --lora_alpha=128 \
  --lora_weight=0.7


# accelerate launch
# --config_file "default_config_fp16.yaml"
# --max_train_samples=32
