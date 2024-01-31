export http_proxy=http://58.34.83.134:31280/
export https_proxy=http://58.34.83.134:31280/
export HTTP_PROXY=http://58.34.83.134:31280/
export HTTPS_PROXY=http://58.34.83.134:31280/
export HF_HOME=/cpfs01/shared/pjlab-lingjun-landmarks/mali1/.cache

accelerate launch --mixed_precision="fp16" marigold_train.py \
  --pretrained_model_name_or_path "stabilityai/stable-diffusion-2" \
  --val_data_dir examples \
  --output_dir "/cpfs01/shared/pjlab-lingjun-landmarks/mali1/outputs/try0" \
  --noise_offset 0.1 \
  --train_batch_size 16 \
  --learning_rate 3e-05 \
  --max_train_steps 18000 \
  --checkpointing_steps 18000 \
  --gradient_accumulation_steps 2 \
  --lr_scheduler "cosine" --lr_warmup_steps 0 \
  --use_ema \
  --allow_tf32 \
  --use_8bit_adam \
  --mixed_precision "fp16" \
  --dataloader_num_workers 16 \
  --enable_xformers_memory_efficient_attention
  