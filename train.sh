export MODEL_NAME=/home/xinhuang/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9/
# export DATASET_NAME="/home/xinhuang/taichi_syndata/SynDatasets/Thumans_half_head_depth/train"
export DATASET_NAME="/home/xinhuang/data/SD_Train_Humans_part_final/image"
# export DATASET_NAME="lambdalabs/pokemon-blip-captions"

accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --dataloader_num_workers=8 \
  --resolution=512 \
  --train_batch_size=4 \
  --max_train_steps=15000 \
  --checkpointing_steps=400000 \
  --learning_rate=5e-06 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir="/home/xinhuang/data/trained_models/Humans_15k_text2image" \
  --gradient_accumulation_steps=1 \
  --use_8bit_adam \
  --use_ema \