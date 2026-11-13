# usage: train.sh <out_dir> <epochs> <repeat> [extra args]
cd /w/ds
M=/w/models/Wan-AI/Wan2.1-VACE-1.3B
OUT=$1; EP=$2; REP=$3; shift 3
accelerate launch --num_processes 1 --mixed_precision bf16 examples/wanvideo/model_training/train.py \
  --dataset_base_path /w/sync/train --dataset_metadata_path /w/sync/train/metadata.csv \
  --data_file_keys "video,vace_video" --height 384 --width 384 --num_frames 81 --dataset_repeat $REP \
  --model_paths "[\"$M/diffusion_pytorch_model.safetensors\",\"$M/models_t5_umt5-xxl-enc-bf16.pth\",\"$M/Wan2.1_VAE.pth\"]" \
  --tokenizer_path /w/models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl \
  --learning_rate 1e-4 --num_epochs $EP --remove_prefix_in_ckpt "pipe.vace." \
  --output_path $OUT --lora_base_model vace --lora_target_modules "q,k,v,o,ffn.0,ffn.2" --lora_rank 32 \
  --extra_inputs "vace_video" --use_gradient_checkpointing --dataset_num_workers 4 --enable_csv_log "$@"
