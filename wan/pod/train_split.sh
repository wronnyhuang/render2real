# usage: train_split.sh <stage: cache|train> <out_dir> <epochs> [extra]
cd /w/ds
M=/w/models/Wan-AI/Wan2.1-VACE-1.3B
ST=$1; OUT=$2; EP=$3; shift 3
MP="[\"$M/diffusion_pytorch_model.safetensors\",\"$M/models_t5_umt5-xxl-enc-bf16.pth\",\"$M/Wan2.1_VAE.pth\"]"
COMMON="--height 384 --width 384 --num_frames 81 --model_paths $MP --tokenizer_path /w/models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl --learning_rate 1e-4 --num_epochs $EP --remove_prefix_in_ckpt pipe.vace. --lora_base_model vace --lora_target_modules q,k,v,o,ffn.0,ffn.2 --lora_rank 32 --extra_inputs vace_video --use_gradient_checkpointing"
if [ $ST = cache ]; then
accelerate launch --num_processes 1 --mixed_precision bf16 examples/wanvideo/model_training/train.py $COMMON \
  --dataset_base_path /w/sync/train --dataset_metadata_path /w/sync/train/metadata.csv --data_file_keys video,vace_video \
  --dataset_repeat 1 --output_path /w/cache --task sft:data_process --dataset_num_workers 8 "$@"
else
accelerate launch --num_processes 1 --mixed_precision bf16 examples/wanvideo/model_training/train.py $COMMON \
  --dataset_base_path /w/cache --dataset_repeat 1 --output_path $OUT --task sft:train --enable_csv_log \
  --offload_models "$M/models_t5_umt5-xxl-enc-bf16.pth,$M/Wan2.1_VAE.pth" "$@"
fi
