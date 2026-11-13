#!/bin/bash
# GPU box setup. Needs ffmpeg + rsync; pins from requirements.txt. Base model: stabilityai/sd-turbo (HF).
set -e; cd "$(dirname "$0")/.."
pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python - <<'P'
from huggingface_hub import snapshot_download
snapshot_download('stabilityai/sd-turbo', allow_patterns=['*.json','*.txt','unet/diffusion_pytorch_model.fp16.safetensors','vae/diffusion_pytorch_model.fp16.safetensors','text_encoder/model.fp16.safetensors','tokenizer/*'])
snapshot_download('madebyollin/taesd')
import lpips; lpips.LPIPS(net='vgg')
P
# CPU eval venv (optional, for scripts/05_eval.sh):  python -m venv eval/venv && eval/venv/bin/pip install -r requirements-eval.txt
# MediaPipe landmarker models for eval/data (face_landmarker, pose_landmarker_full, hand_landmarker) -> $WORK/models/
mkdir -p models
B=https://storage.googleapis.com/mediapipe-models
curl -L -o models/face_landmarker.task $B/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
curl -L -o models/pose_landmarker_full.task $B/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
curl -L -o models/hand_landmarker.task $B/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
echo SETUP_OK
