#!/bin/bash
# Stage 2: train the one-step SD-Turbo LoRA (main model). Reported run: RTX 5090, 72 min, 14,451 steps, batch 2.
set -e; cd "$(dirname "$0")/.."; . scripts/env.sh
cd fast/code
python train_turbo.py --data $DATA --out $RUNS/turbo_lora --minutes 72 --bs 2 --lr 5e-5          # -> $RUNS/turbo_lora/last.pt
# speed-anchor U-Net (A5000, 15 min):  python train_unet.py --data $DATA --out $RUNS/unet_l1lpips --minutes 15 --bs 8
# Wan 2.1 VACE LoRA (RunPod-style GPU, 24-48 GB): bash ../../wan/pod/setup.sh && bash ../../wan/pod/train_split.sh train $RUNS/wan_lora 2
