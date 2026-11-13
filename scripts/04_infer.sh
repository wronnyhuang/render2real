#!/bin/bash
# Stage 3: inference on heldout clips + extra render-only clips (DRIVE). Frames 384x384, 30 fps.
set -e; cd "$(dirname "$0")/.."; . scripts/env.sh
CK=${1:-$RUNS/turbo_lora/last.pt}      # or fast/weights/turbo_lora.pt from the artifacts
cd fast/code
python infer.py --kind turbo --ckpt $CK --variant turbo_lora     --data $DATA --drive $DRIVE --outroot $WORK/fast/out
python infer.py --kind turbo --ckpt $CK --variant turbo_lora_ema --lat_ema 0.35 --data $DATA --drive $DRIVE --outroot $WORK/fast/out
# latency: python latency.py (see file header); reported: 21.3 ms/frame, 47 fps (5090, fp16 + torch.compile)
