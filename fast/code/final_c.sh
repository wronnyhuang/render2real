#!/bin/bash
# run on pod c after turbo_lora training: inference for turbo_lora + turbo_lora_ema, then latency
cd /root/code
TS=M002_P0116,M010_P0299A
python infer.py --kind turbo --ckpt /root/runs/turbo_lora/last.pt --variant turbo_lora --train_samples $TS > /root/infer_turbo.log 2>&1
python infer.py --kind turbo --lat_ema 0.35 --ckpt /root/runs/turbo_lora/last.pt --variant turbo_lora_ema --train_samples $TS > /root/infer_turbo_ema.log 2>&1
echo INFER_C_DONE > /root/infer_c.done
