# render2real — stylised 3D render → photoreal video

Turns a low-poly / stylised 3D character render into a photoreal person that keeps the render's pose, head turn, blinks, mouth
shapes and timing. Two model families were built and compared on the same data:

* **fast** — one-step SD-Turbo img2img (pix2pix-turbo style) with a LoRA, skip-connected VAE decoder, 384×384, 47 fps on one GPU.
* **wan** — Wan 2.1 VACE 1.3B video model with a LoRA, 6-step CausVid sampler, ~7 fps.
* a tiny 20M-param conv U-Net as a speed anchor (370 fps, blurry).

Training data are 138 frame-aligned 8-second pairs (render | real, 384×384, 30 fps) built by "de-rendering" Meta Seamless
Interaction footage: track face/body with MediaPipe, drive a Blender character rig with it, render with the same crop as the real
video. 126 clips / 33 identities train, 12 clips / 8 unseen identities are held out.

## Results (12 held-out clips, 8 unseen people; blind Claude judge + CPU metrics)
| model | realism /10 (judge) | preferred over plain render | mouth-open corr. w/ render | blink F1 | speed |
|---|---|---|---|---|---|
| plain render (input) | 1.5 | – | – | – | – |
| **fast / turbo_lora** (recommended) | **8.0** | 24 / 24 | 0.92 (real video: 0.93) | 0.88 | **21 ms/frame = 47 fps** (RTX 5090, fp16 + compile) |
| fast / turbo_lora_ema (latent blend 0.35) | ≈ same, less shimmer (frame-diff 2.53 vs 3.13; real 2.68) | | | | same |
| fast / unet_l1lpips | blurry (3.3 vs 6.4 in 3-way test) | | | | 2.7 ms = 370 fps |
| wan / lora_prompt_cv6 | similar realism, steadier identity | | 0.55 | 0.59 | 6.9 fps |
| zero-shot SD-Turbo / Wan (no fine-tune) | 2.1–2.4 | | | | |
| real footage (ceiling) | 8.9 in the 3-way test (turbo_lora 6.4) | | 0.93 (vs render) | | |

Known weaknesses: identity is "a plausible person" (33 training identities), hair/edges shimmer without the EMA blend, training
renders are a different style from other renderers, a plush-toy character does not work. Full write-up: `fast/REPORT.md`,
`data/REPORT.md`, `eval/{summary,worked,didnt,next}.md`. GPU spend for everything: $4.57 on rented RTX 5090 / A5000 / L4-class pods.
Per-clip metrics: `eval/results/*.json`.

## Hardware used
* Training/inference: 1× RTX 5090 32 GB (turbo_lora: 72 min, 14,451 steps, batch 2), 1× RTX A5000 24 GB (U-Net, temporal variant), Wan LoRA on a 24–32 GB pod.
* Data + eval: CPU box, 70 cores (MediaPipe tracking, Blender render, SyncNet-free metrics); judge via the Claude CLI.
* Software: Python 3.12, torch 2.8.0+cu128, diffusers 0.43.0, peft 0.21.2 (`requirements.txt`; eval pins in `requirements-eval.txt`).

## Base models (Hugging Face)
`stabilityai/sd-turbo`, `madebyollin/taesd`, `Wan-AI/Wan2.1-VACE-1.3B`, `Wan-AI/Wan2.1-T2V-1.3B` (umt5 tokenizer only),
`Kijai/WanVideo_comfy` (`Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors`), `depth-anything/Depth-Anything-V2-Small-hf`;
Wan training uses [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) at commit `28638042df62d206486a8bcff3ef22973c84e414`
(cloned by `wan/pod/setup.sh`; not vendored here).

## Reproduce, stage by stage
All paths come from `scripts/env.sh` (override with env vars). No hard-coded machine paths remain; Python files read `$WORK`,
`$SEAMLESS_ROOT`, `$OUT_DIR` through `os.path.expandvars`.
```bash
. scripts/env.sh
bash scripts/00_setup.sh                            # pinned deps + base models + MediaPipe task files
bash scripts/01_fetch_artifacts.sh <DRIVE_FOLDER>   # optional: prebuilt pairs (render side), weights, eval outputs (ARTIFACTS.md)
bash scripts/02_data.sh --targets-only              # cut real.mp4 crops from YOUR Seamless Interaction copy (DATA.md) ...
#   or: bash scripts/02_data.sh                       # ... full rebuild (Blender + MediaPipe + Claude CLI for the scene-look call)
bash scripts/03_train.sh                            # turbo_lora (72 min on a 5090)
bash scripts/04_infer.sh [ckpt.pt]                  # turbo_lora + turbo_lora_ema videos for heldout + extra render clips
bash scripts/05_eval.sh turbo_lora_ema              # metrics + judge + results page
```
Wan: `bash wan/pod/setup.sh && bash wan/pod/train_split.sh train $RUNS/wan_lora 2`, then `wan/pod/infer.py` (see `wan/pod/run_all.sh`
for the exact inference commands used). `baselines/` holds the zero-shot baselines and the audio-muxing/page checks.

Layout: `fast/` (model + trainers + report), `wan/` (Wan scripts, prompts, latency), `data/` (pipeline `src/`, manifest, per-clip `pairs/*/meta.json`, QC), `eval/` (metrics, judge, page builder, results), `baselines/`, `scripts/`.
Limits of reproducibility: the judge uses `claude -p`; swap `call()` in `eval/judge.py` for any vision LLM. Rebuilding the render side needs Blender with `bpy`; the artifacts contain the finished renders so you can skip it.

See `DATA.md` for dataset licences and attribution, `ARTIFACTS.md` for the downloadable files.
