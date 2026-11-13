# render2real FAST: per-frame render → photo model

## Result
- **turbo_lora** is the main variant. It is a one-step SD-Turbo img2img model in the pix2pix-turbo style, trained on 126 clips from 33 identities.
  - On the 12 heldout clips (8 unseen identities) it produces a sharp, plausible photoreal person. Pose, head turn, eyelids (closed eyes included), mouth opening, clothing colours and framing all follow the render.
  - **Measured: 21.3 ms/frame = 47 fps at 384 on an RTX 5090.** This is fp16 + torch.compile, batch 1, end to end: uint8 in → VAE encode → UNet → skip-VAE decode → uint8 out.
  - Without compile it runs at 43.7 ms (22.9 fps).
- **turbo_lora_ema** uses the same weights. At inference it blends 35 % of the previous frame's latent into the current one. It cuts frame-to-frame change by 19 % (3.13 → 2.53; real video = 2.68) for free.
- **unet_l1lpips** is the speed anchor: a 20M-param conv U-Net trained without a GAN for 15 min. It runs at 2.7 ms/frame (370 fps, fp16 + compile) at 384. The output is blurry and smeared, and it falls apart on the Muse/Juno drive clips.
- **Failed: the temporal model run on its own previous output.** It turns to noise within about 10 frames. Failure examples are in `out/turbo_lora_temporal_feedback_FAILED/` (4 heldout, juno_01, muse_01). It is not a variant to evaluate.
- **Drive clips:**
  - Juno becomes a convincing photoreal woman with the same cardigan colour, pose, gaze and mouth.
  - Muse (the plush character) becomes a half-plush, half-human hybrid with doll eyes, or a full human. That is expected: nothing in the training data looks like it.

## Deliverables (384x384, 240 frames, 30 fps)
| variant | heldout (12) | drive (8) | train samples |
|---|---|---|---|
| `out/turbo_lora/` | yes | yes | `M002_P0116_trainsample`, `M010_P0299A_trainsample` |
| `out/turbo_lora_ema/` | yes | yes | yes |
| `out/unet_l1lpips/` | yes | yes | yes |
| `out/turbo_lora_temporal_feedback_FAILED/` | 4 (failure examples) | juno_01, muse_01 | no |

- Contact sheets at frame 60:
  - `sheets/heldout_sheet.jpg` (columns: render, turbo_lora, turbo_lora_ema, unet_l1lpips, real)
  - `sheets/drive_sheet.jpg` (columns: render, turbo_lora, ema, unet)
- `latency.json`, `spend.json` and `flicker.json` are in the same folder.
- Weights are kept locally (178 MB total):
  - `weights/turbo_lora.pt` (38 MB: LoRA, skip convs and conv_in only; the base is `stabilityai/sd-turbo` from HF)
  - `weights/turbo_lora_temporal.pt` (38 MB)
  - `weights/unet_l1lpips.pt` (110 MB)
- Code is in `code/`: `turbo.py`, `train_turbo.py`, `train_unet.py`, `infer.py`, `latency.py`, `flicker.py`, `common.py`.
  - Load the turbo weights with `Turbo().load_trainable(torch.load(path))`.
- Training logs and sample grids are in `train_logs/`.

## What was trained
**Data**
- Final manifest only: 126 train clips, 33 identities. Heldout clips were never used.
- Seed-stage runs were thrown away: the seed clips S01/S04/S08 share identities P0115, P0485 and P0845 with the final heldout set. Every model below was retrained from scratch on the final manifest.

**Augmentation**
- Same random affine on render and target: scale 0.9–1.12, ±6 % shift, ±4°, horizontal flip.
- Render input only: down/up-scaling to 35–90 %, blur, bilateral "flat shading", posterize, mild brightness/contrast/saturation jitter (no hue shift, so colours carry through), noise and gamma.

**turbo_lora**
- Base: SD-Turbo UNet with LoRA r8, full conv_in, fixed prompt, t=999. The render latent is used as x_t.
- VAE: LoRA r4 on the SD-VAE, plus 4 skip 1x1 convs from encoder to decoder, initialised at 1e-5. 9.5M trainable params.
- Loss: L2 + 5·LPIPS(VGG) + 0.5·hinge GAN. The GAN is a 3-scale spectral-norm PatchGAN conditioned on the clean render and switches on at step 500.
- AdamW, lr 5e-5, bf16, batch 2.
- **14,451 steps in 72 min** on the RTX 5090. Final train l2 0.017, LPIPS 0.21.

**turbo_lora_temporal**
- Same model with an 8-channel conv_in. The extra channels take the VAE latent of the previous *real* frame (t-1..t-3), degraded, and zeroed 15 % of the time.
- 4,247 steps in 44 min on an RTX A5000. Train LPIPS 0.15.

**unet_l1lpips**
- 5-level residual U-Net, 48→448 channels. Loss: L1 + 2·LPIPS.
- 2,641 steps at batch 8 in 15 min on the A5000. Final l1 0.106, LPIPS 0.26.
- The first try, `unet_gan` (adds a hinge GAN from step 1500), diverged once the GAN switched on (LPIPS 0.34 → 0.59) and was discarded. Its log is in `train_logs/unet_gan_diverged/`.

## Quality findings
**Generalisation (heldout)**
- Clean photoreal faces with the render's expression, mouth and colours.
- Identity is "a plausible person matching the render", not the real person: curly red hair becomes straight brown, and an older Latina becomes an older Black woman.
- With only 33 training identities, different heldout women drift toward similar training "prototype" faces.
- Train clips are near-memorised.

**Flicker** (`flicker.json`, mean |frame_t − frame_t-1| on a 0–255 scale, heldout)

| clip type | all detail | high-frequency only |
|---|---|---|
| render | 1.94 | 1.23 |
| real video | 2.68 | 1.53 |
| turbo_lora | 3.13 | 1.18 |
| turbo_lora_ema | 2.53 | 0.96 |
| unet_l1lpips | 1.44 | 0.53 |

- turbo_lora shows about 17 % more frame-to-frame change than real video, mostly low-frequency: shimmering hair and edges, and slight identity wobble. The EMA brings it below real.
- The U-Net's low numbers come from blur, not stability.

**Identity and mouth**
- Deterministic encoding (VAE mode, no noise) keeps the identity mostly stable within a clip.
- Mouth shapes (open, closed, teeth) track the render well, because the encoder skips carry the render's mouth outline.
- No identity-embedding metric was computed here; that is left to the eval agent.

**Why the temporal model failed**
- It learned to copy the previous frame. With a fixed anchor frame, pose and mouth freeze.
- Fed its own output, errors compound (exposure bias) and it collapses within about 10 frames.
- With a zero previous frame it is a decent single-frame model.
- A fix needs rollout training on its own outputs, or a reference-identity input instead of the previous frame. Not attempted within this budget.

**Drive clips**
- Juno works. The background is partly pulled toward the training data's grey studio wall.
- Muse fails as expected. It would need stylised or plush training pairs.

## Path to 24 fps live (RTX 5090, measured unless marked)
1. **Already above 24 fps.** turbo_lora at 384, fp16 + compile, full skip-VAE: **21.3 ms = 47 fps**.
   - Temporal variant: 27.6 ms = 36 fps.
   - fp16 without compile: 43.7 ms = 22.9 fps. Without compile it is launch-bound, so compile is required.
2. **Where the time goes** (384): the UNet alone takes 9.5 ms compiled. The SD-VAE encode and decode with skips take most of the remaining ~12 ms.
3. **TAESD swap:** TAESD encode + decode with the same UNet runs at **14.1 ms = 71 fps** at 384, and 15.6 ms at 512.
   - This checkpoint was not trained with TAESD, so that output is not valid yet.
   - Next step: about 30 min of fine-tuning the TAESD decoder together with the LoRA, keeping a light skip path.
4. **512** (estimated): 63 ms in fp16 without compile; about 30 ms with compile; about 16 ms with TAESD. That would give 24+ fps at 512 on a 5090.
5. **RTX 4090** (not measured): expect about 1.3–1.6x the 5090 times, so roughly 28–34 ms compiled at 384, or about 30 fps.
6. **Not tried:** CUDA graphs or TensorRT for the UNet.
7. **Speed floor:** the U-Net runs at 2.7 ms/frame. A student distilled from turbo_lora outputs could reach 100+ fps if its quality holds.
8. **Temporal and identity:** the latent EMA is free and helps. Real fixes:
   - a rollout-trained previous-frame condition, or
   - a fixed reference-identity image as a second input, so one chosen person stays the same across a session.

## Spend
- **$2.12 of the $4.00 cap**, wall hours × $/h from creation to verified termination (`spend.json`):

| pod | rate | hours | cost |
|---|---|---|---|
| RTX A5000 | $0.27/h | 2.27 | $0.61 |
| RTX 5090 | $0.69/h | 2.06 | $1.42 |
| RTX 3090 + first RTX 5090 (both died at CUDA init, created without `NVIDIA_DRIVER_CAPABILITIES=all`) | — | 0.28 | $0.07 |
| 3 RTX 4090 SECURE pods (auto-created over the price cap, terminated within seconds) | $0.89/h | — | ≈ $0.02 |

- These hours include about 25 min idle during the quota pause. During that gap, training had crashed: an OOM from sharing the 5090, and a frame-index bug. Both were fixed and the runs restarted.
- Every r2r-fast pod is terminated. The RunPod pod list shows none left.
