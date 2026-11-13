**Recommended: build on the one-step turbo model, not Wan, and train it on pairs rendered by the live renderer for one fixed photoreal identity per character.**

- **1. Match the input domain.** Retarget the tracked motion from the Seamless clips onto the Juno rig and render the pairs with the live SDF renderer, so training input looks exactly like what the model sees live. This is the largest gap measured (lip correlation 0.92 on de-render inputs vs about 0.5 on live renders).
- **2. Fix the identity.** A live avatar needs one person, not any person. Either add a reference-image input, or generate one consistent target identity (the Wan output is steady enough to serve as a teacher: identity 0.95) and distil it into the turbo model.
- **3. Remove the shimmer properly.** Train with a rollout loss on the model's own previous output, keeping the latent blend as the fallback.
- **4. Then raise resolution.** Measured turbo cost on the RTX 5090 at 512 px is 63 ms with the full VAE (16 fps) and 15.6 ms with the TAESD tiny VAE (64 fps); TAESD needs about 30 minutes of joint fine-tuning before its picture is usable.

**Path to 24 fps live, in numbers.**
- Budget per frame at 24 fps is 41.7 ms. The turbo model at 384 px takes 21.3 ms on an RTX 5090 (measured), about 28–34 ms on an RTX 4090 (estimate from relative speed, not measured), so 384 px live fits today on one 4090-class GPU beside the renderer.
- With TAESD: 14.1 ms at 384 px and 15.6 ms at 512 px (measured speed; quality not yet valid). TensorRT was not tried; StreamDiffusion reports up to about 91 fps for one-step SD-Turbo image-to-image at 512 px on an RTX 4090 with TensorRT and TAESD (published claim, not measured here).
- Wan: 146 ms per frame at 6 steps on an RTX PRO 4500 (measured), 2.3× slower than its own 16 fps. Published streaming distillations of Wan 1.3B (CausVid about 9 fps, Self-Forcing about 16–17 fps at 832×480 on an H100-class GPU; claims, not measured here) reach near real time only at 16 fps on a much larger GPU, and would still need the lip problem solved.
- Added latency for the live demo would be one frame of model time (about 21–34 ms) plus encode, on top of today's pipeline.

**Estimated cost of the next round:** about $8–12 of GPU (2–3 hours on one RTX 5090 for training, plus renders on the host), no API spend.
