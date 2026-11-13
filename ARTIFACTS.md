# Downloadable artifacts (render2real)

Weights, generated training data and eval outputs are too large for git. They are in a **public** Google Drive folder (anyone with the link can view and download):

* All bundles for both projects: https://drive.google.com/drive/folders/1G7jieF9XQvVJ9VosnpVx-EMzMBytcfAg  ("Avatar photoreal research artifacts")
* This repo's bundles: https://drive.google.com/drive/folders/1Mm-Mdlptx-URoFERc-bJlL5jJl6kMdFW  (`render2real/`)
* `MANIFEST.md` (contents, file counts, sizes): https://drive.google.com/file/d/1-sqYGDXeMgwCUoDzkmTv-EX-FJDk29LN/view  — a copy is in this repo as `MANIFEST.md`

| bundle | size | what is inside | direct link | file id |
|---|---|---|---|---|
| `r2r_weights_fast.tar` | 177.0 MB (185,640,960 B) | fast model weights: `turbo_lora.pt` (main), temporal variant, `unet_l1lpips.pt` | [open](https://drive.google.com/file/d/1kol13rZYwwsYAJe3em6n2nTxFLm5M6nP/view) · [download](https://drive.google.com/uc?export=download&id=1kol13rZYwwsYAJe3em6n2nTxFLm5M6nP) | `1kol13rZYwwsYAJe3em6n2nTxFLm5M6nP` |
| `r2r_weights_wan_lora.tar` | 41.8 MB (43,796,480 B) | Wan 2.1 VACE LoRA weights | [open](https://drive.google.com/file/d/1xUkBqln4F7k4rwU0b1JkGbJud-jCMiMS/view) · [download](https://drive.google.com/uc?export=download&id=1xUkBqln4F7k4rwU0b1JkGbJud-jCMiMS) | `1xUkBqln4F7k4rwU0b1JkGbJud-jCMiMS` |
| `r2r_training_pairs_render_side.tar` | 42.8 MB (44,830,720 B) | training pairs, render side only (`render.mp4` + `meta.json`, `manifest.json`) | [open](https://drive.google.com/file/d/1kb1b2wOh_f2vJYDeTl1X7yPg9fZUhbxj/view) · [download](https://drive.google.com/uc?export=download&id=1kb1b2wOh_f2vJYDeTl1X7yPg9fZUhbxj) | `1kb1b2wOh_f2vJYDeTl1X7yPg9fZUhbxj` |
| `r2r_eval_outputs.tar` | 170.9 MB (179,179,520 B) | generated eval videos + per-clip metric/judge JSON | [open](https://drive.google.com/file/d/1Sm1RZ9SAFmju8gpRCspOf6hRMSUl0BUE/view) · [download](https://drive.google.com/uc?export=download&id=1Sm1RZ9SAFmju8gpRCspOf6hRMSUl0BUE) | `1Sm1RZ9SAFmju8gpRCspOf6hRMSUl0BUE` |

Fetch them all (downloads by file id with `gdown`, no Google sign-in; resumes by skipping complete files, extracts at the repo root):
```bash
bash scripts/01_fetch_artifacts.sh                 # everything
bash scripts/01_fetch_artifacts.sh r2r_weights_fast.tar   # one bundle
```
The id/size list the script uses is `scripts/artifacts.tsv`. If Drive answers with a "too many downloads" page, retry later or download the files from the links above and put them in `artifacts/render2real/`.

Deliberately not in the bundles: real-person videos and voices (re-created from Meta Seamless Interaction by `scripts/02_data.sh`, see `DATA.md`) and base-model weights (Hugging Face ids in `README.md`).
