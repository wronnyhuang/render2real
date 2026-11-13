# render2real eval (metrics, blind judge, page)

Python: `E=$WORK/eval; PY=$E/venv/bin/python` (opencv, mediapipe, insightface (ArcFace) buffalo_l, torch-cpu + lpips, scikit-image).
ArcFace weights live in `eval/models/`. Clip ids: pair ids (data/pairs/<id>) or drive ids (drive/<id>.mp4: muse_01..04, juno_01..04).

## 1. Metrics (CPU, ~1-1.5 min per clip first time; render/real tracking is cached in eval/cache/)
    cd $E
    nice $PY metrics.py --track fast --variant <variant> --all --workers 6      # every <clip>.mp4 in ROOT/fast/out/<variant>/
    nice $PY metrics.py --track wan  --variant <variant> --clips S01a muse_01 ...
    -> eval/results/<track>_<variant>.json  (per clip + mean.{all,heldout,drive,pairs}); extra: --no-lpips, --variant-dir DIR
Key fields: motion_vs_input.{open_corr,width_corr,lag_ms,pose_mae_centered,blink.f1}, face_det_rate, identity.{mean_cos,min_cos},
identity_real_ceiling, flicker.{tdiff_ratio_vs_real,warp_err_ratio_vs_real,tdiff_ratio_vs_render,warp_err_*}, fidelity.{psnr,ssim,lpips}.
`motion_ref_real_vs_input` = what a perfect model scores on lip/pose agreement (real video vs the render).
Outputs at lower fps are resampled by timestamp onto a 30 fps grid; missing faces give NaN-aware stats and a detection rate.

## 2. Blind judge (Claude subscription, `claude -p --model sonnet`, 4 parallel by default, max 6)
    nice $PY judge.py --track fast --variant <variant> --all [--workers 4]
    -> eval/results/judge_<track>_<variant>.json  (summary, summary_heldout, summary_drive; win rate + Wilson / clip-bootstrap CI)
About 3 calls per clip (2 pairwise orders + one three-way with real). Contact strips go to eval/judge_work/.

## 3. Page
Optional text files (markdown-lite): `eval/summary.md` (summary block), `worked.md`, `didnt.md`, `next.md`.
    $PY build_page.py                                  # -> $OUT_DIR/render2real-poc/{index.html,videos/}
    $PY build_page.py --heldout-per-variant 3 --drive-per-variant 3    # videos per variant (default 3 + 3); ~0.4 MB each
Re-runnable; reuses videos that are newer than their sources, prunes the rest. Reads ROOT/{fast,wan}/{latency,spend}.json (any shape).
Pipeline test with fake outputs: `--dummy [--fake-heldout S01a S02a]` includes eval/dummy/out/{render_copy,real_copy,render_blur}.

## Full run (everything found)
    cd $E; for t in fast wan; do for d in ../$t/out/*/; do v=$(basename $d); nice $PY metrics.py --track $t --variant $v --all --workers 6; $PY judge.py --track $t --variant $v --all; done; done; $PY build_page.py
