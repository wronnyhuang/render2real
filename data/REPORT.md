# render2real DATA: render/real training pairs

## Result
- **138 clips** (8.0 s, 240 frames, 30 fps, 384x384, H.264 yuv420p). `render.mp4` and `real.mp4` share the same frames and the same crop.
- **41 identities**: train 126 clips from 33 identities, heldout 12 clips from 8 identities.
- Heldout identities: P0115 (older Asian man), P0217 (young white man, curly hair), P0370 (Black man, locs), P0485 (older Black woman, grey hair), P0845 (white woman, red hair), P0989 (white man, glasses), P1067 (older Latina/Indigenous woman, grey hair), P1174 (older white man).
- At most 5 clips per identity, at most 2 per source video, and windows within a video are at least 30 s apart.
- `manifest.json` lists every clip with its id, identity, split, source and frames, plus `mouth_corr` and `face_width_frac`. It also has a `dropped` list.
- `pairs/<id>/meta.json` holds the source file, start frame, crop in 1080x1920 source pixels, speech and frontal fractions, and the QC numbers.

## Identity shortfall (90 identities were asked for)
- The local Seamless copy has 210 videos but only **50 distinct P-ids**, not ~121. Videos exist only for dev pairs, and the P-id with its `A` suffix removed gives the same count.
- 9 of those 50 are filmed standing full-body. Their faces are about 70 px wide in the 1080p source, so a 384 crop would be a 2.7x upscale. They were excluded, which leaves 41 identities.
- To reach the clip target, the per-identity cap was raised to 5.

## Pipeline (src/)
1. `select.py` uses Seamless keypoints and VAD to pick windows with at least 90% frontal frames, at least 50% speech and low head motion.
   - Static square crop: S = 2 x ear-to-ear width, so the face is about 47% of the crop width.
   - The nose sits at 47% of S. The face box must stay inside the crop on at least 98% of frames.
   - 272 candidate windows gave 139 selected.
2. `pipe.py` stage A: one ffmpeg pass produces both
   - `real.mp4`, the source crop scaled to 384 with lanczos (a 0.9-1.4x scale from native 1080p), and
   - a native-resolution context crop for tracking, 1.45*S wide in 9:16, so the face is about 1/3 of the frame width.
   - Then MediaPipe tracking (`track.py`) and the rig with default gains (`rig.py`, no Sonnet calibration).
3. Stage S: one Sonnet "look" call per clip through `claude -p --model sonnet` (`scene.py`, same as earlier stage's `stage_scene`), 10 in parallel. The QA, calibration and keyframe stages are skipped.
4. Stage R: `render_blender.py` with a new `--crop x,y,S --size 384` option.
   - Blender renders only the crop window, natively at 384x384, by narrowing the camera FOV and shifting the lens. Nothing is rendered and then thrown away.
   - Settings: 1 thread per Blender process, AA 5 (looks the same as AA 8 and is 1.6x faster), 40 in parallel.
5. QC (`qc.py`): runs MediaPipe face tracking on both 384 videos and measures face coverage, face-in-crop, second face, face width, centre offset, and mouth-open correlation at lags -6 to +6.
   - Gate (`finalize.py`): 240/240 frames, real coverage >= 0.95, render coverage >= 0.9, face inside >= 0.95, mouth corr >= 0.5, |lag| <= 1, no persistent second face.

## QC numbers
- Mouth-open correlation between real and render: median 0.95 (5th percentile 0.79). The best lag is 0 for every kept clip, so the frames are aligned.
- Face width as a fraction of the crop: real median 0.47, render 0.48. Face centres match to within about 1% of the crop.
- All 139 clips were looked at on contact sheets (`qc/full_0..4.jpg`): no intruding second person and no minors.
- Dropped: M109_P1276A, where a hand covers the face and mouth correlation is 0.26.
- Kept after a visual check: M009 and M121. The tracker briefly flagged a second face in each, but no second person is visible.

## Timings (wall clock, today)
| Stage | Time | Per clip |
|---|---|---|
| Seed (12:31-12:35) | ~4 min | |
| Selection | ~1 min | |
| Stage A: cut + track + rig, 16 parallel | ~6 min (12:58-13:04) | track median 38 s, cut 7 s, rig 1.4 s |
| Stage S: Sonnet, 10 parallel, overlapped with A | | median 10.5 s |
| Stage R: render, 40 parallel | ~20 min (12:59-13:18) | median 257 s, about 1.1-1.3 s per frame on one core |
| QC, overlapped | done 13:18 | |
| Finalize | after the coordinator's resume at 13:23 | |

There was also about 10 minutes of lost work:
- **First tracking attempt:** the context crop was 3*S wide, the faces were too small for the face detector, and coverage fell to 0-0.9.
- **First render attempt:** too many threads per process drove the machine load to about 1000.

## Sonnet
- 231 `scene` calls in total, 6 of them retries.
- 139 belong to the final set. About 90 went to the discarded first context crop, plus 1 test call.

## Known defects of the renders (the earlier stage renderer's style)
- It is a stylised toy character, not a likeness. Hair is helmet-like or bob-shaped (long curly hair turns into a cap or a bob), and beards and moustaches are oversized.
- Heads look a bit small on long necks, and the torso is a simple shape. The clothing colour is right but the garment shape and patterns are crude (blocky camo, plaid).
- Glasses are drawn as thin frames. Earrings are added to many people who don't wear them (hoops on most men, e.g. P0947, P0844A).
- Using default eye gains, blinks and eyes-closed come through, but gaze is approximate. Hands only appear when they rise in front of the body.
- The background is a flat grey studio with soft shapes, which roughly matches Seamless's grey backdrop.

## Seed pairs
- The `pairs/S*` folders (16) are the earlier stage seed conversions. They are **not in the final manifest**: different render settings (Sonnet-calibrated `_cal` rig, 540x960 upscale, heads cut at the top), and they overlap train and heldout identities.
