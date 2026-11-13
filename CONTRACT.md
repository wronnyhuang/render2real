# render2real render→photoreal PoC — shared contract

Goal: a neural "last layer" that turns the avatar's code-RENDERED frames into much more photoreal video,
keeping pose, expression, lip shapes and timing. Long-term target is live ~24 fps; this PoC proves quality
offline at LOW RESOLUTION (steer 12:31 PM: "You can do really low resolution to make it faster").

ROOT = $WORK   (each agent writes ONLY inside its own subfolder + reads others)

## Video format (everything: pairs, drive clips, model outputs)
- 384x384 square, head-and-shoulders crop (face roughly 45-55% of frame width, static crop per clip), H.264 yuv420p mp4, no audio needed.
- pairs: 30 fps, 8.0 s (240 frames), render.mp4 and real.mp4 frame-aligned.
- model outputs: 384x384, same duration as input; fps may be lower (e.g. 16) — eval aligns by timestamp.

## Folders
- data/pairs/<clip_id>/{render.mp4,real.mp4,meta.json}   (DATA agent)
- data/manifest.json : {"clips":[{"id","identity","split":"train"|"heldout","source":"seamless","frames":240}]}
- data/SEED_DONE  : written once the first >=12 pairs exist (so trainers can smoke-test early)
- data/DONE       : written when the full set + manifest are final
- drive/<name>.mp4 : Muse / Juno avatar renders in the video format above (EVAL agent), drive/DONE marker
- fast/out/<variant>/<clip_id>.mp4 , wan/out/<variant>/<clip_id>.mp4 : outputs for EVERY heldout clip and EVERY drive clip
- fast/REPORT.md, wan/REPORT.md, */latency.json, */spend.json ({"usd":..,"pod_hours":..,"rate":..})
- eval/ : metrics + blind judge + page builder (EVAL agent)

## Hard rules (project's)
- Money: the only paid thing is RunPod GPU via $RUNPOD_API_KEY (already in env; never print it). Each GPU agent has its own USD cap in its brief.
  RunPod account balance is only ~$15 and ANOTHER pod (avatar-demo, the project's live demo) draws from it:
  NEVER touch any pod you did not create. Before creating a pod and every ~15 min, query clientBalance; if it is < $4.50, finish/stop at once.
  Name your pod r2r-<agent>. TERMINATE (delete) your pod when done and verify it is gone. Track spend = wall hours x $/h.
- Text / judge / vision LLM calls: Claude subscription only (`claude -p --model sonnet ...`), never any OpenAI key, no paid APIs.
- Data: Meta Seamless Interaction only (CC BY-NC, research). Adults only. Outputs stay private (no uploads to public hosts, no HF pushes).
- Shared machine: never pkill/killall; kill only your own pids. Do not restart/stop any service. Use `nice` for heavy CPU jobs. Host has 70 cores, a useless K620 GPU.
- Do not edit files outside ROOT (reading elsewhere is fine). Python: make your own venv under your subfolder if you need packages.
- Final report (your last message): what you built, exact paths, numbers, what failed, spend. No promises of later work.
