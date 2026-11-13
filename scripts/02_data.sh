#!/bin/bash
# Stage 1: data prep. Two routes.
#  (A) --targets-only : you downloaded artifacts (render.mp4 + meta.json per clip); this cuts the matching real.mp4 crops
#      from your local Seamless Interaction copy with ffmpeg, using the crop box in pairs/<id>/meta.json.
#  (B) full rebuild   : select windows -> track -> rig -> (Claude "look" call) -> Blender render. Needs Blender 4.x python (bpy),
#      MediaPipe, the Claude CLI for the scene-look step (data/src/scene.py), and many CPU cores. See data/REPORT.md.
set -e; cd "$(dirname "$0")/.."; . scripts/env.sh
if [ "$1" = "--targets-only" ]; then
python3 - <<'P'
import json,os,subprocess
D=os.environ['DATA']; S=os.environ['SEAMLESS_ROOT']
for c in json.load(open(f'{D}/manifest.json'))['clips']:
    m=json.load(open(f"{D}/pairs/{c['id']}/meta.json")); x,y,s=m['crop_src_1080x1920']
    src=m['src_video'].replace('$SEAMLESS_ROOT',S)
    out=f"{D}/pairs/{c['id']}/real.mp4"
    if os.path.exists(out): continue
    subprocess.run(['ffmpeg','-v','error','-y','-ss',str(m['src_start_s']),'-i',src,'-t','8','-vf',f'crop={s}:{s}:{x}:{y},scale=384:384:flags=lanczos,fps=30','-c:v','libx264','-pix_fmt','yuv420p','-an',out],check=True)
P
else
python3 data/src/select.py && python3 data/src/pipe.py && python3 data/src/finalize.py   # see data/REPORT.md for env vars (NA/NS/NR pools)
fi
