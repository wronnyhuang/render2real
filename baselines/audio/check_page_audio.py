#!/usr/bin/env python3
"""ffprobe every video referenced by the comparison page; print audio coverage and each video's mean volume.
  check_page_audio.py [index.html]   (default $OUT_DIR/render2real-poc/index.html)
Exit 1 if any referenced video is missing, has no audio stream, or its audio is digital silence (mean <= -90 dB)."""
import os, re, subprocess, sys

page = sys.argv[1] if len(sys.argv) > 1 else __import__('os').path.expandvars('$OUT_DIR/render2real-poc/index.html')
base = os.path.dirname(os.path.abspath(page))
srcs = sorted(set(re.findall(r'<source src="([^"]+\.mp4)"', open(page).read())))
SILENCE_DB = -90.0
ok = 0; bad = []
for s in srcs:
    p = os.path.join(base, s)
    if not os.path.exists(p): bad.append((s, 'file missing')); print(f'MISSING  {s}'); continue
    a = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a', '-show_entries', 'stream=codec_name,channels,sample_rate,duration',
                        '-of', 'csv=p=0', p], capture_output=True, text=True).stdout.strip()
    v = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v', '-show_entries', 'stream=nb_frames,r_frame_rate,duration',
                        '-of', 'csv=p=0', p], capture_output=True, text=True).stdout.strip()
    if not a: bad.append((s, 'no audio stream')); print(f'NO AUDIO {s}  video[{v}]'); continue
    r = subprocess.run(['ffmpeg', '-v', 'info', '-i', p, '-map', '0:a:0', '-af', 'volumedetect', '-f', 'null', '-'], capture_output=True, text=True).stderr
    m = re.search(r'mean_volume: (-?[\d.]+|-inf) dB', r); mx = re.search(r'max_volume: (-?[\d.]+|-inf) dB', r)
    mv = -999.0 if not m or m.group(1) == '-inf' else float(m.group(1))
    if mv <= SILENCE_DB: bad.append((s, f'digital silence ({mv} dB)')); print(f'SILENT   {s} mean {mv} dB'); continue
    ok += 1
    print(f'ok  {s:48s} mean {mv:6.1f} dB  max {mx.group(1) if mx else "?":>5s} dB  audio[{a}]  video[{v}]')
print(f'\n{ok} with audio / {len(srcs)} total')
if bad:
    print('FAIL:', '; '.join(f'{s} ({w})' for s, w in bad)); sys.exit(1)
