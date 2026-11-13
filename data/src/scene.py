"""Sonnet LOOK stage (from earlier stage stage_scene.py): ctx frames + face close-up -> scene json."""
import sys, json, os, subprocess, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sonnet import call
SPEC = open(__import__('os').path.expandvars('$WORK/data/spec/INTERFACE.md')).read()
SCHEMA = SPEC[SPEC.index('## scene/<id>.json'):SPEC.index('## Renderer CLI')]
def grab(vid, t, out, vf=None):
    cmd = ['ffmpeg', '-v', 'error', '-ss', str(t), '-i', vid, '-frames:v', '1']
    if vf: cmd += ['-vf', vf]
    subprocess.run(cmd + ['-q:v', '3', '-y', out], check=True)
def run(cid, wd):
    vid = f'{wd}/ctx.mp4'; d = np.load(f'{wd}/track.npz'); W, H = int(d['W']), int(d['H'])
    imgs = []
    for k, t in enumerate([1.0, 4.0, 7.0]):
        p = f'{wd}/s_full{k}.jpg'; grab(vid, t, p, 'scale=-2:960' if H > 960 else None); imgs.append((f'Full frame at t={t}s ({W}x{H}):', p))
    fl = d['face_lm'][0]; ok = np.where(~np.isnan(fl[:, 0, 0]))[0]; t = ok[len(ok) // 2]; lm = fl[t]
    x0, x1 = np.nanmin(lm[:, 0]) * W, np.nanmax(lm[:, 0]) * W; y0, y1 = np.nanmin(lm[:, 1]) * H, np.nanmax(lm[:, 1]) * H
    cx, cy, s = (x0 + x1) / 2, (y0 + y1) / 2 - 0.1 * (y1 - y0), max(x1 - x0, y1 - y0) * 2.2
    cx0, cy0 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2)); s = int(min(s, W - cx0, H - cy0))
    p = f'{wd}/s_face.jpg'; grab(vid, t / 30.0, p, f'crop={s}:{s}:{cx0}:{cy0},scale=384:384')
    imgs.append(('Close-up of person 0 (head and hair):', p))
    text = f"""These are frames from a real {W}x{H} video clip of 1 talking person(s). We are re-creating it as a
NON-photorealistic stylised 3D scene (simple toy-like 3D character + flat simplified background). Motion comes from a tracker;
YOUR job is the LOOK: describe every person's appearance and the background so the re-creation matches the original as
closely as a simple stylised render can. Pick colours by actually sampling what you see (hex), including lighting tint.
People are numbered left to right in the frame (person 0 = leftmost). If the frame is a split-screen of separate camera
views, give one panel per view (x0/x1 = its horizontal extent, 0..1) — otherwise a single panel 0..1. Background shapes:
up to 8 per panel, the main large visible elements (walls, door frames, lamps, furniture, plants, bright windows,
on-screen captions as a dark/white band), in normalised FULL-frame image coordinates (y down), back to front. Skip anything
hidden behind the person.

Output ONE JSON object exactly in this schema (all keys present):
{SCHEMA}"""
    js, raw = call('scene', cid, text, imgs)
    json.dump(js, open(f'{wd}/scene.json', 'w'), indent=1)
    return js
if __name__ == '__main__':
    print(json.dumps(run(sys.argv[1], sys.argv[2]))[:300])
