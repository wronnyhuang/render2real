#!/usr/bin/env python3
"""Build side-by-side videos + the single self-contained results page.
  build_page.py [--out $OUT_DIR/render2real-poc] [--heldout-per-variant 3] [--drive-per-variant 3]
                [--dummy] [--no-videos] [--only-variants fast/v1 wan/v2 ...]
Reads (all optional, page tolerates anything missing):
  ROOT/{fast,wan}/out/<variant>/<clip>.mp4, ROOT/data/{manifest.json,pairs/..}, ROOT/drive/*.mp4
  eval/results/<track>_<variant>.json (metrics), eval/results/judge_<track>_<variant>.json
  eval/summary.md (lead's summary), eval/worked.md, eval/didnt.md, eval/next.md (optional free text; markdown-lite)
  ROOT/{fast,wan}/latency.json, ROOT/{fast,wan}/spend.json
"""
import argparse, html, json, os, re, subprocess, sys, glob, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
TRACKS = ['fast', 'wan']
# per-clip original audio (baselines): <clip_id>.m4a + index.json {clip_id: {path, has_audio, mean_volume_db, source, reason?}}
AUDIO_DIR = __import__('os').path.expandvars('$WORK/audio')
AUDIO_CUTTER = os.path.join(AUDIO_DIR, 'cut_audio.py')  # run with --only <ids> to cut a clip that has no .m4a yet

# ------------------------------------------------------------------ model names shown on the page
# (track, variant) -> (human label, fine_tuned). Add new variants here; unknown ones fall back to model_label().
MODEL_LABELS = {
    ('fast', 'turbo_lora'):      ('SD-Turbo + LoRA (fine-tuned, 1 step)', True),
    ('fast', 'turbo_lora_ema'):  ('SD-Turbo + LoRA + latent blend', True),
    ('fast', 'unet_l1lpips'):    ('Tiny U-Net (fine-tuned)', True),
    ('wan', 'lora_prompt_cv6'):  ('Wan VACE + LoRA, text prompt (fine-tuned, 6 steps)', True),
    ('wan', 'lora_causvid_s6'):  ('Wan VACE + LoRA, no prompt (fine-tuned, 6 steps)', True),
    ('wan', 'lora_raw_s20'):     ('Wan VACE + LoRA, 20 steps', True),
    ('fast', 'base_s025'):       ('SD-Turbo base, strength 0.25 (no fine-tune, 1 step)', False),
    ('fast', 'base_s050'):       ('SD-Turbo base, strength 0.5 (no fine-tune, 1 step)', False),
    ('fast', 'base_s075'):       ('SD-Turbo base, strength 0.75 (no fine-tune, 1 step)', False),
    ('fast', 'base_asft'):       ('SD-Turbo base wired like the fine-tuned model, untrained', False),
    ('wan', 'base_prompt_cv6'):  ('Wan VACE base, text prompt (no fine-tune, 6 steps)', False),
    ('wan', 'base_causvid_s6'):  ('Wan VACE base, no prompt (no fine-tune, 6 steps)', False),
    ('wan', 'z_raw_s30'):        ('Wan VACE base, 30 steps (no fine-tune, 1 clip)', False),
}
TRACK_NAMES = {'fast': 'SD-Turbo track', 'wan': 'Wan VACE track', 'dummy': 'Dummy (pipeline test)'}


def model_label(tr, var):
    """-> (label, fine_tuned True/False/None). Fallback: base_* = zero-shot baseline, anything else = track + variant."""
    if (tr, var) in MODEL_LABELS: return MODEL_LABELS[(tr, var)]
    base = {'fast': 'SD-Turbo', 'wan': 'Wan VACE'}.get(tr, tr)
    if var.startswith('base'):
        rest = var[4:].strip('_').replace('_', ' ')
        return (f'{base} base, zero-shot (no fine-tune)' + (f', {rest}' if rest else ''), False)
    return (f'{TRACK_NAMES.get(tr, tr)}: {var}', None)


def jload(p, default=None):
    try: return json.load(open(p))
    except Exception: return default


def tload(p):
    try: return open(p).read()
    except Exception: return None


# ------------------------------------------------------------------ audio
AUDIO_IDX = {}
EXEMPT_REASONS = ('source has no audio track', 'digital silence')  # the only reasons a page video may stay silent


def audio_id(cid): return cid[:-len('_trainsample')] if cid.endswith('_trainsample') else cid


def load_audio_index(cids):
    """make sure every clip id has an entry (cuts missing ones with baselines/audio/cut_audio.py), then load the index."""
    global AUDIO_IDX
    idx = jload(os.path.join(AUDIO_DIR, 'index.json'), {}) or {}
    need = sorted({audio_id(c) for c in cids if audio_id(c) not in idx or
                   (idx[audio_id(c)].get('has_audio') and not os.path.exists(idx[audio_id(c)].get('path', '')))})
    if need and os.path.exists(AUDIO_CUTTER):
        print('cutting audio for', ' '.join(need))
        subprocess.run([sys.executable, AUDIO_CUTTER, '--only'] + need, capture_output=True, text=True)
        idx = jload(os.path.join(AUDIO_DIR, 'index.json'), {}) or {}
    AUDIO_IDX = idx
    return idx


def audio_for(cid):
    """-> (m4a path or None, index record or None)"""
    r = AUDIO_IDX.get(audio_id(cid))
    if r and r.get('has_audio') and os.path.exists(r.get('path', '')): return r['path'], r
    return None, r


def has_audio_stream(mp4):
    r = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a', '-show_entries', 'stream=codec_name', '-of', 'csv=p=0', mp4],
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


# ------------------------------------------------------------------ videos
def sbs(inputs, labels, out_mp4, poster, size=384, fps=24, kbps=420, audio=None):
    """inputs: list of mp4 paths (left->right); labels: unused since baselines (panel captions are HTML above the video);
    audio: m4a muxed in as-is (AAC copy), video length stays 8.05 s cap (no -shortest)."""
    srcs = list(inputs) + ([audio] if audio else [])
    if (os.path.exists(out_mp4) and all(os.path.getmtime(out_mp4) > os.path.getmtime(p) for p in srcs) and os.path.exists(poster)
            and (not audio or has_audio_stream(out_mp4))):
        return True
    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    cmd = ['ffmpeg', '-v', 'error', '-y']
    for p in inputs: cmd += ['-i', p]
    if audio: cmd += ['-i', audio]
    parts = []
    for i in range(len(inputs)):
        parts.append(f"[{i}:v]fps={fps},scale={size}:{size}:flags=area,setsar=1[v{i}]")
    parts.append(''.join(f'[v{i}]' for i in range(len(inputs))) + f'hstack={len(inputs)}[o]')
    amap = ['-map', f'{len(inputs)}:a:0', '-c:a', 'copy'] if audio else ['-an']
    cmd += ['-filter_complex', ';'.join(parts), '-map', '[o]'] + amap + ['-t', '8.05', '-c:v', 'libx264', '-preset', 'slow',
            '-b:v', f'{kbps}k', '-maxrate', f'{int(kbps * 1.6)}k', '-bufsize', f'{kbps * 2}k', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out_mp4]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0: print('ffmpeg failed', out_mp4, r.stderr[-300:], file=sys.stderr); return False
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', '2.5', '-i', out_mp4, '-frames:v', '1', '-q:v', '5', poster], capture_output=True)
    return os.path.exists(poster)


# ------------------------------------------------------------------ markdown-lite
def esc(s): return html.escape(str(s))


def inline(s):
    s = esc(s)
    s = re.sub(r'`([^`]+)`', r'<code>\1</code>', s)
    s = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', s)
    s = re.sub(r'(?<![\w*])\*([^*\n]+)\*(?!\w)', r'<em>\1</em>', s)
    return s


def md(text):
    if not text or not text.strip(): return None
    out = []; lst = None; para = []
    def flush():
        nonlocal para
        if para: out.append('<p>' + inline(' '.join(para)) + '</p>'); para = []
    def close():
        nonlocal lst
        if lst: out.append(f'</{lst}>'); lst = None
    for ln in text.splitlines():
        s = ln.rstrip()
        m = re.match(r'^(#{1,4})\s+(.*)', s)
        b = re.match(r'^\s*[-*]\s+(.*)', s); n = re.match(r'^\s*\d+[.)]\s+(.*)', s)
        if not s.strip(): flush(); close(); continue
        if m: flush(); close(); lv = min(len(m.group(1)) + 2, 5); out.append(f'<h{lv}>{inline(m.group(2))}</h{lv}>')
        elif b or n:
            flush(); kind = 'ul' if b else 'ol'
            if lst != kind: close(); out.append(f'<{kind}>'); lst = kind
            out.append('<li>' + inline((b or n).group(1)) + '</li>')
        else: close(); para.append(s.strip())
    flush(); close()
    return '\n'.join(out)


def section_text(name):
    t = md(tload(os.path.join(HERE, name)))
    return t or '<p class="muted">Not written yet.</p>'


# ------------------------------------------------------------------ tables
def g(d, path, default=None):
    for k in path.split('.'):
        if not isinstance(d, dict) or k not in d: return default
        d = d[k]
    return d


def fmt(v, nd=3):
    if v is None: return '<span class="muted">n/a</span>'
    if isinstance(v, (int, float)): return f'{v:.{nd}f}'.rstrip('0').rstrip('.') if abs(v) < 1000 else f'{v:.0f}'
    return esc(v)


def table(headers, rows, cls=''):
    h = ''.join(f'<th>{x}</th>' for x in headers)
    b = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def flat_rows(obj, prefix=''):
    """generic dict -> list of (key, value) rows for latency/spend json of unknown shape."""
    rows = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)): rows += flat_rows(v, f'{prefix}{k}.')
            else: rows.append((prefix + str(k), v))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, (dict, list)): rows += flat_rows(v, f'{prefix}{i}.')
            else: rows.append((prefix + str(i), v))
    return rows


def kv_table(obj):
    if obj is None: return '<p class="muted">No file yet.</p>'
    # list of flat dicts -> proper table
    if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
        cols = []
        for x in obj:
            for k in x:
                if k not in cols and not isinstance(x[k], (dict, list)): cols.append(k)
        return table([esc(c) for c in cols], [[fmt(x.get(c)) for c in cols] for x in obj])
    # dict of dict (variant -> stats)
    if isinstance(obj, dict) and obj and all(isinstance(v, dict) for v in obj.values()):
        cols = []
        for v in obj.values():
            for k, x in flat_rows(v):
                if k not in cols: cols.append(k)
        return table(['name'] + [esc(c) for c in cols], [[esc(n)] + [fmt(dict(flat_rows(v)).get(c)) for c in cols] for n, v in obj.items()])
    return table(['key', 'value'], [[esc(k), fmt(v)] for k, v in flat_rows(obj)])


# ------------------------------------------------------------------ discovery
def discover(args):
    variants = []  # (track, variant, dir)
    for tr in TRACKS:
        for d in sorted(glob.glob(os.path.join(ROOT, tr, 'out', '*'))):
            if os.path.isdir(d): variants.append((tr, os.path.basename(d), d))
    if args.dummy:
        for d in sorted(glob.glob(os.path.join(HERE, 'dummy', 'out', '*'))):
            if os.path.isdir(d): variants.append(('dummy', os.path.basename(d), d))
    if args.only_variants:
        keep = set(args.only_variants); variants = [v for v in variants if f'{v[0]}/{v[1]}' in keep]
    return variants


def pick_clips(vdir, manifest, drive_ids, nh, nd):
    have = {f[:-4] for f in os.listdir(vdir) if f.endswith('.mp4')}
    held = [c['id'] for c in manifest.get('clips', []) if c.get('split') == 'heldout' and c['id'] in have]
    dr = [c for c in drive_ids if c in have]
    def spread(l, n):
        if len(l) <= n: return l
        return [l[int(i * len(l) / n)] for i in range(n)]
    # drive: interleave muse and juno
    mu = [c for c in dr if c.startswith('muse')]; ju = [c for c in dr if c.startswith('juno')]
    mixed = []
    for i in range(max(len(mu), len(ju))):
        if i < len(mu): mixed.append(mu[i])
        if i < len(ju): mixed.append(ju[i])
    return spread(held, nh), (mixed[:nd] if mixed else spread(dr, nd))


# ------------------------------------------------------------------ metric tables per variant
MET_COLS = [  # (header, path, digits, better)  better: + higher, - lower
    ('lip open r vs input', 'motion_vs_input.open_corr', 3, '+'), ('lip width r vs input', 'motion_vs_input.width_corr', 3, '+'),
    ('lag ms', 'motion_vs_input.lag_ms', 1, '0'), ('pose MAE deg (centred)', 'motion_vs_input.pose_mae_centered', 2, '-'),
    ('blink F1', 'motion_vs_input.blink.f1', 2, '+'), ('face found', 'face_det_rate.output', 2, '+'),
    ('id cos (within)', 'identity.mean_cos', 3, '+'), ('id cos min', 'identity.min_cos', 3, '+'),
    ('id ceiling (real)', 'identity_real_ceiling.mean_cos', 3, None),
    ('flicker ratio vs real', 'flicker.tdiff_ratio_vs_real', 2, '1'), ('warp err vs real', 'flicker.warp_err_ratio_vs_real', 2, '1'),
    ('PSNR', 'fidelity.psnr', 1, None), ('SSIM', 'fidelity.ssim', 3, None), ('LPIPS', 'fidelity.lpips', 3, None)]
DRV_COLS = [('face found', 'face_det_rate.output', 2), ('lip open r vs render', 'motion_vs_input.open_corr', 3),
            ('pose MAE deg', 'motion_vs_input.pose_mae_centered', 2), ('flicker ratio vs render', 'flicker.tdiff_ratio_vs_render', 2),
            ('warp err out', 'flicker.warp_err_out', 2), ('warp err render', 'flicker.warp_err_render', 2)]


def figure(cid, kind, name, npanel, mlabel, has_aud, arec, caps=None, grid=False, note=None):
    """one comparison row: HTML captions aligned over each panel of the composite video + sound toggle.
    caps: list of (role class, role text, detail). grid=True: 2x2 composite, captions overlaid on each quadrant."""
    if caps is None:
        caps = [('r1', 'Render input', 'the plain avatar render the model is given'), ('r2', 'Model output', mlabel)]
        if npanel == 3: caps.append(('r3', 'Real target', 'the real recording the render was made from'))
        if note is None and npanel == 2: note = 'No real target: this is a live avatar render, so no real recording of it exists.'
    ch = ''.join(f'<div class="cap {c}"><b>{esc(r)}</b><br><span>{esc(t)}</span></div>' for c, r, t in caps)
    if has_aud:
        snd = '<button type="button" class="snd" aria-pressed="false">&#128263; Tap for sound</button>'
    else:
        why = (arec or {}).get('reason') or 'no audio found for this clip'
        snd = f'<span class="noaud" title="{esc(why)}">no audio</span>'
    note = f'<p class="nogt">{esc(note)}</p>' if note else ''
    vid = (f'<video controls preload="none" playsinline loop muted poster="videos/{name}.jpg">'
           f'<source src="videos/{name}.mp4" type="video/mp4"></video>')
    if grid:
        body = f'<div class="gridwrap">{vid}<div class="caps g4">{ch}</div></div>'
    else:
        body = f'<div class="caps n{len(caps)}">{ch}</div>{vid}'
    return (f'<figure class="v w{npanel}">{body}{note}'
            f'<figcaption>{snd}<span class="cid">{esc(cid)} &middot; {esc(kind)}</span></figcaption></figure>')


# ------------------------------------------------------------------ fine-tuned vs not (baselines)
T4469 = __import__('os').path.expandvars('$WORK')


def compare_cfg():
    """baselines/compare.json: {"families":[{"name","track","base","tuned","base_label","tuned_label","speed_base":{"ms","gpu"},"speed_tuned":{..}}],
    "heldout":[ids],"drive":[ids],"notes_html": "..."}"""
    return jload(os.path.join(T4469, 'compare.json'), {}) or {}


def scoreboard(cfg):
    fams = cfg.get('families') or []
    if not fams: return ''
    def res(tr, var): return (jload(os.path.join(HERE, 'results', f'{tr}_{var}.json'), {}) or {}, jload(os.path.join(HERE, 'results', f'judge_{tr}_{var}.json'), {}) or {})
    cols = []; ref = None
    for f in fams:
        for kind in ('base', 'tuned'):
            M, J = res(f['track'], f[kind]); cols.append((f, kind, M, J))
            if kind == 'tuned' and ref is None: ref = (M, J)
    def cell(v, nd): return (f'{v:.{nd}f}' if isinstance(v, (int, float)) else str(v)) if v is not None else '<span class="muted">n/a</span>'
    def row(title, fn, nd, reffn=None, reflab=''):
        r = [title] + [cell(fn(f, kind, M, J), nd) for f, kind, M, J in cols]
        rv = reffn(*ref) if reffn else None
        r.append(f'<span class="muted">{reflab}</span> {cell(rv, nd)}' if rv is not None else '<span class="muted">&ndash;</span>'); return r
    jh = lambda J, k: (J.get('summary_heldout') or {}).get(k)
    jd = lambda J, k: (J.get('summary_drive') or {}).get(k)
    mh = lambda M, k: ((M.get('mean') or {}).get('heldout') or {}).get(k)
    mdv = lambda M, k: ((M.get('mean') or {}).get('drive') or {}).get(k)
    def wins(J):
        s = J.get('summary_heldout') or {}
        return f"{round(s['output_win_rate'] * s['trials'])} of {s['trials']}" if s.get('trials') else None
    rows = [
        row('<b>Realism</b>, held-out people: judge score 1&ndash;10, model vs render', lambda f, k, M, J: jh(J, 'mean_realism_output'), 1, lambda M, J: jh(J, 'mean_realism_render'), 'plain render'),
        row('Realism when the real video is in the line-up (1&ndash;10)', lambda f, k, M, J: jh(J, 'three_way_mean_output'), 1, lambda M, J: jh(J, 'three_way_mean_real'), 'real video'),
        ['Judge picked the model over the plain render'] + [wins(J) or '<span class="muted">n/a</span>' for f, k, M, J in cols] + ['<span class="muted">&ndash;</span>'],
        row('Realism, our avatars (Muse and Juno), 1&ndash;10', lambda f, k, M, J: jd(J, 'mean_realism_output'), 1, lambda M, J: jd(J, 'mean_realism_render'), 'plain render'),
        row('<b>Lip sync</b>, held-out: mouth-opening correlation with the render (1 = perfect)', lambda f, k, M, J: mh(M, 'motion_vs_input.open_corr'), 2, lambda M, J: mh(M, 'motion_ref_real_vs_input.open_corr'), 'real video'),
        row('Lip sync, our avatars', lambda f, k, M, J: mdv(M, 'motion_vs_input.open_corr'), 2),
        row('Blinks kept (F1, held-out)', lambda f, k, M, J: mh(M, 'motion_vs_input.blink.f1'), 2, lambda M, J: mh(M, 'motion_ref_real_vs_input.blink.f1'), 'real video'),
        row('<b>Speed</b>, ms per frame', lambda f, k, M, J: (f.get('speed_' + k) or {}).get('ms'), 1),
        row('Speed, frames per second', lambda f, k, M, J: (1000.0 / f['speed_' + k]['ms']) if (f.get('speed_' + k) or {}).get('ms') else None, 1),
    ]
    h1 = '<tr><th rowspan="2"></th>' + ''.join(f'<th colspan="2" class="fam">{esc(f["name"])}</th>' for f in fams) + '<th rowspan="2">Reference</th></tr>'
    h2 = '<tr>' + ''.join('<th class="nft">not fine-tuned</th><th class="ft">fine-tuned</th>' for f in fams) + '</tr>'
    body = ''.join('<tr>' + ''.join(f'<td class="{"ft" if (i and i <= 2 * len(fams) and i % 2 == 0) else ""}">{c}</td>' for i, c in enumerate(r)) + '</tr>' for r in rows)
    return f'<div class="tw"><table class="num sb"><thead>{h1}{h2}</thead><tbody>{body}</tbody></table></div>'


def grid4(inputs, out_mp4, poster, audio, size=384, fps=24, kbps=620):
    """2x2 composite: top row = inputs[0], inputs[1]; bottom row = inputs[2], inputs[3]. Audio muxed like sbs()."""
    srcs = list(inputs) + ([audio] if audio else [])
    if (os.path.exists(out_mp4) and all(os.path.getmtime(out_mp4) > os.path.getmtime(p) for p in srcs) and os.path.exists(poster)
            and (not audio or has_audio_stream(out_mp4))):
        return True
    cmd = ['ffmpeg', '-v', 'error', '-y']
    for p in srcs: cmd += ['-i', p]
    parts = [f"[{i}:v]fps={fps},scale={size}:{size}:flags=area,setsar=1[v{i}]" for i in range(4)]
    parts.append('[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[o]')
    amap = ['-map', '4:a:0', '-c:a', 'copy'] if audio else ['-an']
    cmd += ['-filter_complex', ';'.join(parts), '-map', '[o]'] + amap + ['-t', '8.05', '-c:v', 'libx264', '-preset', 'slow',
            '-b:v', f'{kbps}k', '-maxrate', f'{int(kbps * 1.6)}k', '-bufsize', f'{kbps * 2}k', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out_mp4]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0: print('ffmpeg failed', out_mp4, r.stderr[-300:], file=sys.stderr); return False
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', '2.5', '-i', out_mp4, '-frames:v', '1', '-q:v', '5', poster], capture_output=True)
    return os.path.exists(poster)


def head_to_head(cfg, vid_dir, page_videos, no_videos=False):
    out = ''
    for f in cfg.get('families') or []:
        tr = f['track']; bdir = os.path.join(ROOT, tr, 'out', f['base']); tdir = os.path.join(ROOT, tr, 'out', f['tuned'])
        figs = ''
        for cid in cfg.get('heldout') or []:
            base = f'{ROOT}/data/pairs/{audio_id(cid)}'
            ins = [f'{base}/render.mp4', f'{base}/real.mp4', f'{bdir}/{cid}.mp4', f'{tdir}/{cid}.mp4']
            if not all(os.path.exists(p) for p in ins): continue
            name = f'h2h_{tr}_{f["base"]}__{cid}'; aud, arec = audio_for(cid)
            if not grid4(ins, f'{vid_dir}/{name}.mp4', f'{vid_dir}/{name}.jpg', aud): continue
            page_videos.append((name, cid, arec))
            caps = [('r1', 'Render input', 'what the model is given'), ('r3', 'Real target', 'the real recording'),
                    ('r4', 'Model, NOT fine-tuned', f['base_label']), ('r2', 'Model, fine-tuned', f['tuned_label'])]
            figs += figure(cid, 'held-out pair', name, 4, '', has_audio_stream(f'{vid_dir}/{name}.mp4'), arec, caps=caps, grid=True)
        for cid in cfg.get('drive') or []:
            ins = [f'{ROOT}/drive/{cid}.mp4', f'{bdir}/{cid}.mp4', f'{tdir}/{cid}.mp4']
            if not all(os.path.exists(p) for p in ins): continue
            name = f'h2h_{tr}_{f["base"]}__{cid}'; aud, arec = audio_for(cid)
            if not sbs(ins, None, f'{vid_dir}/{name}.mp4', f'{vid_dir}/{name}.jpg', audio=aud): continue
            page_videos.append((name, cid, arec))
            caps = [('r1', 'Render input', 'what the model is given'), ('r4', 'Model, NOT fine-tuned', f['base_label']), ('r2', 'Model, fine-tuned', f['tuned_label'])]
            figs += figure(cid, 'live avatar render', name, 3, '', has_audio_stream(f'{vid_dir}/{name}.mp4'), arec, caps=caps,
                           note='No real target: this is a live avatar render, so no real recording of it exists.')
        out += f'<section class="card"><h3>{esc(f["name"])}: not fine-tuned vs fine-tuned</h3><div class="vids">{figs}</div></section>'
    return out


def build(args):
    out = args.out; vid_dir = os.path.join(out, 'videos'); os.makedirs(vid_dir, exist_ok=True)
    manifest = jload(os.path.join(ROOT, 'data', 'manifest.json'), {}) or {}
    if args.fake_heldout:  # pipeline test only
        for c in manifest.get('clips', []):
            if c['id'] in args.fake_heldout: c['split'] = 'heldout'
    drive_ids = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(ROOT, 'drive', '*.mp4')))
    variants = discover(args)
    pair_ids = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(ROOT, 'data', 'pairs', '*', 'meta.json')))
    load_audio_index(pair_ids + drive_ids)
    page_videos = []  # (name, cid, audio record) for the audio check at the end
    cards = []; mrows_h = []; mrows_d = []; jrows = []; detail = []
    for tr, var, vdir in variants:
        tag = f'{tr}/{var}'; slug = f'{tr}_{var}'
        mlabel, tuned = model_label(tr, var)
        held, dr = pick_clips(vdir, manifest, drive_ids, args.heldout_per_variant, args.drive_per_variant)
        vids = []
        if tag in set(args.table_only_variants or []): held, dr = [], []
        if not args.no_videos:
            for cid in held:
                base = f'{ROOT}/data/pairs/{audio_id(cid)}'
                ins = [f'{base}/render.mp4', os.path.join(vdir, cid + '.mp4'), f'{base}/real.mp4']
                if not all(os.path.exists(p) for p in ins): continue
                name = f'{slug}__{cid}'; aud, arec = audio_for(cid)
                if sbs(ins, None, f'{vid_dir}/{name}.mp4', f'{vid_dir}/{name}.jpg', audio=aud):
                    vids.append((cid, 'held-out pair', name, 3)); page_videos.append((name, cid, arec))
            for cid in dr:
                ins = [f'{ROOT}/drive/{cid}.mp4', os.path.join(vdir, cid + '.mp4')]
                if not all(os.path.exists(p) for p in ins): continue
                name = f'{slug}__{cid}'; aud, arec = audio_for(cid)
                if sbs(ins, None, f'{vid_dir}/{name}.mp4', f'{vid_dir}/{name}.jpg', audio=aud):
                    vids.append((cid, 'live avatar render', name, 2)); page_videos.append((name, cid, arec))
        else:
            for f in sorted(glob.glob(f'{vid_dir}/{slug}__*.mp4')):
                n = os.path.basename(f)[:-4]; c = n.split('__')[1]
                vids.append((c, 'live avatar render' if c in drive_ids else 'held-out pair', n, 2 if c in drive_ids else 3))
                page_videos.append((n, c, AUDIO_IDX.get(audio_id(c))))
        vh = ''
        for cid, kind, name, np_ in vids:
            vh += figure(cid, kind, name, np_, mlabel, os.path.exists(f'{vid_dir}/{name}.mp4') and has_audio_stream(f'{vid_dir}/{name}.mp4'),
                         AUDIO_IDX.get(audio_id(cid)))
        ft = {True: 'fine-tuned', False: 'no fine-tune (zero-shot)', None: ''}[tuned]
        if tag not in set(args.table_only_variants or []): cards.append(f'<section class="card"><h3>{esc(mlabel)}</h3><p class="vtag"><code>{esc(tag)}</code>'
                     f'{" &middot; " + ft if ft else ""}</p>'
                     f'<div class="vids">{vh or "<p class=muted>No side-by-side videos yet.</p>"}</div></section>')

        M = jload(os.path.join(HERE, 'results', f'{slug}.json'))
        if M:
            mh = M.get('mean', {}).get('heldout') or {}; md_ = M.get('mean', {}).get('drive') or {}; ma = M.get('mean', {}).get('all') or {}
            src = mh if mh.get('n') else ma
            mrows_h.append(([esc(tag), fmt(src.get('n'), 0)] + [fmt(src.get(c[1]), c[2]) for c in MET_COLS]))
            if md_.get('n'): mrows_d.append([esc(tag), fmt(md_.get('n'), 0)] + [fmt(md_.get(c[1]), c[2]) for c in DRV_COLS])
            errs = [(k, v.get('error')) for k, v in M.get('clips', {}).items() if v.get('error')]
            if errs: detail.append(f'<p class="muted">{esc(tag)}: {len(errs)} clip(s) not scored: ' + esc(', '.join(f'{k} ({e})' for k, e in errs[:6])) + '</p>')
        else:
            mrows_h.append([esc(tag), '0'] + ['<span class="muted">n/a</span>'] * len(MET_COLS))
        J = jload(os.path.join(HERE, 'results', f'judge_{slug}.json'))
        if J:
            for lab, key in (('held-out', 'summary_heldout'), ('drive', 'summary_drive'), ('all', 'summary')):
                s = J.get(key) or {}
                if not s.get('trials'): continue
                ci = s.get('output_win_ci95_wilson') or [None, None]
                jrows.append([esc(tag), lab, fmt(s.get('clips'), 0), fmt(s.get('trials'), 0),
                              f"<b>{fmt(s.get('output_win_rate'), 2)}</b> <span class=muted>[{fmt(ci[0], 2)}, {fmt(ci[1], 2)}]</span>",
                              fmt(s.get('mean_realism_render'), 1), fmt(s.get('mean_realism_output'), 1), fmt(s.get('three_way_mean_real'), 1),
                              fmt(s.get('mean_same_pose_any'), 1), fmt(s.get('order_consistency'), 2)])
        else:
            jrows.append([esc(tag), '-', '0', '0'] + ['<span class="muted">n/a</span>'] * 6)

    cfg = compare_cfg()
    h2h = head_to_head(cfg, vid_dir, page_videos)
    sb = scoreboard(cfg)
    cards.append(h2h)
    if not args.no_videos:  # prune videos from variants/clips no longer on the page
        keep = {n for c in cards for n in re.findall(r'videos/([^"]+)\.mp4', c)}
        for f in glob.glob(f'{vid_dir}/*'):
            if os.path.splitext(os.path.basename(f))[0] not in keep: os.remove(f)
    summary = md(tload(os.path.join(HERE, 'summary.md'))) or '<p class="muted">Summary not written yet.</p>'
    mt_h = table(['variant', 'clips'] + [f'{c[0]} <span class="arr">{ {"+":"&uarr;","-":"&darr;","1":"&rarr;1","0":"&rarr;0"}.get(c[3], "") }</span>' for c in MET_COLS], mrows_h, 'num') if mrows_h else '<p class="muted">No variants yet.</p>'
    mt_d = table(['variant', 'clips'] + [c[0] for c in DRV_COLS], mrows_d, 'num') if mrows_d else '<p class="muted">No drive-clip metrics yet.</p>'
    jt = table(['variant', 'set', 'clips', 'trials', 'output judged more real (win rate [95% CI])', 'realism render', 'realism output',
                'realism real', 'same pose/mouth (1-10)', 'both orders agree'], jrows, 'num') if jrows else '<p class="muted">No judge results yet.</p>'
    lat = ''
    for tr in TRACKS:
        lat += f'<h4>{tr}</h4>' + kv_table(jload(os.path.join(ROOT, tr, 'latency.json')))
    sp = ''; total = 0.0; any_sp = False
    for tr in TRACKS:
        s = jload(os.path.join(ROOT, tr, 'spend.json'))
        sp += f'<h4>{tr}</h4>' + kv_table(s)
        if isinstance(s, dict) and isinstance(s.get('usd'), (int, float)): total += s['usd']; any_sp = True
    if any_sp: sp += f'<p><b>Total GPU spend: ${total:.2f}</b></p>'
    dummy_note = '<p class="warn">DUMMY variants included (--dummy): not real model output.</p>' if args.dummy else ''

    cards.pop()
    page = TEMPLATE.format(sb=sb, h2h=h2h, cmpnotes=cfg.get('notes_html', ''), summary=summary, cards=''.join(cards) or '<p class="muted">No model outputs yet.</p>', mt_h=mt_h, mt_d=mt_d,
                           detail=''.join(detail), jt=jt, lat=lat, sp=sp, worked=section_text('worked.md'), didnt=section_text('didnt.md'),
                           nxt=section_text('next.md'), made=time.strftime('%Y-%m-%d %H:%M'), dummy=dummy_note)
    open(os.path.join(out, 'index.html'), 'w').write(page)
    tot = sum(os.path.getsize(f) for f in glob.glob(f'{vid_dir}/*.mp4'))
    print(f'wrote {out}/index.html; {len(glob.glob(vid_dir + "/*.mp4"))} videos, {tot / 1e6:.1f} MB')
    # every page video must carry its clip's original audio; only a source with no audio at all is exempt
    bad = []; exempt = []
    for name, cid, arec in page_videos:
        if has_audio_stream(f'{vid_dir}/{name}.mp4'): continue
        why = (arec or {}).get('reason') or 'no entry in the audio index'
        (exempt if (arec and not arec.get('has_audio') and why.startswith(EXEMPT_REASONS)) else bad).append((name, why))
    print(f'audio: {len(page_videos) - len(bad) - len(exempt)}/{len(page_videos)} page videos carry audio')
    for n, w in exempt: print(f'  exempt (source has no audio): {n}: {w}')
    if bad:
        print('\n' + '!' * 70 + f'\nWARNING: {len(bad)} page video(s) have NO audio stream:', file=sys.stderr)
        for n, w in bad: print(f'  {n}: {w}', file=sys.stderr)
        print('!' * 70, file=sys.stderr)
        if not args.allow_missing_audio: sys.exit(2)


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Render to photoreal PoC</title>
<style>
:root{{--bg:#0e1116;--panel:#171b23;--line:#2c3340;--fg:#f2f4f8;--mut:#a8b1c0;--acc:#6cb6ff;--acc2:#ffb454}}
*{{box-sizing:border-box}}html{{-webkit-text-size-adjust:100%}}
body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:16px 14px 60px}}
h1{{font-size:1.5rem;margin:.6em 0 .2em}}h2{{font-size:1.25rem;margin:1.8em 0 .5em;padding-bottom:.25em;border-bottom:2px solid var(--acc)}}
h3{{font-size:1.1rem;margin:.2em 0 .6em}}h4,h5{{margin:1em 0 .3em;color:var(--acc2)}}
a{{color:var(--acc)}}code{{background:#232a36;padding:1px 5px;border-radius:4px}}
.muted{{color:var(--mut)}}.warn{{border:2px solid var(--acc2);padding:8px 10px;border-radius:8px;color:var(--acc2)}}
.sep{{color:var(--mut)}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px;margin:12px 0}}
.vids{{display:grid;grid-template-columns:repeat(auto-fill,minmax(460px,1fr));gap:12px}}
.v{{margin:0}}.v.w3{{grid-column:span 1}}
video{{width:100%;height:auto;background:#000;border-radius:6px;display:block}}
figcaption{{font-size:.85rem;color:var(--mut);padding-top:5px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}}
.caps{{display:grid;gap:0;margin-bottom:3px}}.caps.n3{{grid-template-columns:1fr 1fr 1fr}}.caps.n2{{grid-template-columns:1fr 1fr}}
.cap{{font-size:.78rem;line-height:1.25;padding:4px 5px;border-top:3px solid #5b6577;background:#11151c;color:var(--mut);overflow-wrap:anywhere}}
.cap b{{color:var(--fg);font-weight:700}}
.cap.r2{{border-top-color:var(--acc2)}}.cap.r2 b{{color:var(--acc2)}}
.cap.r1{{border-top-color:var(--acc)}}.cap.r1 b{{color:var(--acc)}}
.cap.r3{{border-top-color:#7ee2a8}}.cap.r3 b{{color:#7ee2a8}}
.cap.r4{{border-top-color:#d6a2ff}}.cap.r4 b{{color:#d6a2ff}}
.gridwrap{{position:relative}}.caps.g4{{position:absolute;inset:0;grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;pointer-events:none;margin:0}}
.caps.g4 .cap{{align-self:start;justify-self:start;background:rgba(10,13,18,.82);border-top-width:3px;border-radius:0 0 6px 0;max-width:96%}}
.caps.g4 .cap span{{color:#d5dbe6}}
table.sb th.fam{{text-align:center;border-left:1px solid var(--line)}}table.sb th.nft{{color:#d6a2ff}}table.sb th.ft{{color:var(--acc2)}}
table.sb td.ft{{font-weight:700;color:var(--fg)}}table.sb td:first-child{{min-width:190px}}
.nogt{{font-size:.8rem;color:var(--mut);margin:3px 0 0;font-style:italic}}
.snd{{font:inherit;font-size:.9rem;font-weight:600;color:var(--fg);background:#26324a;border:1px solid var(--acc);border-radius:18px;
  padding:7px 14px;min-height:40px;cursor:pointer;touch-action:manipulation}}
.snd[aria-pressed=true]{{background:var(--acc2);color:#111;border-color:var(--acc2)}}
.noaud{{font-size:.75rem;border:1px solid var(--mut);border-radius:10px;padding:1px 7px}}
.vtag{{margin:-.4em 0 .6em;font-size:.85rem;color:var(--mut)}}
.roles{{font-size:.92rem;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:7px 10px}}
.roles b.r1{{color:var(--acc)}}.roles b.r2{{color:var(--acc2)}}.roles b.r3{{color:#7ee2a8}}
.tw{{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:8px 0;border:1px solid var(--line);border-radius:8px}}
table{{border-collapse:collapse;width:100%;font-size:.9rem}}
th,td{{padding:6px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}
th{{background:#202734;position:sticky;top:0;font-weight:600}}
table.num td:not(:first-child){{text-align:right;font-variant-numeric:tabular-nums}}
.arr{{color:var(--acc2)}}
.legend{{font-size:.9rem;color:var(--mut)}}
@media (max-width:520px){{.vids{{grid-template-columns:1fr}}body{{font-size:15px}}}}
</style></head><body><main>
<h1>Render to photoreal: proof of concept</h1>
<p class="muted">Generated {made}. Everything on this page is a 384x384 low-resolution test. Videos load on tap.</p>
<p class="roles">Each video shows, left to right: <b class="r1">Render input</b> (the plain avatar render the model gets),
<b class="r2">Model output</b> (what the model made from it) and <b class="r3">Real target</b> (the real recording; live avatar clips have none).
Videos start muted: tap <b>Tap for sound</b> under a video to hear its original voice track (one video plays sound at a time).</p>
{dummy}
<h2>Summary</h2><div class="card">{summary}</div>

<h2>Fine-tuned vs not fine-tuned</h2>
<p class="legend">The same base models with no training of ours ("not fine-tuned", zero-shot) next to our fine-tuned versions, on the same 12 held-out clips and 8 live avatar clips,
scored by the same metrics and the same blind judge. The Reference column is the plain render for realism and the real recording where one exists.</p>
{sb}
{cmpnotes}
<p class="legend">In the four-panel videos: top row is <b class="r1" style="color:var(--acc)">Render input</b> and <b style="color:#7ee2a8">Real target</b>;
bottom row is the model <b style="color:#d6a2ff">not fine-tuned</b> and <b style="color:var(--acc2)">fine-tuned</b>. Tap for sound to judge lip sync by ear.</p>
{h2h}

<h2>Side-by-side videos, per model</h2>
<p class="legend">Left to right: plain render (the model's input), model output, and for held-out pairs the real recording the render was made from.
Held-out pairs were not used for training. Each video carries the clip's original audio, cut from the source recording at the same 8 s:
for pairs the speaker's own close-talk microphone, for live avatar clips the session recording (avatar voice and user microphone mixed).</p>
{cards}

<h2>Metrics</h2>
<p class="legend">Arrows: &uarr; higher is better, &darr; lower is better, &rarr;1 closest to 1 is best (flicker ratio 1 = as steady as real video), &rarr;0 closest to 0.
"vs input" compares the output to the render it came from (does the model keep lips, pose, blinks).
"id cos" is the face-embedding similarity of each frame to the clip's own mean face (the "id ceiling" is the same number on the real video).
PSNR/SSIM/LPIPS vs the real frame are context only: the mapping from render to a real person is one-to-many.
Held-out pairs, mean over clips.</p>
{mt_h}
<h4>Live avatar renders (Muse and Juno, no ground truth)</h4>
{mt_d}
{detail}

<h2>Blind judge</h2>
<p class="legend">Claude (Sonnet) sees two strips of 6 frames, the plain render and the model output, in random order under neutral file names,
and picks which looks more like real camera footage; each clip is judged in both orders. A win rate of 0.5 means no preference; 1.0 means the output was always chosen.
Realism scores are 1 to 10. "realism real" is the ceiling from the three-way ranking that includes the real video.</p>
{jt}

<h2>Latency</h2>{lat}
<h2>Spend</h2>{sp}

<h2>What worked</h2><div class="card">{worked}</div>
<h2>What did not</h2><div class="card">{didnt}</div>
<h2>Next step</h2><div class="card">{nxt}</div>
</main>
<script>
(function(){{
  var vids=[].slice.call(document.querySelectorAll('figure.v video'));
  function btnOf(v){{return v.closest('figure').querySelector('button.snd');}}
  function setBtn(v){{var b=btnOf(v); if(!b) return; var on=!v.muted&&!v.paused;
    b.setAttribute('aria-pressed',on?'true':'false'); b.innerHTML=on?'&#128266; Sound on (tap to mute)':'&#128263; Tap for sound';}}
  function silenceOthers(keep){{vids.forEach(function(o){{if(o!==keep&&!o.muted){{o.muted=true;setBtn(o);}}}});}}
  vids.forEach(function(v){{
    var b=btnOf(v);
    if(b) b.addEventListener('click',function(){{
      if(!v.muted&&!v.paused){{v.muted=true;setBtn(v);return;}}
      silenceOthers(v); v.muted=false; v.volume=1;
      var p=v.play(); if(p&&p.catch) p.catch(function(){{}}); setBtn(v);
    }});
    v.addEventListener('volumechange',function(){{if(!v.muted) silenceOthers(v); setBtn(v);}});
    v.addEventListener('play',function(){{if(!v.muted) silenceOthers(v); setBtn(v);}});
    v.addEventListener('pause',function(){{setBtn(v);}});
  }});
}})();
</script>
</body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=__import__('os').path.expandvars('$OUT_DIR/render2real-poc'))
    ap.add_argument('--heldout-per-variant', type=int, default=3); ap.add_argument('--drive-per-variant', type=int, default=3)
    ap.add_argument('--dummy', action='store_true', help='include eval/dummy/out/* variants (pipeline test)')
    ap.add_argument('--fake-heldout', nargs='*', help='test only: treat these clip ids as held-out')
    ap.add_argument('--no-videos', action='store_true'); ap.add_argument('--allow-missing-audio', action='store_true'); ap.add_argument('--only-variants', nargs='*')
    ap.add_argument('--table-only-variants', nargs='*', help='track/variant that get metric + judge rows but no video card')
    build(ap.parse_args())


if __name__ == '__main__': main()
