#!/usr/bin/env python3
"""Cut the original 8.0 s audio for every pair clip and drive clip -> baselines/audio/<clip_id>.m4a + index.json.
  cut_audio.py [--offset-ms N] [--only id ...]
Pairs: the Seamless source mp4's own AAC track (48 kHz stereo container, voice on the left channel only;
it is the same signal as the participant's close-talk wav, see index 'src_check'), from src_start_s, mono AAC 96k.
Drive: the avatar session recording's AAC track (mono mix of the user's mic + the avatar's voice), from start_s.
"""
import argparse, glob, json, os, re, subprocess, sys
import numpy as np

ROOT = __import__('os').path.expandvars('$WORK')
OUT = os.path.dirname(os.path.abspath(__file__))
DS = __import__('os').path.expandvars('$SEAMLESS_ROOT/naturalistic/dev')
DUR = 8.0
SILENT_DB = -80.0
MEAN_TARGET = -22.0; GAIN_CAP = 30.0
QUIET_DB = -60.0  # below this mean level there is no speech, just the noise floor: kept as is, never amplified  # mean volume below this = digital silence for our purposes


def pcm(path, t0, dur, ch=None, sr=16000):
    cmd = ['ffmpeg', '-v', 'error', '-ss', f'{t0:.4f}', '-t', f'{dur:.4f}', '-i', path, '-map', '0:a:0']
    cmd += (['-af', f'pan=mono|c0=c{ch}'] if ch is not None else ['-ac', '1'])
    cmd += ['-ar', str(sr), '-f', 'f32le', '-']
    r = subprocess.run(cmd, capture_output=True)
    return np.frombuffer(r.stdout, np.float32)


def db(x): return float(20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)) if len(x) else -240.0


def env(x, hop=160): k = len(x) // hop; return np.sqrt((x[:k * hop].reshape(k, hop) ** 2).mean(1) + 1e-12)


def ecorr(a, b):
    n = min(len(a), len(b))
    if n < 10: return None
    a = np.log(env(a[:n])); b = np.log(env(b[:n])); return float(np.corrcoef(a, b)[0, 1])


def probe_audio(path):
    r = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a', '-show_entries', 'stream=codec_name,channels,sample_rate',
                        '-of', 'json', path], capture_output=True, text=True)
    try: return (json.loads(r.stdout).get('streams') or [None])[0]
    except Exception: return None


def volume(path):
    r = subprocess.run(['ffmpeg', '-v', 'info', '-i', path, '-map', '0:a:0', '-af', 'volumedetect', '-f', 'null', '-'], capture_output=True, text=True)
    m = re.search(r'mean_volume: (-?[\d.]+|-inf) dB', r.stderr); x = re.search(r'max_volume: (-?[\d.]+|-inf) dB', r.stderr)
    f = lambda s: -240.0 if (s is None or s.group(1) == '-inf') else float(s.group(1))
    return f(m), f(x)


def cut(src, t0, out, chans, nch, gain_db=0.0):
    if chans is None or (nch == 1): af = ['-ac', '1']
    elif len(chans) == 1: af = ['-af', f'pan=mono|c0=c{chans[0]}']
    else: af = ['-ac', '1']
    if gain_db:  # static gain only (no compression / no timing change)
        if af[0] == '-af': af[1] += f',volume={gain_db:.2f}dB'
        else: af += ['-af', f'volume={gain_db:.2f}dB']
    cmd = ['ffmpeg', '-v', 'error', '-y', '-ss', f'{t0:.4f}', '-t', f'{DUR:.4f}', '-i', src, '-map', '0:a:0'] + af + \
          ['-ar', '48000', '-c:a', 'aac', '-b:a', '96k', '-movflags', '+faststart', out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0, r.stderr[-300:]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--offset-ms', type=float, default=0.0); ap.add_argument('--only', nargs='*')
    a = ap.parse_args()
    idx_p = os.path.join(OUT, 'index.json'); idx = json.load(open(idx_p)) if os.path.exists(idx_p) else {}
    jobs = []
    for mp in sorted(glob.glob(f'{ROOT}/data/pairs/*/meta.json')):
        m = json.load(open(mp)); jobs.append(('pair', m['id'], m))
    src = json.load(open(f'{ROOT}/drive/sources.json'))
    for cid in sorted(src): jobs.append(('drive', cid, src[cid]))
    for kind, cid, m in jobs:
        if a.only and cid not in a.only: continue
        out = os.path.join(OUT, cid + '.m4a'); rec = {'path': out, 'kind': kind}
        if kind == 'pair':
            s = m['src_video']; t0 = (m['src_start_frame'] / 30.0) if 'src_start_frame' in m else float(m['src_start_s'])
        else:
            s = m['source']; t0 = float(m['start_s'])
        t0 += a.offset_ms / 1000.0
        rec.update(src_file=s, start_s=round(t0, 4), duration_s=DUR, offset_ms_applied=a.offset_ms)
        st = probe_audio(s)
        if not st:
            rec.update(has_audio=False, reason='source has no audio track', mean_volume_db=None, source=s)
            if os.path.exists(out): os.remove(out)
            idx[cid] = rec; print(cid, 'NO AUDIO TRACK'); continue
        nch = int(st.get('channels') or 1); chans = None
        if kind == 'pair':
            lv = [db(pcm(s, t0, DUR, ch=c)) for c in range(nch)]
            chans = [c for c in range(nch) if lv[c] > SILENT_DB] or list(range(nch))
            rec['channel_db'] = [round(x, 1) for x in lv]
            # which participant is it? compare with the close-talk wavs of this session
            fid = m['file_id']; sess, pid = fid.rsplit('_', 1)
            mono = pcm(s, t0, DUR, ch=chans[0])
            chk = {}
            for w in sorted(glob.glob(f'{DS}/audio/{sess}_*.wav')):
                p = os.path.basename(w)[:-4].rsplit('_', 1)[1]
                chk[('own ' if p == pid else 'partner ') + p] = round(ecorr(mono, pcm(w, t0, DUR)) or float('nan'), 3)
            rec['src_check'] = {'env_corr_with_closetalk_wav': chk}
            rec['source'] = (f'Seamless source mp4 AAC track (channel {"+".join(map(str, chans))} of {nch}; '
                             f'= participant {pid} close-talk mic), from {t0:.3f} s')
        else:
            rec['source'] = (f'avatar session recording AAC (mono mix of the user mic + the avatar voice as heard, '
                             f'per recorder.js), from {t0:.3f} s; character {m.get("character")}')
            sj = json.load(open(s[:-4] + '.json')) if os.path.exists(s[:-4] + '.json') else {}
            rec['session'] = {k: sj.get(k) for k in ('voice_label', 'automated', 'ua', 'duration_s') if k in sj}
        ok, err = cut(s, t0, out, chans, nch)
        if not ok:
            rec.update(has_audio=False, reason='ffmpeg cut failed: ' + err, mean_volume_db=None); idx[cid] = rec; print(cid, 'FAIL', err); continue
        mv, mx = volume(out)
        rec.update(src_mean_volume_db=mv, src_max_volume_db=mx)
        # level for listening on a phone: static gain toward MEAN_TARGET, peaks kept <= -1 dBFS, at most +GAIN_CAP dB
        gain = 0.0 if mv <= QUIET_DB else max(0.0, min(MEAN_TARGET - mv, -1.0 - mx, GAIN_CAP))
        if gain >= 0.5:
            ok, err = cut(s, t0, out, chans, nch, gain)
            mv, mx = volume(out)
        rec.update(gain_db=round(gain, 2), mean_volume_db=mv, max_volume_db=mx)
        if rec['src_mean_volume_db'] <= SILENT_DB:
            rec.update(has_audio=False, reason=f'digital silence (mean {rec["src_mean_volume_db"]} dB, max {rec["src_max_volume_db"]} dB)'); os.remove(out)
        else:
            rec['has_audio'] = True
            if rec['src_mean_volume_db'] < QUIET_DB: rec['note'] = f'no speech in this stretch: the source track sits at its noise floor (mean {rec["src_mean_volume_db"]} dB, max {rec["src_max_volume_db"]} dB); kept unamplified'
        idx[cid] = rec; print(cid, rec['has_audio'], mv, rec.get('src_check', ''))
    json.dump(idx, open(idx_p, 'w'), indent=1)
    print('wrote', idx_p, sum(1 for v in idx.values() if v.get('has_audio')), 'with audio of', len(idx))


if __name__ == '__main__': main()
