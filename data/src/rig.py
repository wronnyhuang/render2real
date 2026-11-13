"""tracks/<id>.npz (+ optional calib/<id>.json from Sonnet) -> rig/<id>.npz : renderer-ready, camera-space meters.
Camera: OpenGL convention (x right, y up, looks down -z), at origin, vertical FOV 63 deg, image W x H."""
import sys, json, os, numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as Rot
VFOV = 63.0
CH = ['eye_open_L', 'eye_open_R', 'gaze_yaw', 'gaze_pitch', 'brow_raise_L', 'brow_raise_R', 'brow_inner',
      'mouth_open', 'mouth_width', 'smile', 'frown', 'pucker', 'jaw_open', 'cheek_squint']
# person's LEFT side = image right for a camera-facing person. MediaPipe indices:
EYE_L = dict(c0=362, c1=263, up=386, lo=374, iris=473, brow=[334, 296, 336, 285, 295, 282])   # subject's left eye
EYE_R = dict(c0=33, c1=133, up=159, lo=145, iris=468, brow=[105, 66, 107, 55, 65, 52])
LIPS_OUT = [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185]
LIPS_IN = [78,95,88,178,87,14,317,402,318,324,308,415,310,311,312,13,82,81,80,191]
POSE_J = [11, 12, 13, 14, 15, 16, 23, 24]   # L_sh R_sh L_el R_el L_wr R_wr L_hip R_hip (person's sides)

def fill(a, valid):
    """linear-interpolate invalid frames along axis 0 (hold at the ends)."""
    a = a.copy(); T = len(a); idx = np.arange(T)
    if valid.sum() == 0: return a
    flat = a.reshape(T, -1)
    for k in range(flat.shape[1]):
        flat[~valid, k] = np.interp(idx[~valid], idx[valid], flat[valid, k])
    return flat.reshape(a.shape)

def smooth(a, s):
    return gaussian_filter1d(a, s, axis=0, mode='nearest') if s > 0 else a

def build(track_path, calib=None):
    d = np.load(track_path, allow_pickle=False)
    W, H, fps = int(d['W']), int(d['H']), float(d['fps'])
    fpx = (H / 2) / np.tan(np.radians(VFOV / 2))
    P, T = d['face_lm'].shape[:2]
    bnames = [str(b) for b in d['bnames']]
    out = dict(W=W, H=H, fps=fps, vfov_deg=VFOV, P=P, T=T, channels=np.array(CH), blend_names=np.array(bnames))
    head_M = np.zeros((P, T, 4, 4), np.float32); head_valid = np.zeros((P, T), bool)
    chans = np.zeros((P, T, len(CH)), np.float32); blend = np.zeros((P, T, 52), np.float32)
    lips_o = np.zeros((P, T, 20, 2), np.float32); lips_i = np.zeros((P, T, 20, 2), np.float32)
    mouth_c = np.zeros((P, T, 3), np.float32); face_local = np.zeros((P, T, 478, 3), np.float32)
    joints = np.zeros((P, T, 8, 3), np.float32); jvis = np.zeros((P, T, 8), np.float32)
    hands = np.zeros((P, T, 2, 21, 3), np.float32); hvalid = np.zeros((P, T, 2), bool)
    stats = []
    for p in range(P):
        cal = (calib or {}).get(str(p), {}) if calib else {}
        M = d['face_mat'][p].astype(np.float64); lm = d['face_lm'][p].astype(np.float64)
        v = ~np.isnan(M[:, 0, 0]) & ~np.isnan(lm[:, 0, 0]); head_valid[p] = v
        if v.sum() < 5: stats.append({}); continue
        M = fill(M, v); lm = fill(lm, v); bl = fill(d['blend'][p].astype(np.float64), v)
        # --- head pose: smooth translation + rotation (quaternion), cm -> m
        tr = smooth(M[:, :3, 3], 1.5)
        q = Rot.from_matrix(M[:, :3, :3]).as_quat()
        for t in range(1, T):
            if np.dot(q[t], q[t-1]) < 0: q[t] = -q[t]
        q = smooth(q, 1.5); q /= np.linalg.norm(q, axis=1, keepdims=True)
        R = Rot.from_quat(q).as_matrix()
        # optional Sonnet pose corrections (degrees, applied in head frame)
        if cal.get('head_pitch_offset_deg') or cal.get('head_yaw_offset_deg') or cal.get('head_roll_offset_deg'):
            cl = lambda k: float(np.clip(cal.get(k, 0) or 0, -10, 10))
            C = Rot.from_euler('xyz', [-cl('head_pitch_offset_deg'), cl('head_yaw_offset_deg'), cl('head_roll_offset_deg')],
                               degrees=True).as_matrix()
            R = R @ C
        Mh = np.zeros((T, 4, 4)); Mh[:, :3, :3] = R; Mh[:, :3, 3] = tr / 100.0; Mh[:, 3, 3] = 1
        head_M[p] = Mh
        # --- unproject landmarks -> camera cm -> head-local m
        dep = -M[:, 2, 3][:, None] + lm[:, :, 2] * W * (-M[:, 2, 3][:, None] / fpx)
        X = (lm[:, :, 0] * W - W / 2) * dep / fpx; Y = -(lm[:, :, 1] * H - H / 2) * dep / fpx; Z = -dep
        Pc = np.stack([X, Y, Z, np.ones_like(X)], -1)
        Minv = np.linalg.inv(M)
        loc = np.einsum('tij,tnj->tni', Minv, Pc)[:, :, :3] / 100.0
        loc = smooth(loc, 0.8); face_local[p] = loc
        def eye(e):
            w = np.linalg.norm(loc[:, e['c0']] - loc[:, e['c1']], axis=1)
            gap = np.linalg.norm(loc[:, e['up']] - loc[:, e['lo']], axis=1) / w
            mid = (loc[:, e['c0']] + loc[:, e['c1']]) / 2
            ir = (loc[:, e['iris']] - mid) / w[:, None]
            browh = (loc[:, e['brow']].mean(1)[:, 1] - mid[:, 1]) / w
            return w, gap, ir, browh
        wl, gl, il, bl_ = eye(EYE_L); wr, gr, irr, br_ = eye(EYE_R)
        # eye openness: normalize by the clip's open-eye level (p85) unless Sonnet gives thresholds
        def eopen(g, side):
            c = cal.get('eye', {})
            closed = c.get('closed_ratio', np.percentile(g, 85) * 0.35); open_ = c.get('open_ratio', np.percentile(g, 85))
            return np.clip((g - closed) / max(open_ - closed, 1e-4), 0, 1.4)
        eoL, eoR = eopen(gl, 'L'), eopen(gr, 'R')
        gz = (il + irr) / 2
        gz = gz - np.median(gz, 0) * cal.get('gaze_center_frac', 1.0)   # assume median gaze ~ forward (Sonnet can disable)
        ggain = cal.get('gaze_gain', 2.2)
        gyaw = np.clip(gz[:, 0] * ggain, -0.6, 0.6); gpitch = np.clip(gz[:, 1] * ggain * 1.4, -0.5, 0.5)
        bgain = cal.get('brow_gain', 8.0)
        bL = np.clip((bl_ - np.median(bl_)) * bgain, -1, 1); bR = np.clip((br_ - np.median(br_)) * bgain, -1, 1)
        bi = bl[:, bnames.index('browInnerUp')]
        # mouth (metric)
        mo = np.linalg.norm(loc[:, 13] - loc[:, 14], axis=1) * cal.get('mouth_open_gain', 1.0)
        mw = np.linalg.norm(loc[:, 61] - loc[:, 291], axis=1)
        cy = (loc[:, 61, 1] + loc[:, 291, 1]) / 2 - (loc[:, 0, 1] + loc[:, 17, 1]) / 2
        cyn = (cy - np.median(cy)) / np.median(mw)
        sg = cal.get('smile_gain', 1.0)
        smile_bs = (bl[:, bnames.index('mouthSmileLeft')] + bl[:, bnames.index('mouthSmileRight')]) / 2
        smile = np.clip(np.maximum(smile_bs, cyn * 6) * sg, 0, 1)
        frown = np.clip((bl[:, bnames.index('mouthFrownLeft')] + bl[:, bnames.index('mouthFrownRight')]) / 2 * 2, 0, 1)
        puck = bl[:, bnames.index('mouthPucker')]; jaw = bl[:, bnames.index('jawOpen')]
        csq = (bl[:, bnames.index('cheekSquintLeft')] + bl[:, bnames.index('cheekSquintRight')]) / 2
        C = np.stack([eoL, eoR, gyaw, gpitch, bL, bR, bi, mo, mw, smile, frown, puck, jaw, csq], 1)
        sig = np.array([0.6, 0.6, 1.2, 1.2, 1.5, 1.5, 1.5, 0.7, 0.9, 1.2, 1.2, 1.2, 0.7, 1.2])
        for k in range(len(CH)): C[:, k] = smooth(C[:, k], sig[k])
        chans[p] = C; blend[p] = smooth(bl, 0.8)
        mc = (loc[:, 13] + loc[:, 14]) / 2; mouth_c[p] = mc
        lips_o[p] = (loc[:, LIPS_OUT] - mc[:, None])[:, :, :2]; lips_i[p] = (loc[:, LIPS_IN] - mc[:, None])[:, :, :2]
        # --- body: pose landmarks -> camera space on rays at head-ish depth
        pl = d['pose_lm'][p].astype(np.float64)
        pv = ~np.isnan(pl[:, 0, 0])
        dh = -M[:, 2, 3] / 100.0   # head depth (m)
        if pv.sum() > 5:
            pl = fill(pl, pv)
            pw = d['pose_w'][p].astype(np.float64); pw = fill(pw, pv)
            zs = (pw[:, 11, 2] + pw[:, 12, 2]) / 2
            for j, k in enumerate(POSE_J):
                dz = np.clip((pw[:, k, 2] - zs) * 0.7, -0.25, 0.12)
                dj = dh + 0.04 + dz
                x = (pl[:, k, 0] * W - W / 2) * dj / fpx; y = -(pl[:, k, 1] * H - H / 2) * dj / fpx
                joints[p, :, j] = smooth(np.stack([x, y, -dj], 1), 1.2)
                jvis[p, :, j] = smooth(np.where(pv, pl[:, k, 3], 0), 2)
        # --- hands
        for h in range(2):
            hl = d['hand_lm'][p, :, h].astype(np.float64); hv = ~np.isnan(hl[:, 0, 0]); hvalid[p, :, h] = hv
            if hv.sum() == 0: continue
            hl = fill(hl, hv)
            wj = joints[p, :, 4 + h]  # wrist joint depth if pose present
            dw = np.where(np.abs(wj[:, 2]) > 0.05, -wj[:, 2], dh + 0.1)
            dzz = hl[:, :, 2] * W * dw[:, None] / fpx
            dd = dw[:, None] + np.clip(dzz, -0.15, 0.15)
            x = (hl[:, :, 0] * W - W / 2) * dd / fpx; y = -(hl[:, :, 1] * H - H / 2) * dd / fpx
            hands[p, :, h] = smooth(np.stack([x, y, -dd], -1), 0.8)
        stats.append(dict(eye_ratio_p5=float(np.percentile(gl, 5)), eye_ratio_p50=float(np.median(gl)), eye_ratio_p85=float(np.percentile(gl, 85)),
                          mouth_open_m_p50=float(np.median(mo)), mouth_open_m_p95=float(np.percentile(mo, 95)),
                          smile_p50=float(np.median(smile)), smile_p95=float(np.percentile(smile, 95)),
                          head_depth_m=float(np.median(dh)), face_cov=float(v.mean()),
                          hand_cov=[float(hvalid[p, :, h].mean()) for h in range(2)],
                          shoulder_vis=float(np.nanmean(jvis[p, :, :2]))))
    out.update(head_M=head_M, head_valid=head_valid, chans=chans, blend=blend, lips_outer=lips_o, lips_inner=lips_i,
               mouth_center=mouth_c, face_local=face_local, joints=joints, joint_vis=jvis, hands=hands, hand_valid=hvalid)
    return out, stats

if __name__ == '__main__':
    out, stats = build(sys.argv[1], None)   # default gains (no Sonnet calibration)
    np.savez_compressed(sys.argv[2], **out)
    json.dump(stats, open(sys.argv[2].replace('.npz', '.stats.json'), 'w'), indent=1)
    print(json.dumps(stats))
