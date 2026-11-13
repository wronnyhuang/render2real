"""MediaPipe tracking of one clip -> tracks/<id>.npz (raw, per person sorted left->right)."""
import sys, json, numpy as np, cv2, mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision as V
cid = sys.argv[1]; VID = sys.argv[2]; OUT = sys.argv[3]
P = 1
MOD = __import__('os').path.expandvars('$WORK/models/')
n2 = 0   # frames where a 2nd face is detected (intrusion check)
cap = cv2.VideoCapture(VID); fps = cap.get(cv2.CAP_PROP_FPS)
W, H = int(cap.get(3)), int(cap.get(4))
RM = V.RunningMode.VIDEO
face = V.FaceLandmarker.create_from_options(V.FaceLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=MOD + 'face_landmarker.task'),
        running_mode=RM, num_faces=2, output_face_blendshapes=True, output_facial_transformation_matrixes=True,
        min_face_detection_confidence=0.4, min_tracking_confidence=0.4))
pose = V.PoseLandmarker.create_from_options(V.PoseLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=MOD + 'pose_landmarker_full.task'),
        running_mode=RM, num_poses=P, min_pose_detection_confidence=0.4))
hand = V.HandLandmarker.create_from_options(V.HandLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=MOD + 'hand_landmarker.task'),
        running_mode=RM, num_hands=2*P, min_hand_detection_confidence=0.4))
frames = []
while True:
    ok, f = cap.read()
    if not ok: break
    frames.append(f)
T = len(frames)
face_lm = np.full((P, T, 478, 3), np.nan, np.float32); blend = np.full((P, T, 52), np.nan, np.float32)
face_mat = np.full((P, T, 4, 4), np.nan, np.float32)
pose_lm = np.full((P, T, 33, 4), np.nan, np.float32); pose_w = np.full((P, T, 33, 3), np.nan, np.float32)
hand_lm = np.full((P, T, 2, 21, 3), np.nan, np.float32)   # [.., 0=person's left hand, 1=right hand]
bnames = None
def slot(xc, centers):
    # assign to person index by x position (people sorted left->right)
    if P == 1: return 0
    return int(np.argmin([abs(xc - c) for c in centers]))
centers = [ (i + 0.5) / P for i in range(P)]
for t, f in enumerate(frames):
    ts = int(round(t * 1000 / fps))
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    r = face.detect_for_video(img, ts)
    if len(r.face_landmarks) > 1:
        n2 += 1
        sz = [np.ptp([p.x for p in lm]) for lm in r.face_landmarks]; keep = int(np.argmax(sz))
    else: keep = 0
    for i, lm in enumerate(r.face_landmarks):
        if i != keep: continue
        a = np.array([[p.x, p.y, p.z] for p in lm], np.float32)
        s = slot(a[:, 0].mean(), centers)
        face_lm[s, t] = a[:478]
        if r.face_blendshapes:
            blend[s, t] = [b.score for b in r.face_blendshapes[i]]
            if bnames is None: bnames = [b.category_name for b in r.face_blendshapes[i]]
        if r.facial_transformation_matrixes: face_mat[s, t] = np.array(r.facial_transformation_matrixes[i])
    if P > 1:  # update centers from face positions
        for s in range(P):
            if not np.isnan(face_lm[s, t, 0, 0]): centers[s] = 0.8 * centers[s] + 0.2 * float(np.nanmean(face_lm[s, t, :, 0]))
    r = pose.detect_for_video(img, ts)
    for i, lm in enumerate(r.pose_landmarks):
        a = np.array([[p.x, p.y, p.z, p.visibility] for p in lm], np.float32)
        s = slot(a[0, 0], centers); pose_lm[s, t] = a
        if r.pose_world_landmarks: pose_w[s, t] = np.array([[p.x, p.y, p.z] for p in r.pose_world_landmarks[i]], np.float32)
    r = hand.detect_for_video(img, ts)
    for i, lm in enumerate(r.hand_landmarks):
        a = np.array([[p.x, p.y, p.z] for p in lm], np.float32)
        s = slot(a[0, 0], centers)
        # MediaPipe handedness assumes a mirrored (selfie) image: label "Left" == person's RIGHT hand in a non-mirrored video.
        lab = r.handedness[i][0].category_name
        # disambiguate with pose wrists when available
        k = 1 if lab == 'Left' else 0
        pw = pose_lm[s, t]
        if not np.isnan(pw[15, 0]):
            dl = np.hypot(*(a[0, :2] - pw[15, :2])); dr = np.hypot(*(a[0, :2] - pw[16, :2]))
            k = 0 if dl < dr else 1   # pose 15 = person's left wrist
        hand_lm[s, t, k] = a
np.savez_compressed(OUT, face_lm=face_lm, blend=blend, face_mat=face_mat, pose_lm=pose_lm, pose_w=pose_w,
                    hand_lm=hand_lm, fps=fps, W=W, H=H, n_second_face=n2, bnames=np.array(bnames or []))
cov = [float(np.mean(~np.isnan(face_lm[s, :, 0, 0]))) for s in range(P)]
hc = [float(np.mean(~np.isnan(hand_lm[s, :, :, 0, 0]))) for s in range(P)]
print(cid, T, 'face coverage', cov, 'hand frac', hc, 'second_face_frames', n2)
