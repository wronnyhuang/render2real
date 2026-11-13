# De-render pipeline — interfaces (earlier stage)

Task folder: $WORK (all paths relative to it). Python venv with numpy/scipy/cv2/mediapipe: env/mp/bin/python.
Blender 4.0.2 at /usr/bin/blender; there is no display: run as `xvfb-run -a -s "-screen 0 1920x1080x24" blender -b --python <script> -- <args>`.
Engine: BLENDER_WORKBENCH (Eevee is ~1.3 s/frame on llvmpipe; Workbench ~0.3 s). Machine has 70 cores shared with other jobs.

## Clips
clips.json: list of {id, src, fps (30 or 25), out_size [W,H], people (1 or 2), ...}. Source video: clips/<id>.mp4 (20 s, audio AAC).
S01..S12 = portrait 540x960 30 fps, one person, grey studio, waist-up, hands often visible.
R01..R08 = landscape 1280x720 25 fps, SPLIT SCREEN: two people, person 0 in the left half, person 1 in the right half,
           each a head-and-shoulders close-up with its own background. (Each half is a different real camera.)

## rig/<id>.npz (rig/<id>_cal.npz = Sonnet-calibrated variant, rig/<id>_sonnet.npz = Sonnet-only variant; same format)
Camera: at origin, OpenGL convention (x right, y up, looking down -z), vertical FOV = vfov_deg (63), image W x H
(sensor fit VERTICAL). All positions in METERS in that camera space. Arrays (P people, T frames):
- W, H, fps, vfov_deg, P, T
- head_M [P,T,4,4]: head-local -> camera transform. Head-local frame: origin roughly at the centre of the face
  ~mid-nose depth; +x = the person's LEFT (image right when facing camera), +y up, +z out of the face toward the camera.
  Face scale in head-local metres: eye outer corners at x≈±0.05, y≈+0.03; nose tip ≈(0,-0.018,+0.044); mouth centre ≈(0,-0.045,+0.02);
  chin ≈(0,-0.09,+0.007); forehead ≈(0,+0.083,0). Skull centre is roughly (0, 0.01, -0.06).
- head_valid [P,T] bool (gaps already filled; still render).
- channels: names; chans [P,T,14] in this order:
  eye_open_L, eye_open_R  (0 closed .. 1 normal open .. 1.4 wide; L = person's left eye, at +x)
  gaze_yaw, gaze_pitch    (radians relative to head; +yaw = toward +x (person's left), +pitch = up)
  brow_raise_L, brow_raise_R (-1 lowered/frown .. +1 raised), brow_inner (0..1 inner brows up, worried)
  mouth_open (inner-lip gap in metres, 0 .. ~0.03), mouth_width (corner-to-corner metres, ~0.045-0.06)
  smile (0..1 corners up), frown (0..1 corners down), pucker (0..1), jaw_open (0..1), cheek_squint (0..1)
- blend [P,T,52] MediaPipe ARKit blendshapes (names in blend_names) — optional extra detail.
- lips_outer [P,T,20,2], lips_inner [P,T,20,2]: head-local x,y (m) of the lip contours relative to mouth_center [P,T,3]
  (head-local). Contour order goes round the mouth starting at the right corner (index 0). When these are all-zero
  (Sonnet-only variant) build the lips parametrically from mouth_open / mouth_width / smile / frown / pucker.
- face_local [P,T,478,3]: all MediaPipe face-mesh landmarks in head-local metres (all-zero in the Sonnet-only variant).
- joints [P,T,8,3] camera space: L_shoulder, R_shoulder, L_elbow, R_elbow, L_wrist, R_wrist, L_hip, R_hip (person's sides),
  joint_vis [P,T,8] 0..1 (visibility; <0.5 = probably off-screen/occluded; positions still extrapolated).
  For R clips shoulders are usually at/below the bottom edge; hips never visible. All zero => not tracked: derive a torso
  from the head (neck straight down from the head, shoulders ~0.19 m either side, ~0.22 m below the head origin).
- hands [P,T,2,21,3] camera space, MediaPipe hand topology (0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky);
  index 0 = person's left hand, 1 = right; hand_valid [P,T,2] (False = hand not detected that frame: hide the hand
  (or keep it at the wrist joint if the wrist is visible), don't freeze it mid-air).

## scene/<id>.json — written by Sonnet (the look)
{
 "panels": [ {"x0":0.0,"x1":1.0, "background": BG } ],      // one panel for S clips, two (x 0-0.5, 0.5-1) for split-screen R clips
 "lighting": {"key_side":"left|right|front|top","key_strength":0..1,"warmth":-1..1,"ambient":0..1,"contrast":0..1},
 "people": [ PERSON, ... ]   // index = person slot (0 = left in R clips)
}
BG = {"base_color":"#rrggbb", "gradient_bottom":"#rrggbb"|null,
      "shapes":[{"kind":"rect|ellipse|vertical_band|floor", "x0":..,"y0":..,"x1":..,"y1":.. (normalised IMAGE coords, y down, 0..1 of the full frame),
                 "color":"#rrggbb","label":"short text e.g. door, lamp, fireplace"}]}   // drawn back to front, behind the person
PERSON = {
 "skin":"#rrggbb", "lip_color":"#rrggbb", "eye_color":"#rrggbb", "brow_color":"#rrggbb", "brow_thickness":0..1,
 "hair":{"style":"bald|buzz|short|side_part|medium|long_straight|long_wavy|curly_short|curly_big|afro|bun|ponytail|bob|locs|beanie|cap",
         "color":"#rrggbb","volume":0..1,"length":0..1,"part":"none|left|right|center","bangs":true|false},
 "facial_hair":{"style":"none|stubble|mustache|goatee|short_beard|full_beard","color":"#rrggbb"},
 "glasses":{"present":true|false,"frame_color":"#rrggbb","shape":"round|rect|cat_eye","thickness":0..1},
 "earrings":true|false, "necklace":true|false,
 "top":{"type":"tshirt|vneck|crew_sweater|hoodie|button_shirt|blazer|jacket_over_top|turtleneck|tank|cardigan",
        "color":"#rrggbb","secondary_color":"#rrggbb","pattern":"solid|stripes|plaid|tie_dye|graphic|floral","pattern_color":"#rrggbb"},
 "bottom":{"type":"jeans|trousers|skirt|shorts|leggings|dress","color":"#rrggbb"},   // NEW: lower garment (visible in S clips, waist down)
 "build":"slim|average|broad|heavy", "head_scale":0.85..1.15, "neck_length":0..1,
 "headwear":"none|beanie|cap|headband", "headwear_color":"#rrggbb"
}

## Renderer CLI (to build): src/render_blender.py
xvfb-run -a -s "-screen 0 1920x1080x24" blender -b --python src/render_blender.py -- --rig rig/<id>[_cal|_sonnet].npz --scene scene/<id>.json --out renders/<name>.mp4 [--frames a:b] [--still t --png out.png]
Writes frames then encodes MP4 (libx264 crf 20, yuv420p) at the rig's fps and exactly T frames, size W x H, NO audio.
