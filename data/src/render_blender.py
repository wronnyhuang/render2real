"""Stylised clay/toy 3D re-creation of a talking-head clip from a tracking rig (earlier stage).

Run (see spec/INTERFACE.md):
  xvfb-run -a -s "-screen 0 1920x1080x24" blender -b --python src/render_blender.py -- \
      --rig rig/<id>.npz --scene scene/<id>.json --out renders/<name>.mp4 [--frames a:b] [--still t --png out.png]
Extra options: --workers N (parallel Blender children, default auto), --aa {FXAA,5,8,11,16}, --keep-frames DIR.

Geometry (all in the rig camera space, metres; camera at origin looking -Z, vertical FOV = vfov_deg):
  head empty  (matrix_world = head_M[p,t])
   +- face frame F (matrix_local = A_t: similarity fit canonical face template -> face_local[t])
       +- head mesh: the face_local landmark mask (clay mask, lips/eyelids/jaw move with the landmarks)
       |             + a back-of-head shell grown from the face oval; eye and mouth holes
       +- eyeballs (white + iris + pupil), brows, mouth interior + teeth, ears, hair, glasses, earrings
  body: neck capsule, torso (from shoulder joints or derived from the head), arms, hands (21 landmarks), mitten fallback
  backdrop: one textured plane per panel at 3.6 m covering exactly the panel's frustum (exact hex colours)
Per panel (split-screen R clips) the frame is rendered with a border crop holding only that panel's people,
then the halves are stacked by ffmpeg with a dark divider at x = 0.5.
"""
import sys, os, json, math, time, glob, base64, shutil, subprocess, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = __import__('os').path.expandvars('$WORK')
for sp in sorted(glob.glob(os.path.join(ROOT, 'env', 'mp', 'lib', 'python3*', 'site-packages'))):
    if sp not in sys.path:
        sys.path.append(sp)
import numpy as np
import bpy
from mathutils import Matrix

# ----------------------------------------------------------------------------------------------------------------------
# embedded data: symmetric canonical face (478 MediaPipe landmarks, head-local metres, 0.1 mm int16) + face-mesh triangles
# ----------------------------------------------------------------------------------------------------------------------
TEMPLATE_B64 = (
    "AABU/kMBAAA3/+gBAAAA/ysByv8YAF4BAAB2//QBAADS/8UBAAC4AOAALP78AJ3/AABYAZsAAACrAaQAAAAQAxwAAABC/jkBAAA0"
    "/iIBAAAm/vsAAAD3/QMBAADY/QwBAACy/R8BAACK/RcBAABh/c0AAAAZ/8gBvv8f/10B+Pw4AhP++f6/APT/sv64APL/av68AN//"
    "D/7pAJL/Mv/LAOz/eP5yAeX/zf5lAe7/Mf5qAc//Bv5VAbf/zf3DAHb/DP8V/XYAEv79AIn/1PwLARv+cP37ACT/uv6Z/2cAtP9l"
    "/jUBuf85/hQBc/9b/gcBP/9P/s0Aff9C/voATP8//sQA1P74/VkAzf85/+EBxP93/+sBnv2MAav/RP9AAFwARP9j/xoBQv+K//kA"
    "6f2u/wcAx//P/68BRf6tATAA5P2pAfb/Tf2WAp7+cf9jAYQAFf9GAeD/o/4//kAAGP1s/q39cf86/wcBnf8j/xEB8v42/k0A/v42"
    "/lwAvf3ZAcr/Qv9L//IAxv6XAWEAwP6zAWUAdv7/Arj/if02Akb/oP5rAhcAcv20AWT/N/3uAcj+tv9N/i0Bd/9S/gQBQ/9K/ssA"
    "ff8u//wA+v42/lgAF/8Z/oQACP80/mMAiv9I/3EBV/8x/q4Ahv8x/tIAvf8s/vAApf9o/ccAqv+V/Q0Brv+8/RIBtP/g/f4Avf/8"
    "/fYAWP8W/q0ASf8M/rIAOv/6/bsALv/l/bEA9/6j/qcAs/zj/wv9AAAJ/2UBNf8q/oQAI/8t/ocAr/8I/xEBVf8l/8AApv8U/xYB"
    "Af8fAEQAjf71/zgANv9w/9IAyf3ZAjP//f1iArD/LP7mARwAC//C/YIAW/+xAZgARf9nAlMALP8OAwYAMP7MALr/fP2aAFP/Uv/Z"
    "AOH/wf1AAYX/cP9kAHMAZf9y/2QBKf1yAAL/sf1iAJ//Dv5CAOr/lv5HAAsA+v5gABUAPv95ACkAxP+rAMUAPf3c/y//g/1OAWP/"
    "3/8c/8IBTf/5/4YAofwrARv9cP+UADQAL/9x/4QA7v0TAXH/Zf+j/z0B2Pwu/0b9V//sANT/kv+//4IB4v26/VX/2v1s/dv+3fzb"
    "/yD+gv0a/uv+Af2KAVv+/v7S/FIA5f8M/18BEv/N/24ALf37ANj+cP7OANj/s/7KAOv/Cv8P/nkAT/1P/yf/dv9y/EsArP7R/LT/"
    "T/4Q/Vj/AABkAmcAAABo/GEA8/7QAO3/Kv/bAOL/Sf/iANj/Sf1oARH/Ev8bAdv/1P4sAeL/lv4wAd3/X/4pAcv/O/4hAbX/wvzF"
    "AYL9Sv7tALX/AADI/iEBOv/C/s8Acf9D/ygBq//L/hYBAAAPAZ0ARf5d/ab/n/4X/fX/dP+l/J4Ac/3U/UX+Q//3ANL/m/9BAOgA"
    "AACZ/LIACv+a/A8A/vw6/0H+hf8I/tYAeP/1/dwAa//Y/ekAYv+2/eQATv+N/a4AH/88/owAGf9A/osAFf9D/owAw/52/ncAtv0h"
    "/8f/mv+MAJYAd/8TAQEAUf8TAeP/KP82/ocAq/17/mz/sf8HAWoAK/9V/YsAAAAfAHUBxv9eAB0BAABoACwBdP/g//IAAADY/NMA"
    "AAAh/coAkv8q/bkAlP7e/TAA8f4+/3kA2f6R/VoATf5M/1IArf79/nIAFP7m/iIAfP/i/L4AU//C/7wAW/6q/fH/s/5Z/ScAXv41"
    "/iAAZP3X/hz/B/4u/tn/LP2i/nT+ef6r/lgAdf8eALIAef9U/34BV/9L/ygBkf94/6wBPv9LAREAy/53ARQAZP6GAQgAEv6AAej/"
    "3P1qAb3/uv0HAVj/zvx2ABn++P2hAKL/Qv6IANb/o/6FAPP//P6TAPr/Ov+mAP//ZP+0AAIAovyNAPn8Wf89/wUBnP8BACsBof9J"
    "/7oBu/8r/58BoP9B/5gBaP8p/+4Ayf8i/7UBz/8T/18BY//pANr/g//UAAwAlf+/AEcAJv4LAZ7/6/05AZP/NgAYAF4B2gELAZ7/"
    "QgAf/10BCAM4AhP+DQHQAO3/TQHKAOv/kAHOANj/7gH9AIn/1gDbAOL/iAFyAeX/MwFlAe7/zwFqAc//+gFVAbf/MwLDAHb/9AAV"
    "/XYA7gERAYD/LAMLARv+kAL7ACT/RgGZ/2cATABl/jUBRwA5/hQBiQBS/gQBvQBK/ssAgAA6/vUAsAA6/r0ALAH4/VkAMwA5/+EB"
    "PAB3/+sBYgKMAav/vABAAFwAvABj/xoBvgCK//kAFwKu/wcAOQDP/68BuwGtATAAHAKpAfb/swKWAp7+jwBjAYQA6wBGAeD/XQE/"
    "/kAA6AJs/q39jwA6/wcBYwAj/xEBDgE2/k0A+AA0/mMAQwLZAcr/vgBL//IAQAGzAWUARgHOAWkAigH/Arj/dwI2Akb/YAFrAhcA"
    "jgK0AWT/yQLuAcj+SgBN/i0BgwBC/voAtAA//sQAgwAu//wAAgE2/lwA6QAZ/oQA9AAz/mQAdgBI/3EBqQAx/q4AegAx/tIAQwAs"
    "/vAAWwBo/ccAVgCV/Q0BUgC8/RIBTADg/f4AQwD8/fYAqAAW/q0AtwAM/rIAxgD6/bsA0gDl/bEACQGj/qcATQPj/wv9zAAj/oMA"
    "2QAe/oYAUQAI/xEBqwAl/8AAWgAU/xYB/wAfAEQAcwH1/zgAygBw/9IANwLZAjP/AwJiArD/1AHmARwA9QDC/YIApQCxAZgAuwBn"
    "AlMA1AAOAwYA0AHMALr/hAKaAFP/twDiANj/PwJAAYX/kABkAHMAmwBy/2QB1wJyAAL/TwJiAJ//8gFCAOr/agFHAAsABgFgABUA"
    "wgB5ACkAPACrAMUAwwLc/y//fQJOAWP/IQAc/8IBswD5/4YAXwMrARv9kACUADQA0QBx/4QAEgITAXH/mwCj/z0BKAMu/0b9vQD3"
    "ANL/bgC//4IBHgK6/VX/JgJs/dv+IwPb/yD+fgIa/uv+/wKKAVv+AgHS/FIAGwAM/18B7gDN/24A0wL7ANj+jAHhAND/TwHdAOP/"
    "9gAP/nkAsQJP/yf/igBy/EsAVAHR/LT/sQEQ/Vj/EgHjAOX/3gDrANn/wwD3ANL/twJoARH/7gAbAdv/LAEsAeL/agEwAd3/oQEp"
    "Acv/xQEhAbX/PgPFAYL9tgHtALX/xgDC/s8AjwBD/ygBVQDL/hYBuwFd/ab/YQEX/fX/jACl/J4AjQLU/UX+xgABAdX/ZQBBAOgA"
    "9gCa/A8AAgM6/0H+ewAI/tYAiAD1/dwAlQDY/ekAngC2/eQAsgCN/a4A0QAs/okA3QA6/owA4QA8/owAPQF2/ncASgIh/8f/ZgCM"
    "AJYAiQATAQEArwATAeP/ywAq/oQAVQJ7/mz/TwAHAWoA1QBV/YsAOgBeAB0BjADg//IAbgAq/bkAbAHe/TAADwE+/3kAJwGR/VoA"
    "swFM/1IAUwH9/nIA7AHm/iIAhADi/L4ArQDC/7wApQGq/fH/TQFZ/ScAogE1/iAAnALX/hz/+QEu/tn/1AKi/nT+hwGr/lgAiwAe"
    "ALIAhwBU/34BqQBL/ygBbwB4/6wBwgBLAREANQF3ARQAnAGGAQgA7gGAAej/JAJqAb3/RgIHAVj/MgN2ABn+CAKhAKL/vgGIANb/"
    "XQGFAPP/BAGTAPr/xgCmAP//nAC0AAIAXgONAPn8pwA9/wUBZAABACsBXwBJ/7oBRQAr/58BYABB/5gBmAAp/+4ANwAi/7UBMQAT"
    "/18BqQDsANT/fQDUAAwAawC/AEcA3QEaAZ//FQI5AZP/q/4MAcH/7v4LAcD/q/5EAaX/c/75AMb/s/7GAOf/VQEMAcH/ogEdAcb/"
    "VQFEAaX/EgELAcD/VQHWAOL/")
TRIS_B64 = (
    "fwAiAIsACwAAACUA6ADnAHgASAAlACcAgAB5AC8A6AB5AIAAaABFAEMArwCrAJQAdgAyAGUASQAnACgACQCXAGwAMABzAIMAwgDM"
    "ANMASgAoALkAUAAqALcAKABcALoA5gDlAHYAygDUANYAUwASABEATAA9AJIAoAAdAB4AOACdAK0AagDMAMIAhwDWAMAAywClAGIA"
    "FQBHAEQAMwAtAAQAkAAYABcATQCSAFsAzQAyALsAyQDIABIAWwBqALYAWgBbALUAVQBUABEAzgDLACQAlACrAIwAXAAoACcAwQC9"
    "APQAnwCeABwA9wD2AKEA7AADAMQANgBEAGgAwQCoAAgAdQDkAB8AvQDBADcAYgBhAGMAfgAvAGQApgBPANoAmwCaABoA0QAxAIMA"
    "hwCIAJYALwB+ANkA3wA0ADUALQAzAIYA0wCqAIwAQwBFAGwAKwBqAFsA5gB3AHgA4gCCAPcAPwA1ADQA7gAUAPIALgBGAJwATgA+"
    "AGAALgA1AD8AjwAiAOMAewB1AG8ALAB9ABMA7ACGADMA2ADOAM0AmgCZABYAJwAlAKcAyADJANAAJACOAGQAOQDUAMoAFAA8AGMA"
    "HACeAJ0AIwDiAHEAoACfABsAzADKANIAcQDhAC4AKwDKAMwAPgBMAE0AiQB7AHQAKQAmAEgAywCBAI4AQABiAPAAMQBmAEAAKQBJ"
    "AEoA1ADYAM8AKgBKALgAqQCqANMAqgCVALAAaQBCAEUAegAGAKgAewCTALsAYABNAFoAQQA3AGsAWQBaALQAZQBkAHgAPwBpAGgA"
    "XQCJAOMADwBWAFUAgQBmADEADgBXAFYANwAIAAkAZAAvAHkAkQAXABYAWABZALMABgB6AMQAWABfAGAAigCsAIgA1wA6AKwAcwAw"
    "ANsAKgBQAFEAwwADADMAKwCSAD0AqwCvAMcAUQBSACYANQAuAOEAkACjAG4ANABBAEIA5QDkAHUAIgB/AOoAawBsAEUAbQBsAJcA"
    "MABAAOsAPgBOAL8AgQDRAH4AbwAjAI8AdQB7ADIA3gBBADQAEwB9AI0A3QA3AEEAAwDDAMUAGQAHACEA3ADtACwARgBHAIsAegDB"
    "APUA9wCCACEARwAVAKIAqgCpAJYAvACuAMQA2AC6AFwAAgBhAKcAjQB9APEApACnACUASAAmAAwAJgBSAA0APwBEAEcA4gAjAG8A"
    "ZQAyAM0AzgBcAKUA0QDGANkApQCnAGEA3ABzANoAhQBwAPMA7wDuAPEA1gCHAKkAvgCtAIUAqwDQACAAfQAsAO0AVgBXALIAVQBW"
    "ALMAVABVALQAUwBUALUAyQBTALYAiQBdAIQATAA+ALcAPQBMALgAOQA9ALkA1AA5ALoA1gDPALsAIgCPAJwATwDvAO0AewCJALEA"
    "LAABAAQAyQDCACAAQABmAIEA1QDXAIoAOwCmANsA8gBjAGEAAgBeAI0ASwA7AOsAGABuAOQAGQCCAOIAFwAYAOUAFgAXAOYAGgAW"
    "AOcAcAAaAOgAvQC+APMA3QA4AL4AHAA4AN0AGwAcAN4AHQAbAN8AHgAdAOAA9wAeAOEA7gBPABQApgA7AEsAPABLAPAAkwCxANcA"
    "FABPAKYAuwCTANUAcADpAPQA6QCAAPUAgAByALwAcgDZAK4AgwBzANwA2QDGAOwAxgCDAIYAsQCEADoAjwAjAHwAbgCjAAcA5ABu"
    "ABkAZAGFAXABCwAuAQsBxAFeAV0BLgEvAQ0BZQFXARUBxAHFAWUBTQFMASkBrwCYAHkBWwFcAUoBLwEwAQ4BCQBQAVEBFgEXAWgB"
    "ogEGAa8BMAGYAZkBNgGfAZcBDgGZAZoBwgFcAVsBpgGuAbIBOQE6AREAMgEzAXcBgwGEAQQBHgGeAY4BTwGWAaIBbAFvAaABpwFm"
    "AUcB+wAcASoBGQEFAAQAdQF2Af0AMwFAAUEBqQGrAZsBpQE5ARIAQQGVAZYBQAGUAZUBOwEQABEAqgGpAQoBeQGQAXEBQgGHAQ0B"
    "oQHRAdABggEBAQIB0gEEAYQByAGPAaMBHAFMAU0BoQEdAQgAWgFUAQUBnQG5AR0BRwHMAUgBYwFzAUkBiAG3AbYBfgFVAQABrQGk"
    "AWgBbAGKAXsBFQFXAbUBuwG8ARsBEwG4AWsBrwEGAXEBKQFSAVEBEQF3AUEBwgHDAV0BvgFWAdMBJQFOARoBygHNAc4BFAFhAX8B"
    "NAFEAUUBFAEsASUBdAFZAb8BYAFZAVQBEgEBABMAyAH4ABkBtAGrAakBfQEAAfwADQGHAYkByADHAKwBCgFKAUkBHwERAaYB+gDO"
    "AUgBAgEeAYABCQFhAVYBgwEDAQEBqAGvAa4BVgFhARQBEQFPAagBJAFFATMBbgG/AVkBDwEvAS4BpwEKAXMBJgHHAcwBFwEWASYB"
    "DwEQATABsAGyAasBEAGXAZgBigGuAa8BiwFxAZABTgFNASsBXwGhAagAYAEYAZsBRQE/AUABJwEoAVABPwGTAZQBSgFcAV0BJQEq"
    "AU0BQwHGAb8BDwAQADsBZgGtARcBDgAPADwBHQFQAQkASQFdAV4BdgF8AfwAPgGSAZMBBgDFAKMBPgE/AUUBbwFsAW0BswFvAY0B"
    "WAG2AbcBEAEPATcBwwAFABkBEQEfASMBjAGsAccANwEPAQwBGwG8Ab0BdQH+AFMBGgFOASgBwQFbAVoBCAG/AcYBUAEoASsBUgEK"
    "AJcAFgG3AccBJAGXAZ8BZgFzAWMBVAFZAXQBWgFbARgBugG7ARoBEwBeAHIBuQG6AScB+ACjAcUABwH/AGcBuAETARIBLAF/AXAB"
    "XwGcAdEBBwHTAdIBLQFwAYUBiwF6AXsBnAFfAaMBtAGqAUIBAgCkAIkBcgHOAc0BpAAAAAsBLgELAAwADAEMAA0AJQEsAS0BvgEF"
    "AVQBSgEKAakBqgGnAYcBrQFjAbUBhwFHAUYBuAHJAbYBVQF+AWoBywHJAc0BsgGuAYoBngHPAWoBjAFxAQYBYgHNAckBPAGTAZIB"
    "OwGUAZMBOgGVAZQBOQGWAZUBpQGiAZYBbgGRAWkBMgGYAZcBIwGZAZgBHwGaAZkBsAG0AZoBsgGgAZsBCAFwAX8BNQG2AckBYAF4"
    "AZEBEgETAQQApQGsAQYBJgFHAWYBsQGgAW8BIQHHAbcBzgFyAUYBAgBGAXIBMQHMAccB/gDBAcAB/wAFAb4B/QDCAcEB/ADDAcIB"
    "AAHEAcMBVQHFAcQBnQHQAc8BuQGdAZ4BAgG6AbkBAQG7AboBAwG8AbsBBAG9AbwB0wFWAb0BywHKAfoAIQGIASIBIgFIAcwBeAGx"
    "AbMB+gAiAYgBmwGgAbEBVQHPAdABxQHQAdEBZQHRAZwBVwGcAY8BaAFrAbgBtQGPAcgBpAHIAWsBkQGzASABdAF/AWEBUwH/APkA"
    "wAEFAf8AhQDzAL4AhQCbAHAAIQD2APcAIQCCABkAjgGAAR4BagGOAZ4BagHPAVUBBwFnAdMBBwH5AP8A0gHTAQQBSwA8AKYA7gDv"
    "AE8AogB/AIsASAALACUAeQDoAHgASQBIACcAcgCAAC8A6QDoAIAAZwBoAEMAmACvAJQAdwB2AGUASgBJACgAawAJAGwAMQAwAIMA"
    "IADCANMAuABKALkAvwBQALcAuQAoALoAdwDmAHYA0gDKANYAVABTABEATQBMAJIAoQCgAB4AvgA4AK0AtgBqAMIAigCHAMAAgQDL"
    "AGIANgAVAEQABQAzAAQAkQCQABcAWgBNAFsAzwDNALsAUwDJABIAtQBbALYAtABaALUAEABVABEAzQDOACQAsACUAIwApQBcACcA"
    "9QDBAPQAGwCfABwAHgD3AKEArgDsAMQAZwA2AGgANwDBAAgAbwB1AB8A3QC9ADcA8ABiAGMAjgB+AGQA2wCmANoAcACbABoAxgDR"
    "AIMAqQCHAJYAcgAvANkA4ADfADUA3AAtAIYAIADTAIwAbQBDAGwAkgArAFsA5wDmAHgAcQDiAPcAaQA/ADQA8QDuAPIAfAAuAJwA"
    "XwBOAGAARgAuAD8AdACPAOMAdAB7AG8AAQAsABMAAwDsADMAzwDYAM0AGgCaABYApQAnAKcAxwDIANAAZQAkAGQAKwA5AMoA8gAU"
    "AGMAOAAcAJ0AfAAjAHEAHQCgABsA0wDMANIAfABxAC4AagArAMwAYAA+AE0A4wCJAHQASQApAEgAJADLAI4A6wBAAPAAMAAxAEAA"
    "KgApAEoA1gDUAM8AtwAqALgA0gCpANMAjACqALAAaABpAEUAwQB6AKgAMgB7ALsAWQBgAFoAQgBBAGsAswBZALQAdwBlAHgARAA/"
    "AGgA6gBdAOMAEAAPAFUA0QCBADEADwAOAFYAawA3AAkAeABkAHkAmQCRABYAsgBYALMAxQAGAMQAWQBYAGAAhwCKAIgAigDXAKwA"
    "2gBzANsAKQAqAFEABQDDADMAOQArAD0A0ACrAMcAKQBRACYA4AA1AOEAGACQAG4AaQA0AEIAdgDlAHUA4wAiAOoAQgBrAEUACgBt"
    "AJcA2wAwAOsAtwA+AL8AjgCBAH4AdABvAI8AdgB1ADIA3wDeADQAXgATAI0A3gDdAEEAxAADAMUALQDcACwAnABGAIsAvAB6APUA"
    "iwBHAKIAlQCqAJYAegC8AMQAzgDYAFwApAACAKcA8gCNAPEAAACkACUACwBIAAwADAAmAA0ARgA/AEcAHwDiAG8AJABlAM0AywDO"
    "AKUAfgDRANkAYgClAGEA7QDcANoA7QDvAPEA0gDWAKkAjACrACAA8QB9AO0AswBWALIAtABVALMAtQBUALQAtgBTALUAwgDJALYA"
    "sQCJAIQAuABMALcAuQA9ALgAugA5ALkA2ADUALoAwADWALsAiwAiAJwA2gBPAO0AkwB7ALEALQAsAAQA0ADJACAAYgBAAIEAwADV"
    "AIoA6wA7ANsAjQDyAGEAYQACAI0A8ABLAOsA5QAYAOQAHwAZAOIA5gAXAOUA5wAWAOYA6AAaAOcA6QBwAOgA9AC9APMAvQDdAL4A"
    "3gAcAN0A3wAbAN4A4AAdAN8A4QAeAOAAcQD3AOEAYwA8APAA1QCTANcAPAAUAKYAwAC7ANUA8wBwAPQA9ADpAPUA9QCAALwAvABy"
    "AK4AhgCDANwArgDZAOwA7ADGAIYA1wCxADoAnACPAHwAGQBuAAcAHwDkABkACAFkAXABAAALAAsBwwHEAV0BCwEuAQ0BXgFlARUB"
    "XgHEAWUBKwFNASkBjAGvAHkBGAFbAUoBDQEvAQ4BlwAJAFEBWAEWAWgBqAGiAa8BDgEwAZkBEAE2AZcBQgEOAZoBwQHCAVsBsAGm"
    "AbIBEgA5AREAIwEyAXcBAwGDAQQBqAFPAaIBsgFsAaABhwGnAUcBLQH7ACoBEwEZAQQA/gB1Af0AdwEzAUEBGAGpAZsByAClARIA"
    "TwFBAZYBQQFAAZUBOgE7AREApwGqAQoBjAF5AXEBDgFCAQ0BnQGhAdABgQGCAQIB+ADIAaMBKgEcAU0BqAChAQgAwAFaAQUBoQGd"
    "AR0BRgFHAUgBFQFjAUkBNQGIAbYBfQF+AQABFwGtAWgBbQFsAXsBYwEVAbUBGgG7ARsBGQETAWsBiwGvAXEBKwEpAVEBTwERAUEB"
    "XAHCAV0BZwG+AdMBGwElARoB+gDKAc4BLAEUAX8BJAE0AUUBGwEUASUBCAF0Ab8BWgFgAVQBYgESARMAawHIARkBqgG0AakBfAF9"
    "AfwACwENAYkBpQHIAKwBcwEKAUkBsAEfAaYBIgH6AEgBgQECAYABvgEJAVYBggGDAQEBpgGoAa4BvQFWARQBpgERAagBMgEkATMB"
    "YAFuAVkBDAEPAS4BZgGnAXMBRwEmAcwBSwEXASYBLwEPATABtAGwAasBMAEQAZgBiwGKAa8BegGLAZABKAFOASsBBgBfAagAeAFg"
    "AZsBMwFFAUABHQEnAVABQAE/AZQBSQFKAV0BTgElAU0BbgFDAb8BPAEPADsBSwFmARcBPQEOADwBCAAdAQkAFQFJAV4B/QB2AfwA"
    "PwE+AZMBXwEGAKMBRAE+AUUBjQFvAW0BIAGzAY0BFgFYAbcBNgEQATcB+ADDABkBdwERASMBrwCMAccAOAE3AQwBFAEbAb0BhgF1"
    "AVMBJwEaASgBwAHBAVoBZAEIAcYBUQFQASsBUQFSAZcAJgEWAccBNAEkAZ8BrQFmAWMBCQFUAXQBYAFaARgBJwG6ARoBYgETAHIB"
    "HQG5AScBwwD4AMUAyQG4ARIBLQEsAXABoQFfAdEB+wAtAYUBigGLAXsBjwGcAaMBmgG0AUIBRgECAIkBYgFyAc0BiQGkAAsBDAEu"
    "AQwAOAEMAQ0AKgElAS0BCQG+AVQBGAFKAakBQgGqAYcBpAGtAbUBiQGHAUYBWAG4AbYBygHLAc0BbAGyAYoBrAGMAQYBEgFiAckB"
    "PQE8AZIBPAE7AZMBOwE6AZQBOgE5AZUBOQGlAZYBQwFuAWkBJAEyAZcBMgEjAZgBIwEfAZkBHwGwAZoBqwGyAZsBdAEIAX8BywE1"
    "AckBbgFgAZEBAQASAQQAogGlAQYBSwEmAWYBswGxAW8BiAEhAbcBSAHOAUYBXgACAHIBIQExAccBUwH+AMABZwH/AL4B/gD9AMEB"
    "/QD8AMIB/AAAAcMBAAFVAcQBngGdAc8BHgG5AZ4BHgECAbkBAgEBAboBAQEDAbsBAwEEAbwBBAHTAb0BNQHLAfoAMQEhASIBMQEi"
    "AcwBkQF4AbMBNQH6AIgBeAGbAbEBxQFVAdABZQHFAdEBVwFlAZwBtQFXAY8BWAFoAbgBpAG1AcgBaAGkAWsBaQGRASABCQF0AWEB"
    "hgFTAfkAUwHAAf8A")

TEMPLATE = np.frombuffer(base64.b64decode(''.join(TEMPLATE_B64)), '<i2').reshape(478, 3).astype(np.float64) / 10000.0
TRIS = np.frombuffer(base64.b64decode(''.join(TRIS_B64)), '<u2').reshape(-1, 3).astype(np.int64)

LIPS_OUT = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LIPS_IN = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
        150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
# eyes: (outer corner, inner corner, lower interior outer->inner, upper interior outer->inner, iris centre)
EYE_R = dict(c0=33, c1=133, lo=[7, 163, 144, 145, 153, 154, 155], up=[246, 161, 160, 159, 158, 157, 173], iris=468,
             ring=[33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246])
EYE_L = dict(c0=263, c1=362, lo=[249, 390, 373, 374, 380, 381, 382], up=[466, 388, 387, 386, 385, 384, 398], iris=473,
             ring=[263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466])
BROW_R = ([70, 63, 105, 66, 107], [46, 53, 52, 65, 55])        # (upper, lower) outer -> inner
BROW_L = ([300, 293, 334, 296, 336], [276, 283, 282, 295, 285])
FIT_IDX = [33, 133, 362, 263, 6, 168, 197, 195, 10, 151, 9, 109, 338, 67, 297, 234, 454, 127, 356, 162, 389, 21, 251,
           116, 345, 123, 352, 1, 4, 5, 8, 107, 336, 70, 300]
LOWER_LIP_IDX = set([146, 91, 181, 84, 17, 314, 405, 321, 375, 95, 88, 178, 87, 14, 317, 402, 318, 324])

# canonical skull (template frame): centre + semi-axes (x, y up, y down, z back, z front)
SK_C = np.array([0.0, 0.012, -0.072])
SK_AX, SK_AYU, SK_AYD, SK_AZB, SK_AZF = 0.084, 0.118, 0.086, 0.108, 0.090
NRING = 10   # back-shell rings

# torso rings: (kind, y or u, a_S, a_Hip, b, half-depth, z-centre).  kind 'a': metres above the shoulder line;
# 'u': fraction of the shoulder->hip distance below it.  half-width = a_S*S + a_Hip*Hh + b (S = shoulder half-span).
PROF_ROWS = [
    ('a', 0.052, 0.00, 0.0, 0.050, 0.040, -0.060),
    ('a', 0.046, 0.30, 0.0, 0.020, 0.050, -0.050),
    ('a', 0.036, 0.70, 0.0, 0.012, 0.060, -0.036),
    ('a', 0.022, 0.95, 0.0, 0.018, 0.072, -0.022),
    ('a', 0.000, 1.00, 0.0, 0.032, 0.082, -0.010),
    ('u', 0.08, 1.00, 0.0, 0.030, 0.095, -0.002),
    ('u', 0.22, 1.00, 0.0, 0.012, 0.105, 0.000),
    ('u', 0.40, 0.93, 0.0, 0.000, 0.104, 0.000),
    ('u', 0.58, 0.55, 0.30, 0.000, 0.100, 0.000),
    ('u', 0.75, 0.30, 0.65, 0.000, 0.100, 0.000),
    ('u', 0.90, 0.10, 0.90, 0.000, 0.102, 0.000),
    ('u', 1.00, 0.00, 1.00, 0.000, 0.104, 0.000),
    ('u', 1.12, 0.00, 0.97, 0.000, 0.100, 0.000),
]
# relaxed hand (metres): x = toward the thumb, y = along the forearm, z = toward the palm (fingers curl that way)
RELAXED_HAND = np.array([
    (0.0, 0.0, 0.0), (0.018, 0.022, 0.010), (0.030, 0.040, 0.020), (0.038, 0.058, 0.028), (0.042, 0.072, 0.034),
    (0.022, 0.085, 0.004), (0.024, 0.115, 0.014), (0.024, 0.133, 0.028), (0.022, 0.144, 0.042),
    (0.004, 0.088, 0.004), (0.004, 0.121, 0.016), (0.004, 0.140, 0.032), (0.004, 0.151, 0.048),
    (-0.013, 0.084, 0.004), (-0.014, 0.113, 0.016), (-0.014, 0.130, 0.030), (-0.014, 0.140, 0.044),
    (-0.028, 0.076, 0.004), (-0.031, 0.098, 0.013), (-0.032, 0.112, 0.024), (-0.032, 0.121, 0.035)])


def log(*a):
    print('[render]', *a, flush=True)


# ----------------------------------------------------------------------------------------------------------------------
# small math helpers
# ----------------------------------------------------------------------------------------------------------------------
def hex2rgb(h, default=(0.5, 0.5, 0.5)):
    try:
        h = str(h).strip().lstrip('#')
        if len(h) == 3:
            h = ''.join(c * 2 for c in h)
        return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    except Exception:
        return tuple(default)


def srgb2lin(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin2srgb(c):
    c = np.clip(np.asarray(c, np.float64), 0, None)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def mix(a, b, t):
    return tuple(float(x) * (1 - t) + float(y) * t for x, y in zip(a, b))


def shade(c, f):
    return tuple(min(1.0, max(0.0, float(x) * f)) for x in c)


def nrm(v, axis=-1):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, 1e-9)


def smoothstep(a, b, x):
    t = np.clip((np.asarray(x, np.float64) - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def gsmooth(a, sigma):
    """gaussian smoothing along axis 0 (numpy only)."""
    if sigma <= 0 or len(a) < 3:
        return a
    r = int(3 * sigma + 0.5)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.concatenate([np.repeat(a[:1], r, 0), a, np.repeat(a[-1:], r, 0)], 0)
    out = np.zeros_like(a, dtype=np.float64)
    for i, w in enumerate(k):
        out += w * pad[i:i + len(a)]
    return out


def umeyama(src, dst):
    """batched similarity: dst ~ s R src + t. src [K,3], dst [T,K,3] -> M [T,4,4]."""
    mu_s = src.mean(0)
    mu_d = dst.mean(1)
    S = src - mu_s
    D = dst - mu_d[:, None]
    cov = np.einsum('tki,kj->tij', D, S) / len(src)
    U, sig, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U @ Vt))
    E = np.tile(np.eye(3), (len(dst), 1, 1))
    E[:, 2, 2] = d
    R = U @ E @ Vt
    var = (S ** 2).sum() / len(src)
    s = (sig * np.stack([np.ones_like(d), np.ones_like(d), d], 1)).sum(1) / var
    M = np.tile(np.eye(4), (len(dst), 1, 1))
    M[:, :3, :3] = R * s[:, None, None]
    M[:, :3, 3] = mu_d - s[:, None] * np.einsum('tij,j->ti', R, mu_s)
    return M


def apply(M, P):
    return P @ M[:3, :3].T + M[:3, 3]


def frame_basis(a):
    """orthonormal u, v perpendicular to unit vectors a [N,3]."""
    h = np.where(np.abs(a[:, 1:2]) < 0.9, np.array([[0, 1.0, 0]]), np.array([[1.0, 0, 0]]))
    u = nrm(np.cross(a, h))
    v = np.cross(a, u)
    return u, v


# ----------------------------------------------------------------------------------------------------------------------
# mesh building helpers (numpy topology; positions updated per frame with foreach_set)
# ----------------------------------------------------------------------------------------------------------------------
_MATS = {}


def material(rgb, name=None, rough=0.6, metal=0.0):
    key = (tuple(round(float(c), 4) for c in rgb), rough, metal)
    if key in _MATS:
        return _MATS[key]
    m = bpy.data.materials.new(name or ('m_%02x%02x%02x' % tuple(int(c * 255) for c in rgb)))
    lin = srgb2lin(rgb)
    m.diffuse_color = (float(lin[0]), float(lin[1]), float(lin[2]), 1.0)
    m.roughness = rough
    m.metallic = metal
    _MATS[key] = m
    return m


def make_obj(name, verts, faces, mats, mat_idx=None, parent=None, smooth=True, coll=None, shadow=False):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, v)) for v in verts], [], [tuple(map(int, f)) for f in faces])
    for m in mats:
        me.materials.append(m)
    if mat_idx is not None and len(me.polygons):
        me.polygons.foreach_set('material_index', np.asarray(mat_idx, np.int32))
    if len(me.polygons):
        me.polygons.foreach_set('use_smooth', np.full(len(me.polygons), bool(smooth)))
    me.update()
    ob = bpy.data.objects.new(name, me)
    (coll or bpy.context.scene.collection).objects.link(ob)
    if parent is not None:
        ob.parent = parent
        ob.matrix_parent_inverse = Matrix.Identity(4)
    ob.display.show_shadows = bool(shadow)
    return ob


def set_verts(ob, V):
    me = ob.data
    me.vertices.foreach_set('co', np.ascontiguousarray(V, np.float32).ravel())
    me.update()


def empty(name, parent=None, coll=None):
    e = bpy.data.objects.new(name, None)
    (coll or bpy.context.scene.collection).objects.link(e)
    if parent is not None:
        e.parent = parent
        e.matrix_parent_inverse = Matrix.Identity(4)
    return e


def latlong(n_az, polar):
    """unit sphere around +z pole. polar: increasing angles in (0, pi). returns verts, faces, ring index per face."""
    V = [(0, 0, 1.0)]
    for th in polar:
        for k in range(n_az):
            ph = 2 * math.pi * k / n_az
            V.append((math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)))
    V.append((0, 0, -1.0))
    F, R = [], []
    nr = len(polar)
    for k in range(n_az):
        F.append((0, 1 + k, 1 + (k + 1) % n_az)); R.append(0)
    for i in range(nr - 1):
        a, b = 1 + i * n_az, 1 + (i + 1) * n_az
        for k in range(n_az):
            k2 = (k + 1) % n_az
            F.append((a + k, b + k, b + k2, a + k2)); R.append(i + 1)
    last = len(V) - 1
    a = 1 + (nr - 1) * n_az
    for k in range(n_az):
        F.append((a + k, last, a + (k + 1) % n_az)); R.append(nr)
    return np.array(V, np.float64), F, R


def sphere_unit(n_az=12, n_pol=7):
    V, F, _ = latlong(n_az, [math.pi * (i + 1) / (n_pol + 1) for i in range(n_pol)])
    return V, F


class Tubes:
    """fixed-topology bundle of open cylinders (segments) and spheres; positions recomputed per frame."""

    def __init__(self, nseg, nsph, mat_seg=None, mat_sph=None, n=12, sph=(12, 7)):
        self.n = n
        self.nseg, self.nsph = nseg, nsph
        self.sv, sf = sphere_unit(*sph)
        F, M = [], []
        for s in range(nseg):
            b = s * 2 * n
            for k in range(n):
                k2 = (k + 1) % n
                F.append((b + k, b + k2, b + n + k2, b + n + k)); M.append(mat_seg[s] if mat_seg else 0)
        base = nseg * 2 * n
        for j in range(nsph):
            off = base + j * len(self.sv)
            for f in sf:
                F.append(tuple(off + i for i in f)); M.append(mat_sph[j] if mat_sph else 0)
        self.faces, self.mats = F, M
        self.nv = base + nsph * len(self.sv)
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        self.cs = np.stack([np.cos(ang), np.sin(ang)], 1)

    def verts(self, p0, p1, r0, r1, c, r):
        out = []
        if self.nseg:
            p0 = np.asarray(p0, np.float64).reshape(-1, 3); p1 = np.asarray(p1, np.float64).reshape(-1, 3)
            r0 = np.asarray(r0, np.float64).reshape(-1); r1 = np.asarray(r1, np.float64).reshape(-1)
            a = p1 - p0
            ln = np.linalg.norm(a, axis=1, keepdims=True)
            a = np.where(ln > 1e-7, a / np.maximum(ln, 1e-9), np.array([[0, 1.0, 0]]))
            u, v = frame_basis(a)
            ring = self.cs[None, :, 0:1] * u[:, None] + self.cs[None, :, 1:2] * v[:, None]   # [S,n,3]
            A = p0[:, None] + ring * r0[:, None, None]
            B = p1[:, None] + ring * r1[:, None, None]
            out.append(np.concatenate([A, B], 1).reshape(-1, 3))
        if self.nsph:
            c = np.asarray(c, np.float64).reshape(-1, 3); r = np.asarray(r, np.float64).reshape(-1)
            out.append((c[:, None] + self.sv[None] * r[:, None, None]).reshape(-1, 3))
        return np.concatenate(out, 0)


def ellipsoid_mesh(center, axes, rot=None, n_az=16, n_pol=9, power=1.0):
    V, F = sphere_unit(n_az, n_pol)
    if power != 1.0:
        V = np.sign(V) * np.abs(V) ** power
    V = V * np.asarray(axes)
    if rot is not None:
        V = V @ np.asarray(rot).T
    return V + np.asarray(center), F


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def tube_path(pts, radius, n=8, closed=False):
    """swept circle along a polyline -> verts, faces."""
    pts = np.asarray(pts, np.float64)
    m = len(pts)
    rad = np.broadcast_to(np.asarray(radius, np.float64), (m,))
    if closed:
        tang = nrm(np.roll(pts, -1, 0) - np.roll(pts, 1, 0))
    else:
        tang = nrm(np.gradient(pts, axis=0))
    u, v = frame_basis(tang)
    # parallel-transport-ish: keep u continuous
    for i in range(1, m):
        if np.dot(u[i], u[i - 1]) < 0:
            u[i] = -u[i]; v[i] = -v[i]
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    V = (pts[:, None] + rad[:, None, None] * (np.cos(ang)[None, :, None] * u[:, None] + np.sin(ang)[None, :, None] * v[:, None])).reshape(-1, 3)
    F = []
    segs = m if closed else m - 1
    for i in range(segs):
        i2 = (i + 1) % m
        for k in range(n):
            k2 = (k + 1) % n
            F.append((i * n + k, i * n + k2, i2 * n + k2, i2 * n + k))
    if not closed:  # caps
        c0 = len(V); c1 = c0 + 1
        V = np.vstack([V, pts[0], pts[-1]])
        for k in range(n):
            F.append((c0, (k + 1) % n, k))
            F.append((c1, (m - 1) * n + k, (m - 1) * n + (k + 1) % n))
    return V, F


class MeshAcc:
    """accumulate static parts into one mesh with material slots."""

    def __init__(self):
        self.V, self.F, self.M = [], [], []
        self.n = 0
        self.mats = []

    def mat(self, m):
        if m not in self.mats:
            self.mats.append(m)
        return self.mats.index(m)

    def add(self, V, F, m):
        mi = self.mat(m)
        V = np.asarray(V, np.float64)
        self.V.append(V)
        self.F.extend(tuple(self.n + i for i in f) for f in F)
        self.M.extend([mi] * len(F))
        self.n += len(V)

    def build(self, name, parent=None, coll=None, smooth=True):
        if not self.V:
            return None
        return make_obj(name, np.concatenate(self.V, 0), self.F, self.mats, self.M, parent=parent, coll=coll, smooth=smooth)


# ----------------------------------------------------------------------------------------------------------------------
# skull / hair helpers (canonical frame)
# ----------------------------------------------------------------------------------------------------------------------
def skull_r(u):
    """radius of the canonical skull egg along unit directions u [...,3]."""
    ay = np.where(u[..., 1] >= 0, SK_AYU, SK_AYD)
    az = np.where(u[..., 2] >= 0, SK_AZF, SK_AZB)
    return 1.0 / np.sqrt((u[..., 0] / SK_AX) ** 2 + (u[..., 1] / ay) ** 2 + (u[..., 2] / az) ** 2)


def hairline_fn(knots_uz, knots_uy):
    def f(uz):
        return np.interp(uz, knots_uz, knots_uy)
    return f


def hair_cap(scale, hl, n_az=64, n_rows=14, tuck=0.97, extra=None):
    """shell over the skull from the crown down to a hairline; hl(uz_horizontal) gives uy of the lower edge."""
    V = [SK_C + np.array([0, 1.0, 0]) * SK_AYU * scale]
    for i in range(1, n_rows + 1):
        for k in range(n_az):
            ph = 2 * math.pi * k / n_az
            hz, hx = math.cos(ph), math.sin(ph)
            th_max = math.acos(float(np.clip(hl(hz), -0.98, 0.98)))
            th = th_max * (i / n_rows)
            u = np.array([math.sin(th) * hx, math.cos(th), math.sin(th) * hz])
            r = skull_r(u) * scale
            if extra is not None:
                r = r + extra(u)
            if i == n_rows:
                r *= tuck
            V.append(SK_C + u * r)
    F = []
    for k in range(n_az):
        F.append((0, 1 + (k + 1) % n_az, 1 + k))
    for i in range(n_rows - 1):
        a, b = 1 + i * n_az, 1 + (i + 1) * n_az
        for k in range(n_az):
            k2 = (k + 1) % n_az
            F.append((a + k, a + k2, b + k2, b + k))
    return np.array(V), F


def hair_hang(scale, y_top, length, phi_open=1.2, wav=0.0, flare=0.02, n_az=48, n_rows=18, thick=0.012):
    """hair hanging from the temples/back down to y_top-length, open in front of the face (|phi| < phi_open).
    Two shells (outer + inner) joined at the edges so it reads as a solid curtain."""
    phis = np.linspace(phi_open, 2 * math.pi - phi_open, n_az)
    ys = np.linspace(y_top, y_top - length, n_rows)
    out = []
    for layer, dr in ((0, 0.0), (1, -thick)):
        for y in ys:
            dy = y - SK_C[1]
            for ph in phis:
                hx, hz = math.sin(ph), math.cos(ph)
                az = SK_AZF if hz >= 0 else SK_AZB
                rr = 1.0 / math.sqrt((hx / SK_AX) ** 2 + (hz / az) ** 2)
                if dy > 0:
                    r = rr * math.sqrt(max(0.05, 1 - (dy / SK_AYU) ** 2)) * scale
                else:
                    r = rr * scale + flare * (-dy) / max(length, 1e-3)
                r += wav * math.sin(dy * 70 + ph * 3) + dr
                edge = min(ph - phis[0], phis[-1] - ph)
                r -= 0.004 * max(0.0, 1 - edge / 0.25)
                out.append(SK_C + np.array([hx * r, dy, hz * r]))
    V = np.array(out)
    na, nr = len(phis), len(ys)
    F = []
    for L in range(2):
        o = L * na * nr
        for i in range(nr - 1):
            for k in range(na - 1):
                q = (o + i * na + k, o + i * na + k + 1, o + (i + 1) * na + k + 1, o + (i + 1) * na + k)
                F.append(q if L == 0 else q[::-1])
    o2 = na * nr
    for i in range(nr - 1):
        for k in (0, na - 1):
            F.append((i * na + k, (i + 1) * na + k, o2 + (i + 1) * na + k, o2 + i * na + k))
    for k in range(na - 1):
        i = nr - 1
        F.append((i * na + k, i * na + k + 1, o2 + i * na + k + 1, o2 + i * na + k))
    return V, F


def fib_dirs(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.cos(phi), np.sin(th) * np.sin(phi)], 1)   # y = pole


# ----------------------------------------------------------------------------------------------------------------------
# rig loading / cleaning
# ----------------------------------------------------------------------------------------------------------------------
def load_rig(path):
    d = np.load(path, allow_pickle=False)
    R = {k: d[k] for k in d.files}
    W, H = int(R['W']), int(R['H'])
    P, T = int(R['P']), int(R['T'])
    out = dict(W=W, H=H, P=P, T=T, fps=float(R.get('fps', 30.0)), vfov=float(R.get('vfov_deg', 63.0)))

    def arr(name, shape, dtype=np.float64):
        a = R.get(name)
        if a is None or tuple(a.shape[:2]) != (P, T):
            return np.zeros(shape, dtype)
        a = np.asarray(a, dtype)
        return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0) if dtype != bool else a

    out['head_M_raw'] = R.get('head_M', np.zeros((P, T, 4, 4))).astype(np.float64)
    out['chans'] = arr('chans', (P, T, 14))
    out['lips_outer'] = arr('lips_outer', (P, T, 20, 2))
    out['lips_inner'] = arr('lips_inner', (P, T, 20, 2))
    out['mouth_center'] = arr('mouth_center', (P, T, 3))
    out['face_local'] = R.get('face_local', np.zeros((P, T, 478, 3))).astype(np.float64)
    out['joints'] = R.get('joints', np.zeros((P, T, 8, 3))).astype(np.float64)
    out['joint_vis'] = arr('joint_vis', (P, T, 8))
    out['hands'] = R.get('hands', np.zeros((P, T, 2, 21, 3))).astype(np.float64)
    hv = R.get('hand_valid')
    out['hand_valid'] = np.asarray(hv, bool) if hv is not None and hv.shape[:2] == (P, T) else np.zeros((P, T, 2), bool)
    names = [str(c) for c in R.get('channels', [])]
    out['chan_names'] = names
    return out


def chan(rig, p, t, name, default):
    names = rig['chan_names']
    if name in names:
        v = float(rig['chans'][p, t, names.index(name)])
        if math.isfinite(v):
            return v
    return default


def clean_head(Mraw, T):
    """NaN / zero / degenerate matrices -> last good (first good for the leading gap)."""
    M = np.array(Mraw, np.float64)
    ok = np.isfinite(M).all((1, 2))
    det = np.zeros(T)
    det[ok] = np.linalg.det(np.nan_to_num(M[ok][:, :3, :3]))
    ok &= np.abs(det) > 1e-4
    ok &= np.abs(M[:, 2, 3]) > 0.05
    if not ok.any():
        D = np.eye(4); D[:3, 3] = (0, 0.05, -0.7)
        return np.tile(D, (T, 1, 1)), ok
    first = int(np.argmax(ok))
    last = M[first].copy()
    for t in range(T):
        if ok[t]:
            last = M[t].copy()
        else:
            M[t] = last
    for t in range(T):
        R = M[t, :3, :3]
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            R = -R
        M[t, :3, :3] = R
        M[t, 3] = (0, 0, 0, 1)
    return M, ok


# ----------------------------------------------------------------------------------------------------------------------
# scene description defaults
# ----------------------------------------------------------------------------------------------------------------------
def g(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d or d[k] is None:
            return default
        d = d[k]
    return d


def fnum(v, default):
    try:
        v = float(v)
        return v if math.isfinite(v) else default
    except Exception:
        return default


# ----------------------------------------------------------------------------------------------------------------------
# Person builder
# ----------------------------------------------------------------------------------------------------------------------
class Person:
    def __init__(self, R, p, look, coll, ctx):
        self.R, self.p, self.look, self.coll, self.ctx = R, p, look or {}, coll, ctx
        T = R['T']
        self.T = T
        self.HM, _ = clean_head(R['head_M_raw'][p], T)
        fl = R['face_local'][p]
        good = np.isfinite(fl).all((1, 2)) & (np.abs(np.nan_to_num(fl)).sum((1, 2)) > 1e-6)
        self.face_mode = bool(good.mean() > 0.2)
        canon = np.repeat(TEMPLATE[None], T, 0)
        A = np.tile(np.eye(4), (T, 1, 1))
        if self.face_mode:
            fl = np.nan_to_num(fl)
            # hold last good for bad frames
            first = int(np.argmax(good)); lastv = fl[first].copy()
            for t in range(T):
                if good[t]:
                    lastv = fl[t]
                else:
                    fl[t] = lastv
            A = umeyama(TEMPLATE[FIT_IDX], fl[:, FIT_IDX])
            # smooth the fit a little (translation + linear part), re-orthogonalise with the fitted scale
            A[:, :3, :] = gsmooth(A[:, :3, :], 1.0)
            for t in range(T):
                L = A[t, :3, :3]
                s = np.cbrt(abs(np.linalg.det(L)))
                U, _, Vt = np.linalg.svd(L / s)
                A[t, :3, :3] = (U @ Vt) * s
            Ainv = np.linalg.inv(A)
            canon = np.einsum('tij,tnj->tni', Ainv[:, :3, :3], fl) + Ainv[:, None, :3, 3]
            # lips arrays (relative to mouth_center, head-local) override the lip contour x,y when present
            lo, li, mc = R['lips_outer'][p], R['lips_inner'][p], R['mouth_center'][p]
            haslip = (np.abs(lo).sum((1, 2)) > 1e-6) & (np.abs(mc).sum(1) > 1e-6)
            for idx, arr2 in ((LIPS_OUT, lo), (LIPS_IN, li)):
                hl = fl[:, idx].copy()
                hl[:, :, :2] = np.where(haslip[:, None, None], mc[:, None, :2] + arr2, hl[:, :, :2])
                canon[:, idx] = np.einsum('tij,tnj->tni', Ainv[:, :3, :3], hl) + Ainv[:, None, :3, 3]
            self.scale = float(np.median(np.cbrt(np.abs(np.linalg.det(A[:, :3, :3])))))
        else:
            self.scale = 1.0
        self.A = A
        self.canon = canon
        self.rest = np.median(canon, 0)   # person's rest face in canonical frame
        self.build()

    # ---------------------------------------------------------------- look helpers
    def col(self, *keys, default='#808080'):
        return hex2rgb(g(self.look, *keys, default=default) or default, hex2rgb(default))

    # ---------------------------------------------------------------- build
    def build(self):
        L = self.look
        C = self.coll
        self.skin = self.col('skin', default='#d9a888')
        self.lipc = self.col('lip_color', default='#b86a62')
        self.hairc = self.col('hair', 'color', default='#3a2a20')
        self.hair_style = str(g(L, 'hair', 'style', default='short') or 'short').lower()
        self.fh_style = str(g(L, 'facial_hair', 'style', default='none') or 'none').lower()
        self.fhc = self.col('facial_hair', 'color', default=g(L, 'hair', 'color', default='#3a2a20') or '#3a2a20')
        self.build_f = {'slim': 0.9, 'average': 1.0, 'broad': 1.08, 'heavy': 1.2}.get(str(g(L, 'build', default='average')).lower(), 1.0)
        top = g(L, 'top', default={}) or {}
        self.top_type = str(top.get('type') or 'tshirt').lower()
        self.top_c = hex2rgb(top.get('color') or '#4a5a78')
        self.top_c2 = hex2rgb(top.get('secondary_color') or top.get('color') or '#4a5a78')
        self.pattern = str(top.get('pattern') or 'solid').lower()
        self.pat_c = hex2rgb(top.get('pattern_color') or '#ffffff')
        self.headwear = str(g(L, 'headwear', default='none') or 'none').lower()
        self.headwear_c = self.col('headwear_color', default='#303030')
        hs = fnum(g(L, 'head_scale', default=1.0), 1.0)
        self.head_scale = min(1.15, max(0.85, hs))

        self.m_skin = material(self.skin, rough=0.55)
        self.m_skin_d = material(shade(self.skin, 0.84), rough=0.6)
        self.m_lip = material(self.lipc, rough=0.4)
        self.m_hair = material(self.hairc, rough=0.7)
        self.m_hair_d = material(shade(self.hairc, 0.75), rough=0.7)

        self.head = empty('head%d' % self.p, coll=C)
        self.F = empty('face%d' % self.p, parent=self.head, coll=C)
        self.build_head_mesh()
        self.build_eyes()
        self.build_brows()
        self.build_mouth()
        self.build_ears()
        self.build_hair()
        self.build_glasses()
        self.build_body()

    # ---------------------------------------------------------------- head mesh: mask + back shell
    def build_head_mesh(self):
        X = TEMPLATE
        cen = X[TRIS].mean(1)

        def inside(poly_idx, pts):
            poly = X[poly_idx][:, :2]
            x, y = pts[:, 0], pts[:, 1]
            ins = np.zeros(len(pts), bool)
            n = len(poly)
            for i in range(n):
                x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % n]
                c = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1)
                ins ^= c
            return ins

        eye_hole = inside(EYE_R['ring'], cen) | inside(EYE_L['ring'], cen)
        mouth_hole = inside(LIPS_IN, cen)
        lip = inside(LIPS_OUT, cen) & ~mouth_hole
        keep = ~eye_hole & ~mouth_hole
        tris = TRIS[keep]
        cen = cen[keep]
        lip = lip[keep]
        # face-region materials
        mi = np.zeros(len(tris), np.int64)
        mats = [self.m_skin, self.m_lip, self.m_skin_d, self.m_hair]
        mi[lip] = 1
        # eyelid crease band: triangles touching the upper eye contour (outside the hole)
        upper = set(EYE_R['up'] + EYE_L['up'] + [EYE_R['c0'], EYE_R['c1'], EYE_L['c0'], EYE_L['c1']])
        touch = np.array([any(int(v) in upper for v in tr) for tr in tris])
        mi[touch & ~lip] = 2
        # hairline fringe on the mask (bangs lower it)
        st = self.hair_style
        if st not in ('bald', 'buzz') and bool(g(self.look, 'hair', 'bangs', default=False)):
            mi[(cen[:, 1] > 0.062) & (mi == 0)] = 3
        fh = self.fh_style
        # back shell topology
        nO = len(OVAL)
        base = 478
        F = [tuple(t) for t in tris]
        M = list(mi)
        rings = [OVAL] + [[base + (k - 1) * nO + i for i in range(nO)] for k in range(1, NRING)]
        pole = base + (NRING - 1) * nO
        shellM = []
        for k in range(NRING - 1):
            a, b = rings[k], rings[k + 1]
            for i in range(nO):
                i2 = (i + 1) % nO
                F.append((a[i], b[i], b[i2], a[i2])); shellM.append(0)
        for i in range(nO):
            F.append((rings[-1][i], pole, rings[-1][(i + 1) % nO])); shellM.append(0)
        M.extend(shellM)
        self.nhead = pole + 1
        V0 = self.head_verts(0)
        # orientation check: shell normals outward
        Vt = V0
        f = F[len(tris)]
        nrmv = np.cross(Vt[f[1]] - Vt[f[0]], Vt[f[2]] - Vt[f[0]])
        if np.dot(nrmv, Vt[f[0]] - SK_C) < 0:
            F = F[:len(tris)] + [tuple(reversed(q)) for q in F[len(tris):]]
        self.head_faces = F
        if fh != 'none':
            Vc = V0.copy()
            Vc[:478] = TEMPLATE
            bw = self.beard_weights(Vc, fh)
            fw = np.array([bw[list(q)].mean() for q in F])
            mats.append(material(mix(self.skin, self.fhc, 0.45) if fh == 'stubble' else self.fhc, rough=0.85))
            M = np.array(M)
            M[(fw > 0.45) & (M == 0)] = len(mats) - 1
        self.head_obj = make_obj('headmesh%d' % self.p, V0, F, mats, M, parent=self.F, coll=self.coll)
        mod = self.head_obj.modifiers.new('sub', 'SUBSURF')
        mod.levels = mod.render_levels = 0 if 'subsurf' in os.environ.get('RB_OFF', '') else 1
        try:
            mod.boundary_smooth = 'PRESERVE_CORNERS'
        except Exception:
            pass
        # facial hair: smooth per-vertex weight over the canonical face -> mask colour + tapered shell
        self.beard = None
        if fh != 'none':
            w = bw
            if fh != 'stubble':
                bf = [F[i] for i in np.nonzero(np.array([bw[list(q)].max() for q in F]) > 0.02)[0]]
                used = sorted(set(i for q in bf for i in q))
                remap = {v: i for i, v in enumerate(used)}
                self.beard_idx = np.array(used)
                self.beard_w = w[used] ** 0.6
                self.beard_off = {'full_beard': 0.010, 'short_beard': 0.005, 'goatee': 0.006, 'mustache': 0.004}.get(fh, 0.006)
                self.beard = make_obj('beard%d' % self.p, V0[used], [tuple(remap[i] for i in q) for q in bf],
                                      [mats[-1]], parent=self.F, coll=self.coll)
                bm = self.beard.modifiers.new('sub', 'SUBSURF')
                bm.levels = bm.render_levels = 1
        self.head_mats = mats

    def beard_weights(self, V, fh):
        """0..1 facial-hair weight per vertex (canonical coords)."""
        x, y, z = V[:, 0], V[:, 1], V[:, 2]
        ax = np.abs(x)
        yl = np.interp(ax, [0, 0.02, 0.035, 0.05, 0.065, 0.08, 0.095], [-0.028, -0.031, -0.030, -0.022, -0.010, 0.008, 0.025])
        w = smoothstep(yl + 0.004, yl - 0.004, y)
        # lips stay clear
        poly = TEMPLATE[LIPS_OUT][:, :2]
        ins = np.zeros(len(V), bool)
        for i in range(len(poly)):
            x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % len(poly)]
            ins ^= ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1)
        d = np.min(np.linalg.norm(V[:, None, :2] - poly[None], axis=2), 1)
        w *= np.where(ins, 0.0, smoothstep(0.0015, 0.0055, d))
        w *= smoothstep(-0.115, -0.09, z)            # front half only (no beard on the nape)
        if fh == 'goatee':
            w *= smoothstep(0.036, 0.024, ax)
        elif fh == 'mustache':
            w *= smoothstep(0.034, 0.026, ax) * smoothstep(-0.045, -0.036, y)
        return np.clip(w, 0, 1)

    def head_verts(self, t, canon=None):
        X = self.canon[t] if canon is None else canon
        oval = X[OVAL]
        d = oval - SK_C
        r_rim = np.linalg.norm(d, axis=1)
        u0 = d / r_rim[:, None]
        pole = np.array([0, -0.05, -1.0]); pole /= np.linalg.norm(pole)
        out = [X[:478]]
        cosw = np.clip(u0 @ pole, -1, 1)
        om = np.arccos(cosw)[:, None]
        so = np.maximum(np.sin(om), 1e-6)
        for k in range(1, NRING):
            s = k / NRING
            u = (np.sin((1 - s) * om) / so) * u0 + (np.sin(s * om) / so) * pole
            u = nrm(u)
            b = smoothstep(0.0, 0.45, s)[..., None] if np.ndim(s) else float(smoothstep(0.0, 0.45, s))
            r = (1 - b) * r_rim + b * skull_r(u)
            out.append(SK_C + u * r[:, None])
        out.append((SK_C + pole * skull_r(pole[None])[0])[None])
        return np.concatenate(out, 0)

    # ---------------------------------------------------------------- eyes
    def build_eyes(self):
        self.eyes = []
        rest = self.rest
        for side, E in (('L', EYE_L), ('R', EYE_R)):
            ring = rest[E['ring']]
            mxy = ring[:, :2].mean(0)
            w = np.linalg.norm(rest[E['c0']] - rest[E['c1']])
            r = float(np.clip(0.42 * w, 0.0105, 0.0135))
            m = 0.0012
            dxy = np.linalg.norm(ring[:, :2] - mxy, axis=1)
            z0 = float(np.min(ring[:, 2] - np.sqrt(np.maximum(0, (r + m) ** 2 - dxy ** 2))))
            c = np.array([mxy[0], mxy[1], z0])
            ir = math.asin(min(0.95, 0.0058 / r))
            pr = math.asin(min(0.9, 0.0026 / r))
            pol = [pr * 0.5, pr, ir * 0.55, ir * 0.85, ir, ir * 1.04] + list(np.linspace(ir * 1.04 + 0.15, math.pi - 0.2, 8))
            V, F, Rg = latlong(24, pol)
            # ring index -> material: 0,1 pupil ; 2..4 iris ; 5 iris rim ; rest white
            eyec = self.col('eye_color', default='#4a3020')
            mats = [material((0.04, 0.03, 0.03), rough=0.2), material(eyec, rough=0.3),
                    material(shade(eyec, 0.6), rough=0.3), material((0.95, 0.94, 0.92), rough=0.3)]
            mi = [0 if k <= 1 else (1 if k <= 4 else (2 if k == 5 else 3)) for k in Rg]
            ob = make_obj('eye%s%d' % (side, self.p), V * r, F, mats, mi, parent=self.F, coll=self.coll)
            self.eyes.append(dict(ob=ob, c=c, r=r, E=E, side=side, gaze=np.zeros(2)))

    def eye_gaze(self, t, e):
        """yaw, pitch (radians) from iris landmark (face mode) or channels."""
        p = self.p
        idx = 0 if e['side'] == 'L' else 1
        eo = chan(self.R, p, t, 'eye_open_' + e['side'], 1.0)
        if self.face_mode:
            ir = self.canon[t, e['E']['iris']]
            d = ir[:2] - e['c'][:2]
            rr = e['r'] * 0.92
            yaw = math.atan2(float(d[0]), rr)
            pitch = math.atan2(float(d[1]), rr)
            if eo < 0.35 and t > 0:
                return e['gaze']
            g_ = np.clip([yaw, pitch], -0.7, 0.7)
        else:
            g_ = np.array([chan(self.R, p, t, 'gaze_yaw', 0.0), chan(self.R, p, t, 'gaze_pitch', 0.0)])
        e['gaze'] = g_
        return g_

    # ---------------------------------------------------------------- brows
    def build_brows(self):
        n = 5
        F = []
        for b in range(2):
            o = b * 4 * n   # 4 rows: top-front, bottom-front, top-back, bottom-back
            for i in range(n - 1):
                tf, bf, tb, bb = o, o + n, o + 2 * n, o + 3 * n
                F.append((tf + i, bf + i, bf + i + 1, tf + i + 1))      # front
                F.append((tb + i, tb + i + 1, tf + i + 1, tf + i))      # top
                F.append((bb + i, bf + i, bf + i + 1, bb + i + 1)[::-1])  # bottom
            F.append((tf, tb, bb, bf))
            F.append((tf + n - 1, bf + n - 1, bb + n - 1, tb + n - 1))
        bc = self.col('brow_color', default=g(self.look, 'hair', 'color', default='#3a2a20') or '#3a2a20')
        self.brow_thick = 0.38 + 0.55 * min(1, max(0, fnum(g(self.look, 'brow_thickness', default=0.5), 0.5)))
        self.brow_obj = make_obj('brows%d' % self.p, self.brow_verts(0), F, [material(bc, rough=0.8)], parent=self.F, coll=self.coll)
        # lash lines along the upper lids (read clearly as a closed eye on blinks)
        V, F = self.lash_verts(self.canon[0])
        self.lash_obj = make_obj('lashes%d' % self.p, V, F, [material((0.10, 0.07, 0.06), rough=0.8)], parent=self.F, coll=self.coll)

    def lash_verts(self, X):
        Vs, Fs, off = [], [], 0
        for E in (EYE_L, EYE_R):
            pts = np.vstack([X[E['c0']], X[E['up']], X[E['c1']]]) + np.array([0, 0.0004, 0.0012])
            rad = 0.0011 * np.sin(np.linspace(0.25, np.pi - 0.25, len(pts))) + 0.0003
            V, F = tube_path(pts, rad, 6)
            Vs.append(V); Fs += [tuple(i + off for i in f) for f in F]; off += len(V)
        return np.vstack(Vs), Fs

    def brow_verts(self, t):
        X = self.canon[t]
        out = []
        for (up, lo), side in ((BROW_L, 'L'), (BROW_R, 'R')):
            U, Lw = X[up].copy(), X[lo].copy()
            if not self.face_mode:
                raise_ = chan(self.R, self.p, t, 'brow_raise_' + side, 0.0)
                inner = chan(self.R, self.p, t, 'brow_inner', 0.0)
                w = np.linspace(0, 1, 5)   # outer -> inner
                dy = 0.006 * raise_ + 0.004 * inner * w
                U[:, 1] += dy; Lw[:, 1] += dy
            c = (U + Lw) / 2
            h = (U - Lw) / 2 * self.brow_thick
            h[:, 1] = np.maximum(h[:, 1], 0.0018)
            top, bot = c + h, c - h
            fz = np.array([0, 0, 0.0028]); bz = np.array([0, 0, -0.002])
            out += [top + fz, bot + fz, top + bz, bot + bz]
        return np.concatenate(out, 0)

    # ---------------------------------------------------------------- mouth interior + teeth
    def build_mouth(self):
        F, M = [], []
        for j in range(10):
            for r in range(3):
                a = j * 4 + r
                F.append((a, a + 4, a + 5, a + 1)); M.append(1 if r == 1 else 0)
        mats = [material((0.94, 0.92, 0.86), rough=0.3), material((0.16, 0.05, 0.06), rough=0.9)]
        self.mouth_obj = make_obj('mouth%d' % self.p, self.mouth_verts(0), F, mats, M, parent=self.F, coll=self.coll, smooth=False)

    def mouth_verts(self, t):
        X = self.canon[t]
        inn = X[LIPS_IN]
        cols = []
        mc = inn.mean(0)
        for j in range(11):
            if j == 0 or j == 10:
                up = lo = inn[j]
            else:
                lo, up = inn[j], inn[20 - j]
            gap = max(1e-5, up[1] - lo[1])
            tu = min(0.0035, 0.42 * gap) / gap
            tl = min(0.002, 0.22 * gap) / gap
            r0 = up
            r1 = up + (lo - up) * tu
            r2 = lo - (lo - up) * tl
            r3 = lo
            cols.append([r0, r1, r2, r3])
        V = np.array(cols).reshape(-1, 3)
        # enlarge a little around the centre (hidden by the lips) and push back
        V[:, 0] = mc[0] + (V[:, 0] - mc[0]) * 1.12
        V[:, 1] = mc[1] + (V[:, 1] - mc[1]) * 1.15
        V[:, 2] = V[:, 2] - 0.003
        return V

    # ---------------------------------------------------------------- ears
    def build_ears(self):
        acc = MeshAcc()
        rest = self.rest
        for sgn, i_side in ((1, 454), (-1, 234)):
            side = rest[i_side]
            c = np.array([sgn * (abs(side[0]) + 0.004), side[1] - 0.004, side[2] - 0.016])
            Rm = rot_x(-0.25) @ rot_y(sgn * 0.35)
            V, F = ellipsoid_mesh(c, (0.010, 0.030, 0.019), Rm, 16, 9)
            acc.add(V, F, self.m_skin)
            V, F = ellipsoid_mesh(c + Rm @ np.array([sgn * 0.0055, 0.002, 0.001]), (0.004, 0.019, 0.011), Rm, 12, 7)
            acc.add(V, F, self.m_skin_d)
            self.ear_low = getattr(self, 'ear_low', []) + [c + Rm @ np.array([0, -0.027, 0.002])]
        self.ears = acc.build('ears%d' % self.p, parent=self.F, coll=self.coll)
        if bool(g(self.look, 'earrings', default=False)):
            acc = MeshAcc()
            gold = material((0.86, 0.70, 0.30), rough=0.25, metal=0.6)
            for c in self.ear_low:
                pts = [c + np.array([0, -0.017 - 0.017 * math.cos(a), 0.017 * math.sin(a)]) for a in np.linspace(0, 2 * math.pi, 24, endpoint=False)]
                V, F = tube_path(pts, 0.0017, 6, closed=True)
                acc.add(V, F, gold)
            acc.build('earrings%d' % self.p, parent=self.F, coll=self.coll)

    # ---------------------------------------------------------------- hair & headwear
    def build_hair(self):
        st = self.hair_style
        vol = min(1, max(0, fnum(g(self.look, 'hair', 'volume', default=0.5), 0.5)))
        ln = min(1, max(0, fnum(g(self.look, 'hair', 'length', default=0.5), 0.5)))
        bangs = bool(g(self.look, 'hair', 'bangs', default=False))
        part = str(g(self.look, 'hair', 'part', default='none') or 'none').lower()
        acc = MeshAcc()
        mh, md = self.m_hair, self.m_hair_d
        hf = 0.36 if bangs else 0.70
        std = hairline_fn([-1, -0.3, 0.1, 0.32, 1], [-0.62, -0.35, 0.18, hf, hf])
        thick = 1.035 + 0.05 * vol
        rng = np.random.RandomState(7 + self.p)

        def bumps(n, scale, r0, r1, hl, mat=mh, jitter=0.0):
            D = fib_dirs(n)
            hz = D[:, 2] / np.maximum(np.linalg.norm(D[:, [0, 2]], axis=1), 1e-6)
            keep = D[:, 1] > hl(hz) + 0.04
            for u in D[keep]:
                rr = rng.uniform(r0, r1)
                c = SK_C + u * skull_r(u) * scale
                V, F = sphere_unit(10, 6)
                acc.add(V * rr + c, F, mat)

        def slab(c, axes, rot=None, power=0.75, mat=mh):
            V, F = ellipsoid_mesh(c, axes, rot, 16, 10, power)
            acc.add(V, F, mat)

        def part_line():
            if st == 'side_part' and part in ('left', 'right', 'center'):
                xo = {'left': 0.025, 'right': -0.025, 'center': 0.0}[part]
                pts = []
                for a in np.linspace(0.15, 0.95, 10):
                    u = nrm(np.array([xo / 0.09, math.cos(a * 1.2), math.sin(a * 1.2)]))
                    pts.append(SK_C + u * skull_r(u[None])[0] * (thick + 0.004))
                V, F = tube_path(pts, 0.0014, 5)
                acc.add(V, F, md)

        if st == 'bald':
            pass
        elif st == 'buzz':
            V, F = hair_cap(1.012, hairline_fn([-1, -0.3, 0.1, 0.32, 1], [-0.55, -0.3, 0.22, 0.62, 0.62]), tuck=0.995)
            acc.add(V, F, material(mix(self.hairc, self.skin, 0.35), rough=0.9))
        elif st in ('short', 'side_part'):
            V, F = hair_cap(thick, std)
            acc.add(V, F, mh)
            xo = {'left': -0.02, 'right': 0.02}.get(part, 0.0)
            slab(SK_C + np.array([xo, SK_AYU * 0.78, 0.035]), (0.07, 0.03 + 0.02 * vol, 0.06), rot_x(-0.5) @ rot_z(xo * 6))
            if st == 'side_part' or part in ('left', 'right'):
                part_line()
        elif st in ('medium', 'bob', 'long_straight', 'long_wavy'):
            V, F = hair_cap(thick + 0.01, std)
            acc.add(V, F, mh)
            Lh = {'medium': 0.10 + 0.08 * ln, 'bob': 0.13 + 0.12 * ln}.get(st, 0.18 + 0.30 * ln)
            V, F = hair_hang(thick + 0.02 + 0.03 * vol, 0.055, Lh + 0.055 + 0.012, phi_open=1.12 if st != 'medium' else 1.3,
                             wav=0.004 if st == 'long_wavy' else 0.0, flare=0.015 + 0.02 * vol)
            acc.add(V, F, mh)
            part_line()
        elif st == 'ponytail':
            V, F = hair_cap(thick, std)
            acc.add(V, F, mh)
            c = SK_C + nrm(np.array([0, 0.25, -1.0])) * SK_AZB * 1.08
            V, F = sphere_unit(12, 7)
            acc.add(V * 0.022 + c, F, mh)
            Lh = 0.12 + 0.22 * ln
            pts = [c + np.array([0, -s_ * Lh, -0.03 * math.sin(s_ * 2.5)]) for s_ in np.linspace(0, 1, 9)]
            V, F = tube_path(pts, np.linspace(0.024, 0.01, 9) * (0.8 + 0.4 * vol), 10)
            acc.add(V, F, mh)
        elif st == 'curly_short':
            V, F = hair_cap(1.03 + 0.02 * vol, std)
            acc.add(V, F, mh)
            bumps(170, 1.05 + 0.03 * vol, 0.010, 0.016, std)
        elif st == 'curly_big':
            hl = hairline_fn([-1, -0.3, 0.05, 0.3, 1], [-0.75, -0.6, -0.15, 0.55, 0.55])
            V, F = hair_cap(1.10 + 0.08 * vol, hl)
            acc.add(V, F, mh)
            bumps(420, 1.12 + 0.1 * vol, 0.012, 0.02, hl)
        elif st == 'afro':
            hl = hairline_fn([-1, -0.3, 0.05, 0.3, 1], [-0.6, -0.4, 0.0, 0.58, 0.58])
            sc = 1.30 + 0.25 * vol
            V, F = hair_cap(sc, hl, tuck=0.9)
            acc.add(V, F, mh)
            bumps(160, sc, 0.02, 0.032, hl)
        elif st == 'bun':
            V, F = hair_cap(1.04 + 0.07 * vol, std)
            acc.add(V, F, mh)
            u = nrm(np.array([0, 0.9, -0.42]))
            c = SK_C + u * skull_r(u[None])[0] * 1.08
            rr = 0.032 + 0.018 * vol
            slab(c + u * rr * 0.55, (rr, rr * 0.85, rr), power=1.0)
            V, F = tube_path([c + u * rr * 0.55 + rot_x(0.4) @ np.array([rr * 1.0 * math.cos(a), 0, rr * 1.0 * math.sin(a)])
                              for a in np.linspace(0, 2 * math.pi, 24, endpoint=False)], rr * 0.18, 6, closed=True)
            acc.add(V, F, md)
        elif st == 'locs':
            V, F = hair_cap(thick, std)
            acc.add(V, F, mh)
            Lh = 0.10 + 0.3 * ln
            for k, ph in enumerate(np.linspace(0.6, 2 * math.pi - 0.6, 22)):
                hz, hx = math.cos(ph), math.sin(ph)
                uy = float(std(hz)) + 0.12
                u = nrm(np.array([hx * math.sqrt(1 - uy ** 2), uy, hz * math.sqrt(1 - uy ** 2)]))
                c = SK_C + u * skull_r(u[None])[0] * thick
                out = nrm(np.array([hx, 0, hz]))
                pts = [c + out * 0.015 * s + np.array([0, -Lh * s, 0]) for s in np.linspace(0, 1, 6)]
                V, F = tube_path(pts, 0.0075 + 0.003 * vol, 7)
                acc.add(V, F, mh if k % 3 else md)
        elif st in ('beanie', 'cap'):
            V, F = hair_cap(1.03, std)
            acc.add(V, F, mh)
            if self.headwear in ('none', ''):
                self.headwear = st
                self.headwear_c = self.hairc if not g(self.look, 'headwear_color') else self.headwear_c
        else:
            V, F = hair_cap(thick, std)
            acc.add(V, F, mh)
        # headwear
        hw = self.headwear
        mhw = material(self.headwear_c, rough=0.8)
        if hw == 'beanie':
            hl = hairline_fn([-1, -0.3, 0.1, 0.3, 1], [-0.28, -0.12, 0.12, 0.36, 0.36])
            V, F = hair_cap(1.13 + (0.05 if st in ('afro', 'curly_big') else 0), hl, tuck=1.0)
            acc.add(V, F, mhw)
            pts = []
            for ph in np.linspace(0, 2 * math.pi, 48, endpoint=False):
                hz, hx = math.cos(ph), math.sin(ph)
                uy = float(hl(hz)) + 0.05
                u = nrm(np.array([hx * math.sqrt(1 - uy ** 2), uy, hz * math.sqrt(1 - uy ** 2)]))
                pts.append(SK_C + u * skull_r(u[None])[0] * 1.15)
            V, F = tube_path(pts, 0.011, 8, closed=True)
            acc.add(V, F, material(shade(self.headwear_c, 0.85), rough=0.8))
        elif hw == 'cap':
            hl = hairline_fn([-1, -0.3, 0.1, 0.3, 1], [-0.15, 0.0, 0.18, 0.45, 0.45])
            V, F = hair_cap(1.10, hl, tuck=1.0)
            acc.add(V, F, mhw)
            u = nrm(np.array([0, 0.45, 1.0]))
            c = SK_C + u * skull_r(u[None])[0] * 1.10
            Vb, Fb = ellipsoid_mesh(c + np.array([0, -0.004, 0.045]), (0.085, 0.004, 0.06), rot_x(0.25), 20, 6)
            acc.add(Vb, Fb, material(shade(self.headwear_c, 0.9), rough=0.8))
        elif hw == 'headband':
            pts = []
            for ph in np.linspace(0, 2 * math.pi, 48, endpoint=False):
                hz, hx = math.cos(ph), math.sin(ph)
                uy = 0.5 if hz > 0.3 else 0.5 * max(0.0, hz / 0.3) + 0.15
                u = nrm(np.array([hx * math.sqrt(1 - uy ** 2), uy, hz * math.sqrt(1 - uy ** 2)]))
                pts.append(SK_C + u * skull_r(u[None])[0] * (thick + 0.008))
            V, F = tube_path(pts, 0.007, 8, closed=True)
            acc.add(V, F, mhw)
        self.hair = acc.build('hair%d' % self.p, parent=self.F, coll=self.coll)

    # ---------------------------------------------------------------- glasses
    def build_glasses(self):
        gl = g(self.look, 'glasses', default={}) or {}
        if not gl.get('present'):
            return
        fc = hex2rgb(gl.get('frame_color') or '#202020')
        shape = str(gl.get('shape') or 'rect').lower()
        th = min(1, max(0, fnum(gl.get('thickness', 0.4), 0.4)))
        rad = 0.0011 + 0.0022 * th
        mat = material(fc, rough=0.35)
        acc = MeshAcc()
        rest = self.rest
        zf = max(rest[EYE_L['ring']][:, 2].max(), rest[EYE_R['ring']][:, 2].max()) + 0.015
        inner_pts, outer_pts = [], []
        for E, sgn in ((EYE_L, 1), (EYE_R, -1)):
            ring = rest[E['ring']]
            cx, cy = ring[:, 0].mean(), ring[:, 1].mean() + 0.001
            hw, hh = 0.026, 0.019
            pts = []
            for a in np.linspace(0, 2 * math.pi, 40, endpoint=False):
                ca, sa = math.cos(a), math.sin(a)
                if shape == 'round':
                    x, y = 0.022 * ca, 0.02 * sa
                else:
                    pw = 0.25 if shape == 'rect' else 0.35
                    x = hw * np.sign(ca) * abs(ca) ** pw
                    y = hh * np.sign(sa) * abs(sa) ** pw
                    if shape == 'cat_eye' and sa > 0 and ca * sgn > 0:
                        y += 0.007 * (ca * sgn) ** 2
                pts.append((cx + x, cy + y, zf - 0.006 * (x * sgn / hw) ** 2))
            V, F = tube_path(pts, rad, 6, closed=True)
            acc.add(V, F, mat)
            P_ = np.array(pts)
            inner_pts.append(P_[np.argmin(P_[:, 0] * sgn)])
            outer_pts.append(P_[np.argmax(P_[:, 0] * sgn)])
            ear = np.array([sgn * (abs(rest[454 if sgn > 0 else 234][0]) + 0.006), cy + 0.004, rest[454][2] - 0.012])
            o = outer_pts[-1]
            V, F = tube_path([o, o + np.array([sgn * 0.004, 0, -0.012]), ear], rad, 6)
            acc.add(V, F, mat)
        a, b = inner_pts
        mid = (a + b) / 2 + np.array([0, 0.004, 0.003])
        V, F = tube_path([a, (a + mid) / 2 + np.array([0, 0.002, 0]), mid, (b + mid) / 2 + np.array([0, 0.002, 0]), b], rad, 6)
        acc.add(V, F, mat)
        acc.build('glasses%d' % self.p, parent=self.F, coll=self.coll)

    # ---------------------------------------------------------------- body
    def build_body(self):
        tt = self.top_type
        bf = self.build_f
        self.m_top = material(self.top_c, rough=0.75)
        self.m_top2 = material(self.top_c2, rough=0.75)
        self.m_pat = material(self.pat_c, rough=0.75)
        self.m_patmix = material(mix(self.top_c, self.pat_c, 0.5), rough=0.75)
        bname = str(g(self.look, 'build', default='average')).lower()
        self.depth_f = {'slim': 0.9, 'average': 1.0, 'broad': 1.05, 'heavy': 1.15}.get(bname, 1.0)
        self.width_f = {'slim': 0.95, 'average': 1.0, 'broad': 1.06, 'heavy': 1.1}.get(bname, 1.0)
        bot = g(self.look, 'bottom', default={}) or {}
        self.bottom_type = str(bot.get('type') or 'jeans').lower()
        self.bottom_c = hex2rgb(bot.get('color') or '#2c3446')
        self.m_bottom = material(self.bottom_c, rough=0.8)
        self.u_hem = {'tshirt': 0.95, 'vneck': 0.95, 'tank': 0.93, 'hoodie': 0.98, 'blazer': 0.98, 'jacket_over_top': 0.98,
                      'cardigan': 0.98, 'crew_sweater': 0.9, 'turtleneck': 0.9, 'button_shirt': 0.93}.get(tt, 0.95)
        if self.bottom_type == 'dress':
            self.u_hem = 0.66
        elif self.bottom_type == 'skirt':
            self.u_hem = min(self.u_hem, 0.74)   # tops are usually tucked into / end at a skirt's waistband
        self.u_waist = 0.6
        nseg = 48
        nr = len(PROF_ROWS)
        F = []
        ang = np.linspace(0, 2 * np.pi, nseg, endpoint=False)
        self.t_ang = ang
        for i in range(nr - 1):
            for k in range(nseg):
                k2 = (k + 1) % nseg
                F.append((i * nseg + k, i * nseg + k2, (i + 1) * nseg + k2, (i + 1) * nseg + k))
        top_c = nr * nseg
        bot_c = top_c + 1
        for k in range(nseg):
            F.append((top_c, (k + 1) % nseg, k))
            F.append((bot_c, (nr - 1) * nseg + k, (nr - 1) * nseg + (k + 1) % nseg))
        base_m = self.m_top2 if tt in ('blazer', 'jacket_over_top', 'cardigan') else self.m_skin
        self.torso = make_obj('torso%d' % self.p, np.zeros((nr * nseg + 2, 3)), F, [base_m], coll=self.coll, shadow=True)
        # garment: the same surface 3 mm out, from a per-type neckline curve down to the hem
        gs, gr = 112, 2 * (nr - 1) + 1
        self.g_ang = np.linspace(0, 2 * np.pi, gs, endpoint=False)
        self.g_rows = gr
        F, M = [], []
        for i in range(gr - 1):
            for k in range(gs):
                k2 = (k + 1) % gs
                F.append((i * gs + k, i * gs + k2, (i + 1) * gs + k2, (i + 1) * gs + k))
                y_nom = 0.03 - 0.5 * (i + 0.5) / (gr - 1)
                M.append(self.pattern_mat(y_nom, self.g_ang[k] + np.pi / gs))
        mats = [self.m_top, self.m_pat, self.m_patmix, self.m_top2]
        self.garment = make_obj('garment%d' % self.p, np.zeros((gr * gs, 3)), F, mats, M, coll=self.coll, shadow=True)
        # lower body: 8 rings (waist -> crotch, or waist -> skirt hem) + 2 legs of 3 segments
        F, M = [], []
        nl = 8
        for i in range(nl - 1):
            for k in range(nseg):
                k2 = (k + 1) % nseg
                F.append((i * nseg + k, i * nseg + k2, (i + 1) * nseg + k2, (i + 1) * nseg + k)); M.append(0)
        cb = nl * nseg
        for k in range(nseg):
            F.append((cb, (nl - 1) * nseg + k, (nl - 1) * nseg + (k + 1) % nseg)); M.append(0)
        self.lb_nring = nl
        bt = self.bottom_type
        if bt in ('skirt', 'dress'):
            seg_m, sph_m = [1, 1, 1, 1, 1, 1], [1, 1, 1, 1]
        elif bt == 'shorts':
            seg_m, sph_m = [0, 1, 1, 0, 1, 1], [1, 1, 0, 0]
        else:
            seg_m, sph_m = [0, 0, 0, 0, 0, 0], [0, 0, 0, 0]
        self.leg_t = Tubes(6, 4, seg_m, sph_m, n=16, sph=(16, 8))
        off = cb + 1
        F += [tuple(i + off for i in f) for f in self.leg_t.faces]
        M += self.leg_t.mats
        self.lb_nv = off + self.leg_t.nv
        self.lower = make_obj('lower%d' % self.p, np.zeros((self.lb_nv, 3)), F, [self.m_bottom, self.m_skin], M, coll=self.coll, shadow=True)
        # torso-attached details (neckline etc.) in torso-local frame
        self.torso_frame = empty('torsoframe%d' % self.p, coll=self.coll)
        self.build_top_details()
        # neck
        neck_m = self.m_top if tt == 'turtleneck' else self.m_skin
        self.neck_t = Tubes(1, 2, [0], [0, 0], n=16, sph=(16, 8))
        self.neck = make_obj('neck%d' % self.p, np.zeros((self.neck_t.nv, 3)), self.neck_t.faces, [neck_m], self.neck_t.mats, coll=self.coll, shadow=True)
        # arms: segs = [sh->mid, mid->elbow, elbow->wrist], spheres = [shoulder, elbow, wrist]
        long_sl = tt in ('crew_sweater', 'hoodie', 'button_shirt', 'blazer', 'jacket_over_top', 'turtleneck', 'cardigan')
        short_sl = tt in ('tshirt', 'vneck')
        seg_m = [0 if tt != 'tank' else 1, 0 if long_sl else 1, 0 if long_sl else 1]
        sph_m = [0 if tt != 'tank' else 1, 0 if long_sl else 1, 1]
        arm_mats = [self.m_top, self.m_skin]
        self.arm_t = Tubes(3, 3, seg_m, sph_m, n=14, sph=(14, 8))
        self.arms = [make_obj('arm%s%d' % (s, self.p), np.zeros((self.arm_t.nv, 3)), self.arm_t.faces, arm_mats, self.arm_t.mats, coll=self.coll, shadow=True)
                     for s in 'LR']
        # hands: palm prism + 15 finger segments + 20 joint spheres (+ wrist sphere)
        self.hand_t = Tubes(15, 20, n=8, sph=(8, 5))
        pf = []
        base = self.hand_t.nv
        # palm: 6 rim points top + 6 bottom + 2 centres
        for k in range(6):
            k2 = (k + 1) % 6
            pf.append((base + 12, base + k, base + k2))
            pf.append((base + 13, base + 6 + k2, base + 6 + k))
            pf.append((base + k, base + 6 + k, base + 6 + k2, base + k2))
        self.hand_faces = self.hand_t.faces + pf
        self.hands_ob = [make_obj('hand%s%d' % (s, self.p), np.zeros((base + 14, 3)), self.hand_faces, [self.m_skin], coll=self.coll, shadow=True)
                         for s in 'LR']
        if bool(g(self.look, 'necklace', default=False)):
            pts = [(0.075 * math.sin(a), 0.035 - 0.07 * (math.cos(a) * 0.5 + 0.5) ** 1.5 * 0 - 0.06 * max(0, math.cos(a)) ** 2, 0.06 * math.cos(a) + 0.005)
                   for a in np.linspace(0, 2 * math.pi, 40, endpoint=False)]
            V, F = tube_path(pts, 0.0028, 6, closed=True)
            ob = make_obj('necklace%d' % self.p, V, F, [material((0.86, 0.72, 0.36), rough=0.25, metal=0.6)], parent=self.torso_frame, coll=self.coll)

    def pattern_mat(self, y, a):
        """material slot for a garment face: 0 base, 1 pattern colour, 2 base/pattern mix, 3 secondary."""
        pat = self.pattern
        col = int(math.floor(a / (2 * np.pi) * 112))
        row = int(math.floor(-y / 0.0333 + 0.5))
        if pat == 'stripes':
            return 1 if int(math.floor(-y / 0.035)) % 2 == 0 else 0
        if pat == 'plaid':
            hb = int(math.floor(-y / 0.03)) % 4 == 0
            vb = (col // 2) % 4 == 0
            return 2 if (hb and vb) else (1 if (hb or vb) else 0)
        if pat == 'tie_dye':
            v = math.sin(a * 3 + y * 25) + math.sin(a * 5 - y * 40)
            return 1 if v > 0.6 else (2 if v < -0.8 else 0)
        if pat == 'floral':
            h = math.sin(row * 12.9898 + (col // 3) * 78.233) * 43758.5453
            return 1 if (h - math.floor(h)) < 0.13 else 0
        if pat == 'graphic':
            front = math.sin(a) > 0.8   # front = +z (angle ~ pi/2)
            return 1 if (front and -0.24 < y < -0.10) else 0
        return 0

    def profile(self, S, Hh, Lsh):
        """torso rings (torso-local, y down from the shoulder line): ys, half-width, half-depth, z-centre."""
        P = PROF_ROWS
        ys = np.array([r[1] if r[0] == 'a' else -r[1] * Lsh for r in P])
        u = np.array([0.0 if r[0] == 'a' else r[1] for r in P])
        wfac = np.where(u >= 0.2, self.width_f, 1.0)
        hw = (np.array([r[2] for r in P]) * S + np.array([r[3] for r in P]) * Hh + np.array([r[4] for r in P])) * wfac
        hd = np.array([r[5] for r in P]) * np.where(u >= 0.05, self.depth_f, 1.0 + (self.depth_f - 1) * 0.4)
        zc = np.array([r[6] for r in P])
        return ys, hw, hd, zc

    def y_of_u(self, u, Lsh):
        return -u * Lsh

    def front_z(self, x, y, S=0.18, off=0.0):
        """torso-local front surface z at (x, y) (static approximation for the attached details)."""
        ys, hw, hd, zc = self.profile(S, 0.18, 0.46)
        h_ = np.interp(-np.asarray(y), -ys, hw)
        d_ = np.interp(-np.asarray(y), -ys, hd)
        z_ = np.interp(-np.asarray(y), -ys, zc)
        q = np.clip(np.abs(x) / np.maximum(h_, 1e-4), 0, 1)
        return z_ + d_ * np.maximum(0, 1 - q ** 2.6) ** (1 / 2.6) + off

    def surf_patch(self, outline_fn, nu=8, nv=10, off=0.003):
        """grid patch on the torso front. outline_fn(s) -> (y, x_left, x_right) for s in 0..1 (top -> bottom)."""
        V, F = [], []
        for i in range(nv):
            y, xl, xr = outline_fn(i / (nv - 1))
            for j in range(nu):
                x = xl + (xr - xl) * j / (nu - 1)
                V.append((x, y, float(self.front_z(x, y, off=off))))
        for i in range(nv - 1):
            for j in range(nu - 1):
                a = i * nu + j
                F.append((a, a + 1, a + nu + 1, a + nu))
        return np.array(V), F

    def neck_line(self, x, sa):
        """garment top edge (torso-local y) for local x and sin(angle) (front > 0)."""
        tt = self.top_type
        ax = np.abs(x)
        fr = np.clip(sa * 2.5, 0, 1)
        top = PROF_ROWS[0][1] - 0.002
        if tt == 'vneck':
            return top - fr * 0.17 * np.clip(1 - ax / 0.075, 0, 1) ** 1.2
        if tt == 'button_shirt':
            return top - fr * 0.085 * np.clip(1 - ax / 0.045, 0, 1)
        if tt in ('blazer', 'jacket_over_top'):
            return top - fr * 0.40 * np.clip(1 - ax / 0.10, 0, 1)
        if tt == 'cardigan':
            return top - fr * 0.30 * np.clip(1 - ax / 0.085, 0, 1)
        if tt == 'tank':
            return top - 0.004 - fr * 0.03 * np.exp(-(x / 0.05) ** 2) - 0.13 * smoothstep(0.075, 0.12, ax)
        return top - fr * 0.022 * np.exp(-(x / 0.05) ** 2)     # crew-type necklines

    def build_top_details(self):
        tt = self.top_type
        acc = MeshAcc()
        y_top = PROF_ROWS[0][1] - 0.002
        zc0 = PROF_ROWS[0][6]
        if tt == 'button_shirt':
            pts = [(0.062 * math.sin(a), y_top + 0.004 - 0.02 * max(0.0, math.cos(a)) ** 3, zc0 + 0.052 * math.cos(a))
                   for a in np.linspace(0, 2 * math.pi, 36, endpoint=False)]
            V, F = tube_path(pts, 0.008, 6, closed=True)
            acc.add(V, F, self.m_top2)
            for sgn in (1, -1):  # collar points
                pts = [(sgn * 0.008, y_top - 0.06), (sgn * 0.052, y_top - 0.004), (sgn * 0.06, y_top - 0.045)]
                V = np.array([(x, y, float(self.front_z(x, y, off=0.007))) for x, y in pts])
                acc.add(V, [(0, 1, 2) if sgn < 0 else (2, 1, 0)], self.m_top2)
            pts = [(0.0, y, float(self.front_z(0.0, y, off=0.004))) for y in np.linspace(y_top - 0.085, -0.55, 12)]
            V, F = tube_path(pts, 0.005, 6)
            acc.add(V, F, material(shade(self.top_c, 0.88)))
            for y in np.arange(y_top - 0.11, -0.5, -0.075):
                Vb, Fb = sphere_unit(8, 4)
                acc.add(Vb * np.array([0.0045, 0.0045, 0.003]) + np.array([0, y, float(self.front_z(0, y, off=0.009))]), Fb, material((0.96, 0.96, 0.95)))
        if tt in ('blazer', 'jacket_over_top'):
            lap = material(shade(self.top_c, 0.86))
            for sgn in (1, -1):
                pts = [(sgn * 0.085, y_top - 0.01), (sgn * 0.125, y_top - 0.09), (sgn * 0.01, y_top - 0.38), (sgn * 0.05, y_top - 0.15)]
                V = np.array([(x, y, float(self.front_z(x, y, off=0.008))) for x, y in pts])
                acc.add(V, [(0, 1, 2, 3) if sgn > 0 else (3, 2, 1, 0)], lap)
        if tt == 'hoodie':
            V, F = tube_path([(0.115 * math.sin(a), y_top + 0.005 + 0.02 * math.cos(a) ** 2, zc0 - 0.01 + 0.07 * math.cos(a))
                              for a in np.linspace(-2.5, 2.5, 22)], 0.03, 10)
            acc.add(V, F, self.m_top)
            for sgn in (1, -1):
                pts = [(sgn * 0.03, y, float(self.front_z(sgn * 0.03, y, off=0.006))) for y in np.linspace(y_top - 0.03, y_top - 0.17, 6)]
                V, F = tube_path(pts, 0.003, 5)
                acc.add(V, F, material((0.93, 0.93, 0.9)))
        if tt == 'turtleneck':
            pts = [(0.056 * math.sin(a), y_top + 0.002, zc0 + 0.05 * math.cos(a)) for a in np.linspace(0, 2 * math.pi, 32, endpoint=False)]
            V, F = tube_path(pts, 0.012, 8, closed=True)
            acc.add(V, F, self.m_top)
        self.top_details = acc.build('topdetail%d' % self.p, parent=self.torso_frame, coll=self.coll)

    # ---------------------------------------------------------------- per-frame pose
    def pose(self, t, panel):
        R = self.R
        HM = self.HM[t]
        self.head.matrix_world = Matrix(HM.tolist())
        self.F.matrix_basis = Matrix(self.A[t].tolist())
        canon = self.canon[t].copy()
        if not self.face_mode:
            canon = self.param_face(t, canon)
        else:
            canon = self.lids(t, canon)
        self._canon = canon
        HV = self.head_verts(t, canon)
        set_verts(self.head_obj, HV)
        if self.beard is not None:
            set_verts(self.beard, self.beard_verts(HV))
        # brows, mouth (use the possibly modified canon)
        old = self.canon[t]
        self.canon[t] = canon
        set_verts(self.brow_obj, self.brow_verts(t))
        set_verts(self.lash_obj, self.lash_verts(canon)[0])
        set_verts(self.mouth_obj, self.mouth_verts(t))
        for e in self.eyes:
            yaw, pitch = self.eye_gaze(t, e)
            Rm = rot_y(float(yaw)) @ rot_x(-float(pitch))
            M = np.eye(4); M[:3, :3] = Rm; M[:3, 3] = e['c']
            e['ob'].matrix_basis = Matrix(M.tolist())
        self.canon[t] = old
        self.pose_body(t, panel)

    def beard_verts(self, HV):
        V = HV
        F = self.head_faces
        idx = self.beard_idx
        n = np.zeros_like(V)
        tri = np.array([f[:3] for f in F] + [(f[0], f[2], f[3]) for f in F if len(f) == 4])
        fn = np.cross(V[tri[:, 1]] - V[tri[:, 0]], V[tri[:, 2]] - V[tri[:, 0]])
        for k in range(3):
            np.add.at(n, tri[:, k], fn)
        n = nrm(n)
        return V[idx] + n[idx] * (self.beard_off * self.beard_w + 0.0008)[:, None]

    def lids(self, t, X):
        """eyelids: shut only when the calibrated eye_open channel < 0.2; above that keep the tracked lid
        position but never let the aperture collapse (looking down is not a blink)."""
        for E, side in ((EYE_L, 'L'), (EYE_R, 'R')):
            eo = chan(self.R, self.p, t, 'eye_open_' + side, 1.0)
            up, lo = X[E['up']].copy(), X[E['lo']].copy()
            rest_gap = np.maximum(self.rest[E['up']][:, 1] - self.rest[E['lo']][:, 1], 0.002)
            gap = up[:, 1] - lo[:, 1]
            min_gap = 0.3 * rest_gap
            up[:, 1] = np.where(gap < min_gap, lo[:, 1] + min_gap, up[:, 1])
            if eo < 0.2:
                f = float(np.clip((eo - 0.08) / 0.12, 0, 1))
                closed = lo + 0.3 * (up - lo)
                closed[:, 2] = np.maximum(up[:, 2], lo[:, 2])
                up = closed + (up - closed) * f
                lo = closed + (lo - closed) * f
            X[E['up']] = up
            X[E['lo']] = lo
        return X

    def param_face(self, t, X):
        """Sonnet-only rigs: deform the template with the channels (mouth, blink)."""
        p, R = self.p, self.R
        m = chan(R, p, t, 'mouth_open', 0.0)
        mw = chan(R, p, t, 'mouth_width', 0.054)
        sm = chan(R, p, t, 'smile', 0.0)
        fr = chan(R, p, t, 'frown', 0.0)
        pk = chan(R, p, t, 'pucker', 0.0)
        m = float(np.clip(m, 0, 0.04))
        mw = float(np.clip(mw, 0.035, 0.075)) * (1 - 0.3 * pk)
        T0 = TEMPLATE
        y_m = (T0[13, 1] + T0[14, 1]) / 2
        g0 = T0[13, 1] - T0[14, 1]
        w0 = T0[291, 0] - T0[61, 0]
        x, y = X[:, 0], X[:, 1]
        # jaw: lower face moves down by (m - g0)
        lat = np.clip(1 - (np.abs(x) - 0.03) / 0.06, 0, 1)
        wj = np.clip((y_m - y) / 0.004, 0, 1) * lat
        wj[list(LOWER_LIP_IDX)] = 1.0
        wj[OVAL[:1]] = 0
        X[:, 1] -= (m - g0) * 0.8 * wj
        # lower lip contour gets the full opening, upper lip a bit up
        X[list(LOWER_LIP_IDX), 1] -= (m - g0) * 0.2
        upper = [i for i in LIPS_IN + LIPS_OUT if i not in LOWER_LIP_IDX and i not in (61, 291, 78, 308)]
        X[upper, 1] += 0.0
        # width / smile / frown / pucker on the mouth region
        reg = np.exp(-(((x) / 0.045) ** 2 + ((y - y_m) / 0.022) ** 2))
        X[:, 0] = x * (1 + (mw / w0 - 1) * reg)
        corner = np.clip(np.abs(X[:, 0]) / (mw / 2), 0, 1) ** 2 * reg
        X[:, 1] += (0.007 * sm - 0.006 * fr) * corner
        X[:, 2] += 0.006 * pk * reg
        # blink / eye openness
        for E, side in ((EYE_L, 'L'), (EYE_R, 'R')):
            eo = chan(R, p, t, 'eye_open_' + side, 1.0)
            f = float(np.clip((eo - 0.1) / 0.1, 0, 1)) * 0.3 if eo < 0.2 else float(np.clip(max(0.3, eo), 0, 1.4))
            up, lo = X[E['up']].copy(), X[E['lo']].copy()
            closed = lo + 0.3 * (up - lo)
            X[E['up']] = closed + (up - closed) * f
            X[E['lo']] = closed + (lo - closed) * min(f, 1.0)
        return X

    # ---------------------------------------------------------------- body pose
    def to_world(self, M, P):
        return np.asarray(P) @ M[:3, :3].T + M[:3, 3]

    def pose_body(self, t, panel):
        R, p = self.R, self.p
        HM = self.HM[t]
        HA = HM @ self.A[t]
        J = np.nan_to_num(R['joints'][p, t])
        vis = R['joint_vis'][p, t]
        bf = self.build_f
        s = self.scale * self.head_scale
        head_o = HM[:3, 3]
        # ---- derived shoulders from the head (rigs without body tracking)
        f = HM[:3, 2]
        yaw = math.atan2(f[0], f[2])
        nl = min(1, max(0, fnum(g(self.look, 'neck_length', default=0.5), 0.5)))
        mid_d = head_o + np.array([0, -0.19 - 0.04 * (nl - 0.5), -0.05])
        xd = np.array([math.cos(0.35 * yaw), 0, -math.sin(0.35 * yaw)])
        Ld, Rd = mid_d + xd * 0.19 * bf, mid_d - xd * 0.19 * bf
        Ls, Rs = J[0], J[1]
        tracked = (np.abs(J[:2]).sum() > 1e-4)
        sw = np.linalg.norm(Ls - Rs) if tracked else 0
        mid_t = (Ls + Rs) / 2
        below = head_o[1] - mid_t[1] if tracked else 0
        sane = tracked and 0.2 < sw < 0.7 and 0.08 < below < 0.45 and abs(mid_t[2] - head_o[2]) < 0.35
        w = float(smoothstep(0.25, 0.6, min(vis[0], vis[1]))) if sane else 0.0
        self._wtrack = w if not hasattr(self, '_wtrack') else 0.7 * self._wtrack + 0.3 * w
        w = self._wtrack
        Ls = w * Ls + (1 - w) * Ld if tracked else Ld
        Rs = w * Rs + (1 - w) * Rd if tracked else Rd
        mid = (Ls + Rs) / 2
        X = nrm(Ls - Rs)
        S = float(np.clip(np.linalg.norm(Ls - Rs) / 2, 0.13, 0.25))
        # ---- hips: fit / taper the torso to them when tracked
        hips = J[6:8]
        hip_ok = tracked and w > 0.5 and min(vis[6], vis[7]) > 0.4 and np.abs(hips).sum() > 1e-4
        hmid = (hips[0] + hips[1]) / 2
        if hip_ok:
            Lsh_t = float(np.linalg.norm(mid - hmid))
            hip_ok = 0.25 < Lsh_t < 0.75
        if hip_ok:
            up = nrm(mid - hmid)
            Lsh = Lsh_t
            Hh = float(np.clip(np.linalg.norm(hips[0] - hips[1]) / 2 + 0.075, 0.12, 0.27))
        else:
            up = np.array([0, 1.0, 0])
            Lsh = 0.47
            Hh = 0.95 * S + 0.015
        # smooth the body measures over time (avoid popping when hips flicker in and out)
        if hasattr(self, '_bm'):
            self._bm = 0.8 * self._bm + 0.2 * np.array([Lsh, Hh])
        else:
            self._bm = np.array([Lsh, Hh])
        Lsh, Hh = float(self._bm[0]), float(self._bm[1])
        Y = nrm(up - np.dot(up, X) * X)
        Z = np.cross(X, Y)
        O = mid - Z * 0.03   # torso centre slightly behind the shoulder joints
        Mt = np.eye(4); Mt[:3, 0], Mt[:3, 1], Mt[:3, 2], Mt[:3, 3] = X, Y, Z, O
        Md = Mt.copy(); Md[:3, 0] *= S / 0.18
        self.torso_frame.matrix_world = Matrix(Md.tolist())
        ys, hw, hd, zc = self.profile(S, Hh, Lsh)
        ca, sa = np.cos(self.t_ang), np.sin(self.t_ang)
        ex = 2 / 2.6
        px = hw[:, None] * np.sign(ca) * np.abs(ca) ** ex
        pz = hd[:, None] * np.sign(sa) * np.abs(sa) ** ex + zc[:, None]
        loc = np.stack([px, np.repeat(ys[:, None], len(ca), 1), pz], -1).reshape(-1, 3)
        loc = np.vstack([loc, [0, ys[0] + 0.004, zc[0]], [0, ys[-1] - 0.01, 0]])
        set_verts(self.torso, loc @ Mt[:3, :3].T + O)
        self.tinfo = (Mt, ys, hw, hd, zc)
        # garment: neckline -> hem
        ga = self.g_ang
        gc, gsn = np.cos(ga), np.sin(ga)
        x_nom = (S + 0.03) * np.sign(gc) * np.abs(gc) ** ex
        y0 = np.minimum(self.neck_line(x_nom, gsn), ys[0] - 0.002)
        y_hem = -self.u_hem * Lsh
        ysub = np.interp(np.linspace(0, len(ys) - 1, self.g_rows), np.arange(len(ys)), ys)
        Yg = np.clip(ysub[:, None], y_hem, y0[None])
        hwg = np.interp(-Yg, -ys, hw) + 0.003
        hdg = np.interp(-Yg, -ys, hd) + 0.003
        zcg = np.interp(-Yg, -ys, zc)
        gx = hwg * np.sign(gc)[None] * np.abs(gc)[None] ** ex
        gz = hdg * np.sign(gsn)[None] * np.abs(gsn)[None] ** ex + zcg
        gl = np.stack([gx, Yg, gz], -1).reshape(-1, 3)
        set_verts(self.garment, gl @ Mt[:3, :3].T + O)
        # lower body
        set_verts(self.lower, self.lower_verts(Mt, ys, hw, hd, zc, Lsh, Hh, ca, sa, ex))
        # neck: head base -> torso neck base
        hb = self.to_world(HA, [[0, -0.055, -0.07]])[0]
        nb = O + Y * (ys[0] - 0.005) + Z * (zc[0] + 0.01)
        rn = 0.047 * (1.0 if bf < 1.15 else 1.15) * s
        set_verts(self.neck, self.neck_t.verts(hb, nb, rn, rn * 1.12, [hb, nb], [rn, rn * 1.12]))
        # arms & hands
        for h in range(2):
            self.pose_arm(t, h, Ls if h == 0 else Rs, X, Y, Z, w, panel)

    def lower_verts(self, Mt, ys, hw, hd, zc, Lsh, Hh, ca, sa, ex):
        """trousers / skirt rings from the waist (under the top's hem) down, plus two legs."""
        bt = self.bottom_type
        yw, yh, yhip = -self.u_waist * Lsh, -self.u_hem * Lsh, -Lsh
        at = lambda y, arr: np.interp(-y, -ys, arr)
        hd_hip = at(yhip, hd)
        skirt = bt in ('skirt', 'dress')
        if skirt:
            L = 0.30 + 0.25 * (bt == 'dress') if False else 0.45
            Ys = [yw, yh + 0.004, yh - 0.004, min(yhip, yh - 0.01) - 0.02, yhip - 0.12, yhip - 0.24, yhip - 0.36, yhip - L]
            fl = [0, 0, 1.0, 1.02, 1.06, 1.12, 1.18, 1.24]
        else:
            Ys = [yw, yh + 0.004, yh - 0.004, min(yhip, yh - 0.01) - 0.01, yhip - 0.05, yhip - 0.09, yhip - 0.12, yhip - 0.14]
            fl = [0, 0, 1.0, 1.0, 0.99, 0.96, 0.88, 0.6]
        rows_w, rows_d, rows_z = [], [], []
        for i, y in enumerate(Ys):
            if i < 2:   # under the top
                rows_w.append(at(y, hw) - 0.006); rows_d.append(at(y, hd) - 0.006); rows_z.append(at(y, zc))
            else:
                base_w = at(y, hw) if y > ys[-1] else Hh * self.width_f
                wv = max(base_w, Hh * self.width_f * 0.98) * fl[i] + 0.006
                dv = (at(y, hd) if y > ys[-1] else hd_hip) * (fl[i] if not skirt else (1 + 0.6 * (fl[i] - 1))) + 0.006
                if not skirt and i >= 6:
                    dv = hd_hip * fl[i]
                rows_w.append(wv); rows_d.append(dv); rows_z.append(0.0)
        rows_w, rows_d, rows_z, Ys = map(np.array, (rows_w, rows_d, rows_z, Ys))
        px = rows_w[:, None] * np.sign(ca) * np.abs(ca) ** ex
        pz = rows_d[:, None] * np.sign(sa) * np.abs(sa) ** ex + rows_z[:, None]
        loc = np.stack([px, np.repeat(Ys[:, None], len(ca), 1), pz], -1).reshape(-1, 3)
        loc = np.vstack([loc, [0, Ys[-1] - 0.005, 0]])
        # legs
        wf = self.width_f
        lx = Hh * 0.48
        if skirt:
            ytop = Ys[-1] + 0.05
            rt = 0.055 * wf
        else:
            ytop = yhip - 0.06
            rt = 0.085 * wf * (0.88 if bt == 'leggings' else 1.0)
        yk, ya = yhip - 0.50, yhip - 0.92
        ymid = yhip - 0.27 if bt == 'shorts' else (ytop + yk) / 2
        p0, p1, r0, r1, sc_, sr = [], [], [], [], [], []
        for sg in (1, -1):
            a = np.array([sg * lx, ytop, 0.0]); m = np.array([sg * lx * 0.96, ymid, 0.0])
            k = np.array([sg * lx * 0.9, yk, 0.01]); an = np.array([sg * lx * 0.85, ya, -0.01])
            rm = rt * 0.9 if not skirt else rt * 0.95
            p0 += [a, m, k]; p1 += [m, k, an]
            r0 += [rt, rm, rt * 0.68]; r1 += [rm, rt * 0.7, rt * 0.5]
        sc_ = [p1[1], p1[4], p0[0], p0[3]]
        sr = [r1[1], r1[4], r0[0], r0[3]]
        legs = self.leg_t.verts(p0, p1, r0, r1, sc_, sr)
        allv = np.vstack([loc, legs])
        return allv @ Mt[:3, :3].T + Mt[:3, 3]

    def front_alpha(self, P, margin=0.03):
        """per point scale factor (<=1) along the camera ray that puts points overlapping the torso in image space
        in front of its front surface (+margin). Smooth falloff at the torso silhouette edge."""
        Mt, ys, hw, hd, zc = self.tinfo
        Rm, O = Mt[:3, :3], Mt[:3, 3]
        P = np.atleast_2d(P)
        Pl = P @ Rm
        Ol = O @ Rm
        den = np.where(np.abs(Pl[:, 2]) < 1e-6, -1e-6, Pl[:, 2])
        zf = float(np.max(hd))
        sf = (zf + Ol[2]) / den
        Lf = sf[:, None] * Pl - Ol
        yv = Lf[:, 1]
        h_ = np.interp(-yv, -ys, hw)
        q = np.abs(Lf[:, 0]) / np.maximum(h_, 1e-4)
        inside = smoothstep(1.02, 0.82, q) * smoothstep(ys[-1] - 0.02, ys[-1] + 0.05, yv) * smoothstep(ys[0] + 0.02, ys[0] - 0.03, yv)
        front = np.interp(-yv, -ys, zc) + np.interp(-yv, -ys, hd) * np.maximum(0, 1 - np.minimum(q, 1) ** 2.6) ** (1 / 2.6) + margin
        st = (front + Ol[2]) / den
        st = np.clip(st, 0.55, 1.0)
        return 1 - inside * (1 - st)

    def in_panel(self, P, panel, margin=0.0):
        W, H, f = self.ctx['W'], self.ctx['H'], self.ctx['fpx']
        if P[2] > -0.01:
            return False
        u = (W / 2 + f * P[0] / -P[2]) / W
        v = (H / 2 - f * P[1] / -P[2]) / H
        return (panel[0] - margin) <= u <= (panel[1] + margin) and -margin <= v <= 1 + margin

    def pose_arm(self, t, h, sh, X, Y, Z, wtrack, panel):
        R, p = self.R, self.p
        J = np.nan_to_num(R['joints'][p, t])
        vis = R['joint_vis'][p, t]
        bf = self.build_f
        sgn = 1 if h == 0 else -1
        el, wr = J[2 + h].copy(), J[4 + h].copy()
        ev, wv = vis[2 + h], vis[4 + h]
        hv = bool(R['hand_valid'][p, t, h])
        HP = np.nan_to_num(R['hands'][p, t, h])
        if hv and np.abs(HP).sum() < 1e-6:
            hv = False
        # defaults: arm hanging down at the side
        el_d = sh + X * sgn * 0.04 - Y * 0.27 - Z * 0.01
        wr_d = el_d + X * sgn * 0.01 - Y * 0.25 + Z * 0.06
        trackable = wtrack > 0.5 and np.abs(el).sum() > 1e-4
        el_ok = trackable and (ev >= 0.4 or not self.in_panel(el, panel, 0.0))
        if not el_ok:
            el = el_d
        wr_ok = el_ok and np.abs(wr).sum() > 1e-4 and (wv >= 0.4 or not self.in_panel(wr, panel, 0.0))
        if hv:
            wr = HP[0].copy()
        elif not wr_ok:
            wr = el + (wr_d - el_d)
        # anything overlapping the torso in the image goes in front of it (same image position, along the ray)
        el = el * self.front_alpha(el)[0]
        if hv:
            HP = HP * float(np.min(self.front_alpha(HP)))
            wr = HP[0].copy()
        else:
            wr = wr * self.front_alpha(wr)[0]
            HP = self.relaxed_hand(h, el, wr, X, Z)
            HP = HP * float(np.min(self.front_alpha(HP, margin=0.02)))
        ru = 0.046 * bf
        rf = 0.037 * bf
        rw = 0.029 * bf
        sh = sh - Y * 0.022
        mid = sh + (el - sh) * 0.45
        arm = self.arms[h]
        arm.data.vertices.foreach_set('co', self.arm_t.verts(
            [sh, mid, el], [mid, el, wr], [ru * 0.92, ru * 0.97, rf], [ru * 0.97, rf, rw],
            [sh, el, wr], [ru * 0.92, rf, rw]).astype(np.float32).ravel())
        arm.data.update()
        hob = self.hands_ob[h]
        hob.hide_render = False
        ps = float(np.clip(np.linalg.norm(HP[5] - HP[17]), 0.05, 0.11))
        segs0, segs1, r0, r1 = [], [], [], []
        chains = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
        spr = []
        for ci, ch in enumerate(chains):
            rb = (0.15 if ci == 0 else 0.12 if ci < 4 else 0.105) * ps
            for k in range(3):
                segs0.append(HP[ch[k]]); segs1.append(HP[ch[k + 1]])
                r0.append(rb * (1 - 0.1 * k)); r1.append(rb * (1 - 0.1 * (k + 1)))
            spr += [rb, rb * 0.9, rb * 0.8, rb * 0.72]
        V1 = self.hand_t.verts(segs0, segs1, r0, r1, [HP[i] for i in range(1, 21)], spr)
        rim = HP[[0, 1, 5, 9, 13, 17]]
        cpt = rim.mean(0)
        n = nrm(np.cross(HP[5] - HP[0], HP[17] - HP[0]))
        th = 0.16 * ps
        rim = cpt + (rim - cpt) * 1.06
        V2 = np.vstack([rim + n * th, rim - n * th, cpt + n * th * 1.1, cpt - n * th * 1.1])
        set_verts(hob, np.vstack([V1, V2]))

    def relaxed_hand(self, h, el, wr, X, Z):
        """21 landmarks of a small relaxed hand continuing the forearm (used when the hand is not detected)."""
        a = nrm(wr - el)
        c = -X * (1 if h == 0 else -1)                # palm faces the body midline
        c = nrm(c - np.dot(c, a) * a)
        t_ = np.cross(c, a) if h == 0 else np.cross(a, c)   # thumb side (forward for hanging arms)
        hs = 0.92 * self.build_f ** 0.5
        T = RELAXED_HAND * hs
        return wr + T[:, 0:1] * t_ + T[:, 1:2] * a + T[:, 2:3] * c


# ----------------------------------------------------------------------------------------------------------------------
# scene-level setup
# ----------------------------------------------------------------------------------------------------------------------
def write_sl(path, lighting, gain=1.0):
    side = str(lighting.get('key_side', 'front') or 'front').lower()
    ks = min(1, max(0, fnum(lighting.get('key_strength', 0.6), 0.6)))
    amb = min(1, max(0, fnum(lighting.get('ambient', 0.5), 0.5)))
    warm = min(1, max(-1, fnum(lighting.get('warmth', 0.0), 0.0)))
    con = min(1, max(0, fnum(lighting.get('contrast', 0.5), 0.5)))
    kv = {'left': (-0.55, 0.42, 0.72), 'right': (0.55, 0.42, 0.72), 'top': (0.0, 0.85, 0.52)}.get(side, (-0.18, 0.38, 0.91))
    kv = tuple(nrm(np.array(kv)))
    fv = tuple(nrm(np.array((-kv[0] * 1.0 + (0.5 if side in ('front', 'top') else 0), 0.05, 0.8))))
    kc = (0.45 + 0.4 * ks)
    fc = 0.10 + 0.28 * (1 - con)
    ac = 0.18 + 0.30 * amb
    tint_k = np.array([1 + 0.10 * warm, 1.0, 1 - 0.10 * warm])
    tint_f = np.array([1 - 0.05 * warm, 1.0, 1 + 0.05 * warm])
    lights = [(1, 0.35, kc * tint_k, kv), (1, 0.6, fc * tint_f, fv), (1, 0.4, np.array([0.14, 0.14, 0.15]), (0.0, 0.75, -0.66)),
              (0, 0.0, np.zeros(3), (0, 1.0, 0))]
    txt = 'version 1\n'
    for a, v in zip('xyz', ac * np.ones(3) * gain):
        txt += 'light_ambient.%s %f\n' % (a, v)
    for i, (on, sm, c, v) in enumerate(lights):
        txt += 'light[%d].flag %d\nlight[%d].smooth %f\n' % (i, on, i, sm)
        for a, cc in zip('xyz', np.asarray(c) * gain):
            txt += 'light[%d].col.%s %f\n' % (i, a, cc)
        for a, cc in zip('xyz', np.asarray(c) * 0.15 * gain):
            txt += 'light[%d].spec.%s %f\n' % (i, a, cc)
        for a, vv in zip('xyz', v):
            txt += 'light[%d].vec.%s %f\n' % (i, a, vv)
    with open(path, 'w') as fh:
        fh.write(txt)
    return kv, con


def render_to(path):
    sc = bpy.context.scene
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)


def read_png(path):
    img = bpy.data.images.load(path)
    w, h = img.size
    px = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(px)
    bpy.data.images.remove(img)
    return px.reshape(h, w, 4)


def setup_render(R, tmp, aa):
    sc = bpy.context.scene
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    sc.render.engine = 'BLENDER_WORKBENCH'
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = R['W'], R['H'], 100
    sc.render.pixel_aspect_x = sc.render.pixel_aspect_y = 1
    sc.render.film_transparent = False
    sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_mode = 'RGB'
    sc.render.image_settings.compression = 10
    sc.render.use_compositing = False
    sc.render.use_sequencer = False
    sc.view_settings.view_transform = 'Standard'
    sc.view_settings.look = 'None'
    sc.view_settings.exposure = 0.0
    sc.view_settings.gamma = 1.0
    sc.display_settings.display_device = 'sRGB'
    sc.display.render_aa = aa
    cam = bpy.data.cameras.new('cam')
    cam.sensor_fit = 'VERTICAL'
    cam.angle_y = math.radians(R['vfov'])
    if R.get('crop'):
        cx0, cy0, cS = R['crop']; N = R['out_size']
        sc.render.resolution_x = sc.render.resolution_y = N
        fpx = (R['H'] / 2) / math.tan(math.radians(R['vfov']) / 2)
        cam.angle_y = 2 * math.atan((cS / 2) / fpx)
        cam.shift_x = (cx0 + cS / 2 - R['W'] / 2) / cS
        cam.shift_y = -(cy0 + cS / 2 - R['H'] / 2) / cS
    cam.clip_start, cam.clip_end = 0.03, 100.0
    co = bpy.data.objects.new('cam', cam)
    sc.collection.objects.link(co)
    co.matrix_world = Matrix.Identity(4)
    sc.camera = co
    world = bpy.data.worlds.new('w') if not sc.world else sc.world
    sc.world = world
    world.color = (0.2, 0.2, 0.2)
    return sc


def setup_shading(sc, lighting, tmp):
    sh = sc.display.shading
    sh.light = 'STUDIO'
    sh.color_type = 'TEXTURE'
    sh.show_specular_highlight = False
    sh.use_world_space_lighting = False
    # calibration: a camera-facing plane with albedo 0.5 -> measured gain of the studio light for the backdrop normal
    me = bpy.data.meshes.new('calib')
    me.from_pydata([(-50, -50, -5), (50, -50, -5), (50, 50, -5), (-50, 50, -5)], [], [(0, 1, 2, 3)])
    m = bpy.data.materials.new('calib'); m.diffuse_color = (0.5, 0.5, 0.5, 1); m.roughness = 1.0
    me.materials.append(m)
    ob = bpy.data.objects.new('calib', me)
    sc.collection.objects.link(ob)
    hidden = [o for o in sc.objects if o.type == 'MESH' and o is not ob and not o.hide_render]
    for o in hidden:
        o.hide_render = True
    rx, ry = sc.render.resolution_x, sc.render.resolution_y
    sc.render.resolution_x = sc.render.resolution_y = 32
    sh.show_shadows = sh.show_cavity = sh.show_object_outline = False
    k = None
    gain = 1.0
    for it in range(2):
        path = os.path.join(tmp, 'studio_%d.sl' % it)
        kv, con = write_sl(path, lighting, gain)
        sl = bpy.context.preferences.studio_lights.load(path, 'STUDIO')
        sh.studio_light = sl.name
        render_to(os.path.join(tmp, 'calib.png'))
        px = read_png(os.path.join(tmp, 'calib.png'))[12:20, 12:20, :3].reshape(-1, 3).mean(0)
        k = srgb2lin(px) / 0.5
        if it == 0:
            lum = float(k @ np.array([0.2126, 0.7152, 0.0722]))
            gain = 1.04 / max(lum, 1e-3)
    sc.render.resolution_x, sc.render.resolution_y = rx, ry
    bpy.data.objects.remove(ob, do_unlink=True)
    for o in hidden:
        o.hide_render = False
    sh.show_shadows = 'shadow' in os.environ.get('RB_ON', '')   # stencil shadows streak on open meshes; off by default
    sh.shadow_intensity = 0.28 + 0.3 * con
    sc.display.shadow_shift = 0.02
    sc.display.shadow_focus = 0.6
    sc.display.light_direction = tuple(nrm(np.array([kv[0] * 0.8, 0.8, 0.45])))
    sh.show_cavity = True
    sh.cavity_type = 'BOTH'
    sh.cavity_ridge_factor = 0.6
    sh.cavity_valley_factor = 0.9
    sh.curvature_ridge_factor = 0.5
    sh.curvature_valley_factor = 0.7
    sc.display.matcap_ssao_distance = 0.06
    sc.display.matcap_ssao_attenuation = 1.0
    sc.display.matcap_ssao_samples = 8
    sh.show_object_outline = True
    sh.object_outline_color = (0.08, 0.06, 0.06)
    dbg = os.environ.get('RB_OFF', '')
    if 'shadow' in dbg: sh.show_shadows = False
    if 'cavity' in dbg: sh.show_cavity = False
    if 'outline' in dbg: sh.show_object_outline = False
    if 'screen' in dbg: sh.cavity_type = 'WORLD'
    if 'world' in dbg: sh.cavity_type = 'SCREEN'
    log('studio light gain k (linear, per channel) =', np.round(k, 3))
    return k


def draw_panel_image(bg, x0, x1, W, H, k):
    """backdrop texture for one panel in output sRGB, compensated by the studio-light gain -> linear albedo."""
    pw = max(1, int(round((x1 - x0) * W)))
    base = np.array(hex2rgb(bg.get('base_color') or '#808080'))
    img = np.zeros((H, pw, 3))
    gb = bg.get('gradient_bottom')
    if gb:
        bot = np.array(hex2rgb(gb))
        tcol = np.linspace(0, 1, H)[:, None] ** 1.2
        img[:] = (srgb2lin(base)[None] * (1 - tcol) + srgb2lin(bot)[None] * tcol)[:, None, :]
        img = lin2srgb(img)
    else:
        img[:] = base
    yy, xx = np.mgrid[0:H, 0:pw]
    u = (xx + 0.5) / W + x0
    v = (yy + 0.5) / H
    for shp in bg.get('shapes') or []:
        try:
            kind = str(shp.get('kind', 'rect')).lower()
            sx0, sy0 = fnum(shp.get('x0'), 0), fnum(shp.get('y0'), 0)
            sx1, sy1 = fnum(shp.get('x1'), 1), fnum(shp.get('y1'), 1)
            sx0, sx1 = min(sx0, sx1), max(sx0, sx1)
            sy0, sy1 = min(sy0, sy1), max(sy0, sy1)
            c = np.array(hex2rgb(shp.get('color') or '#606060'))
            aa = 1.0 / H
            if kind == 'ellipse':
                cx, cy = (sx0 + sx1) / 2, (sy0 + sy1) / 2
                rx, ry = max(1e-4, (sx1 - sx0) / 2), max(1e-4, (sy1 - sy0) / 2)
                dd = np.sqrt(((u - cx) / rx) ** 2 + ((v - cy) / ry) ** 2)
                a = np.clip((1 - dd) * min(rx * W, ry * H) / 1.0, 0, 1)
            else:
                if kind == 'vertical_band':
                    sy0, sy1 = (sy0, sy1) if (sy1 - sy0) > 0.01 else (0.0, 1.0)
                ax = np.clip(np.minimum(u - sx0, sx1 - u) * W + 0.5, 0, 1)
                ay = np.clip(np.minimum(v - sy0, sy1 - v) * H + 0.5, 0, 1)
                a = ax * ay
                if kind == 'floor':
                    c2 = c
                    img = img * (1 - a[..., None]) + c2 * a[..., None]
                    continue
            img = img * (1 - a[..., None]) + c * a[..., None]
        except Exception as ex:
            log('shape skipped', shp, ex)
    lin = srgb2lin(img) / np.maximum(k, 1e-3)[None, None]
    return np.clip(lin, 0, 1), pw


def build_backdrop(sc, scene_js, R, k, coll):
    W, H = R['W'], R['H']
    D = 30.0   # far enough that no body shadow can reach the visible backdrop
    th = math.tan(math.radians(R['vfov']) / 2)
    objs = []
    for i, pn in enumerate(scene_js['panels']):
        x0, x1 = pn['x0'], pn['x1']
        lin, pw = draw_panel_image(pn.get('background') or {}, x0, x1, W, H, k)
        img = bpy.data.images.new('bg%d' % i, pw, H, alpha=False, float_buffer=True)
        try:
            img.colorspace_settings.name = 'Linear Rec.709'
        except Exception:
            try:
                img.colorspace_settings.name = 'Linear'
            except Exception:
                pass
        rgba = np.ones((H, pw, 4), np.float32)
        rgba[..., :3] = lin[::-1]
        img.pixels.foreach_set(rgba.ravel())
        img.pack()
        m = bpy.data.materials.new('bgmat%d' % i)
        m.use_nodes = True
        nt = m.node_tree
        tex = nt.nodes.new('ShaderNodeTexImage')
        tex.image = img
        tex.interpolation = 'Closest'
        tex.extension = 'EXTEND'
        bsdf = nt.nodes.get('Principled BSDF')
        if bsdf:
            nt.links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
        nt.nodes.active = tex
        m.diffuse_color = (1, 1, 1, 1)
        m.roughness = 1.0
        X0 = (x0 - 0.5) * (W / H) * 2 * D * th
        X1 = (x1 - 0.5) * (W / H) * 2 * D * th
        Y0, Y1 = -D * th, D * th
        me = bpy.data.meshes.new('bg%d' % i)
        me.from_pydata([(X0, Y0, -D), (X1, Y0, -D), (X1, Y1, -D), (X0, Y1, -D)], [], [(0, 1, 2, 3)])
        uv = me.uv_layers.new(name='UVMap')
        for li, (a, b) in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)]):
            uv.data[li].uv = (a, b)
        me.materials.append(m)
        ob = bpy.data.objects.new('backdrop%d' % i, me)
        coll.objects.link(ob)
        objs.append(ob)
    return objs


def default_scene(R):
    P = R['P']
    if P == 2:
        panels = [{'x0': 0.0, 'x1': 0.5, 'background': {'base_color': '#6a6560'}},
                  {'x0': 0.5, 'x1': 1.0, 'background': {'base_color': '#5f6266'}}]
    else:
        panels = [{'x0': 0.0, 'x1': 1.0, 'background': {'base_color': '#6b6b6b', 'gradient_bottom': '#4f4f4f'}}]
    return {'panels': panels, 'lighting': {}, 'people': [{} for _ in range(P)]}


def load_scene(path, R):
    js = {}
    if path and os.path.exists(path):
        try:
            js = json.load(open(path))
        except Exception as ex:
            log('scene json unreadable, using defaults:', ex)
    d = default_scene(R)
    panels = js.get('panels') if isinstance(js.get('panels'), list) and js.get('panels') else d['panels']
    clean = []
    for pn in panels:
        if not isinstance(pn, dict):
            continue
        x0 = min(1, max(0, fnum(pn.get('x0'), 0.0)))
        x1 = min(1, max(0, fnum(pn.get('x1'), 1.0)))
        if x1 - x0 < 0.05:
            continue
        clean.append({'x0': x0, 'x1': x1, 'background': pn.get('background') if isinstance(pn.get('background'), dict) else {}})
    if not clean:
        clean = d['panels']
    clean.sort(key=lambda q: q['x0'])
    people = js.get('people') if isinstance(js.get('people'), list) else []
    people = [pp if isinstance(pp, dict) else {} for pp in people]
    while len(people) < R['P']:
        people.append({})
    return {'panels': clean, 'lighting': js.get('lighting') if isinstance(js.get('lighting'), dict) else {}, 'people': people}


# ----------------------------------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------------------------------
def parse_args():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    a = dict(rig=None, scene=None, out=None, frames=None, still=None, png=None, workers=None, aa='8', framedir=None, keep=None, crop=None, size='384')
    i = 0
    while i < len(argv):
        k = argv[i].lstrip('-').replace('-', '_')
        if k in ('keep_frames',):
            k = 'keep'
        v = argv[i + 1] if i + 1 < len(argv) else None
        a[k] = v
        i += 2
    return a


def frames_range(spec, T):
    if not spec:
        return list(range(T))
    a, b = spec.split(':')
    a = int(a) if a else 0
    b = int(b) if b else T
    return list(range(max(0, a), min(T, b)))


def encode(framedir, panels, frames, fps, out, W, H, png=None):
    n = len(frames)
    ff = ['ffmpeg', '-v', 'error', '-y']
    if len(panels) == 1:
        ff += ['-framerate', str(fps), '-start_number', str(frames[0]), '-i', os.path.join(framedir, 'p0_%05d.png')]
        vf = 'scale=%d:%d:flags=neighbor' % (W, H)
        filt = ['-vf', vf]
    else:
        parts = []
        for i in range(len(panels)):
            ff += ['-framerate', str(fps), '-start_number', str(frames[0]), '-i', os.path.join(framedir, 'p%d_%%05d.png' % i)]
            parts.append('[%d:v]' % i)
        fc = ''.join(parts) + 'hstack=inputs=%d,scale=%d:%d:flags=neighbor' % (len(panels), W, H)
        for pn in panels[1:]:
            x = int(round(pn['x0'] * W))
            fc += ',drawbox=x=%d:y=0:w=3:h=%d:color=0x141414:t=fill' % (x - 1, H)
        filt = ['-filter_complex', fc]
    if png:
        cmd = ff + filt + ['-frames:v', '1', png]
    else:
        cmd = ff + filt + ['-frames:v', str(n), '-c:v', 'libx264', '-crf', '20', '-preset', 'medium', '-pix_fmt', 'yuv420p',
                           '-r', str(fps), '-an', '-movflags', '+faststart', out]
    subprocess.run(cmd, check=True)


def main():
    a = parse_args()
    t_start = time.time()
    R = load_rig(a['rig'])
    T, W, H = R['T'], R['W'], R['H']
    R['fpx'] = (H / 2) / math.tan(math.radians(R['vfov']) / 2)
    if a['crop']:
        R['crop'] = [float(v) for v in a['crop'].split(',')]; R['out_size'] = int(a['size'])
        W = H = int(a['size'])
    js = load_scene(a['scene'], R)
    panels = js['panels']
    if a['still'] is not None:
        frames = [int(a['still'])]
    else:
        frames = frames_range(a['frames'], T)
    if not frames:
        raise SystemExit('no frames to render')
    nproc = os.cpu_count() or 4
    workers = int(a['workers']) if a['workers'] else max(1, min(8, len(frames) // 12, nproc // 8))
    own_tmp = a['framedir'] is None
    framedir = a['framedir'] or tempfile.mkdtemp(prefix='earlier stage_rb_')
    os.makedirs(framedir, exist_ok=True)

    if workers > 1 and a['framedir'] is None and a['still'] is None:
        # parallel children over contiguous chunks, each renders into the shared framedir
        chunks = np.array_split(np.array(frames), workers)
        procs = []
        for ch in chunks:
            if len(ch) == 0:
                continue
            cmd = [bpy.app.binary_path, '-b', '--factory-startup', '--python', os.path.abspath(__file__), '--',
                   '--rig', a['rig'], '--frames', '%d:%d' % (ch[0], ch[-1] + 1), '--framedir', framedir, '--workers', '1', '--aa', a['aa']]
            if a['scene']:
                cmd += ['--scene', a['scene']]
            if a['crop']:
                cmd += ['--crop', a['crop'], '--size', a['size']]
            lf = open(os.path.join(framedir, 'worker_%d.log' % ch[0]), 'w')
            procs.append((subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT), lf, ch))
        log('spawned %d workers (pids %s)' % (len(procs), [pp.pid for pp, _, _ in procs]))
        failed = False
        try:
            for pp, lf, ch in procs:
                rc = pp.wait()
                lf.close()
                if rc != 0:
                    failed = True
                    log('worker for frames %d..%d failed rc=%d; log %s' % (ch[0], ch[-1], rc, lf.name))
        except BaseException:
            for pp, _, _ in procs:
                if pp.poll() is None:
                    pp.kill()
            raise
        if failed:
            raise SystemExit('worker failure')
    else:
        tmp = tempfile.mkdtemp(prefix='earlier stage_sl_')
        sc = setup_render(R, tmp, a['aa'])
        k = setup_shading(sc, js['lighting'], tmp)
        bgc = bpy.data.collections.new('backdrop')
        sc.collection.children.link(bgc)
        build_backdrop(sc, js, R, k, bgc)
        # people -> panels (by median head x)
        people = []
        pcoll = []
        for p in range(R['P']):
            cl = bpy.data.collections.new('person%d' % p)
            sc.collection.children.link(cl)
            pcoll.append(cl)
            people.append(Person(R, p, js['people'][p] if p < len(js['people']) else {}, cl, R))
        owner = []
        for p, pe in enumerate(people):
            tr = pe.HM[:, :3, 3]
            u = np.median((W / 2 + R['fpx'] * tr[:, 0] / np.maximum(1e-3, -tr[:, 2])) / W)
            pi = [i for i, pn in enumerate(panels) if pn['x0'] <= u <= pn['x1']]
            owner.append(pi[0] if pi else min(range(len(panels)), key=lambda i: abs((panels[i]['x0'] + panels[i]['x1']) / 2 - u)))
        log('setup %.1fs; face_mode=%s owners=%s' % (time.time() - t_start, [pe.face_mode for pe in people], owner))
        multi = len(panels) > 1 and not os.environ.get('RB_ONEPASS')
        t0 = time.time()
        tpose = trend = 0.0
        for n, t in enumerate(frames):
            for pi, pn in enumerate(panels):
                for p, pe in enumerate(people):
                    vis = (not multi) or owner[p] == pi
                    if multi:
                        for o in pcoll[p].objects:
                            if o.type == 'MESH':
                                o.hide_render = not vis
                    if vis or n == 0:
                        ta = time.time()
                        pe.pose(t, (pn['x0'], pn['x1']) if vis else (0.0, 1.0))
                        tpose += time.time() - ta
                if multi:
                    # render only this panel: narrower image + horizontal lens shift (same frustum slice as a crop)
                    px0, px1 = int(round(pn['x0'] * W)), int(round(pn['x1'] * W))
                    pw = px1 - px0
                    sc.render.resolution_x = pw
                    sc.camera.data.shift_x = ((px0 + px1) / 2.0 - W / 2.0) / max(pw, H)
                    sc.camera.data.shift_y = 0.0
                bpy.context.view_layer.update()
                if os.environ.get('RB_DEBUG'):
                    for p, pe in enumerate(people):
                        for o in pcoll[p].objects:
                            if o.type == 'MESH':
                                V = np.array([o.matrix_world @ v.co for v in o.data.vertices]) if len(o.data.vertices) else np.zeros((1, 3))
                                log('dbg', pi, o.name, 'hide', o.hide_render, pcoll[p].hide_render, 'min', np.round(V.min(0), 3), 'max', np.round(V.max(0), 3), 'nan', np.isnan(V).any())
                ta = time.time()
                render_to(os.path.join(framedir, 'p%d_%05d.png' % (pi, t)))
                trend += time.time() - ta
            if n % 25 == 0 or n == len(frames) - 1:
                el = time.time() - t0
                log('frame %d (%d/%d) %.3f s/frame (pose %.3f render %.3f)' % (t, n + 1, len(frames), el / (n + 1), tpose / (n + 1), trend / (n + 1)))
        log('render loop %.1fs for %d frames = %.3f s/frame' % (time.time() - t0, len(frames), (time.time() - t0) / len(frames)))
        shutil.rmtree(tmp, ignore_errors=True)
    if a['framedir'] is not None and own_tmp is False:
        return   # child worker: frames only
    if a['still'] is not None:
        png = a['png'] or (os.path.splitext(a['out'] or 'still.mp4')[0] + '_%05d.png' % frames[0])
        encode(framedir, panels, frames, R['fps'], None, W, H, png=png)
        log('wrote', png)
    else:
        if not a['out']:
            raise SystemExit('--out required')
        os.makedirs(os.path.dirname(os.path.abspath(a['out'])), exist_ok=True)
        encode(framedir, panels, frames, R['fps'], a['out'], W, H)
        log('wrote %s (%d frames) total %.1fs = %.3f s/frame wall' % (a['out'], len(frames), time.time() - t_start, (time.time() - t_start) / len(frames)))
    if a['keep']:
        os.makedirs(a['keep'], exist_ok=True)
        for f in glob.glob(os.path.join(framedir, 'p*_*.png')):
            shutil.copy(f, a['keep'])
    if own_tmp:
        shutil.rmtree(framedir, ignore_errors=True)


if __name__ == '__main__':
    try:
        main()
    except SystemExit as ex:
        if ex.code not in (None, 0):
            print('[render] ERROR', ex, flush=True)
            sys.stdout.flush()
            os._exit(1)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)
