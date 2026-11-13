set -e
mkdir -p /w && cd /w
command -v ffmpeg >/dev/null || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg rsync >/dev/null)
[ -d ds ] || (git clone -q https://github.com/modelscope/DiffSynth-Studio.git ds && cd ds && git checkout -q 28638042df62d206486a8bcff3ef22973c84e414)
cd /w/ds && pip install --break-system-packages -q -e . hf_transfer 2>&1 | grep -i error | grep -v resolver || true
pip uninstall -y --break-system-packages torchaudio >/dev/null 2>&1 || true
cd /w && HF_HUB_ENABLE_HF_TRANSFER=1 python -c "
from huggingface_hub import snapshot_download as s
s('Wan-AI/Wan2.1-VACE-1.3B', local_dir='models/Wan-AI/Wan2.1-VACE-1.3B')
s('Wan-AI/Wan2.1-T2V-1.3B', local_dir='models/Wan-AI/Wan2.1-T2V-1.3B', allow_patterns=['google/umt5-xxl/*'])
s('Kijai/WanVideo_comfy', local_dir='models/kijai', allow_patterns=['Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors'])
s('depth-anything/Depth-Anything-V2-Small-hf')
" >/w/dl.log 2>&1
echo SETUP_OK
