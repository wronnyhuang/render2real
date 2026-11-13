set -x
cd /w
while pgrep -f "sft:data_process" >/dev/null; do sleep 20; done
bash sync/pod/train_split.sh train /w/lora_full 2 > train_full.log 2>&1
L=$(ls -t /w/lora_full/*.safetensors | head -1); echo LORA=$L
CV=/w/models/kijai/Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors
CL="M000_P0115=sync/pairs/M000_P0115/render.mp4,M001_P0115=sync/pairs/M001_P0115/render.mp4,M002_P0116_trainsample=sync/pairs/M002_P0116/render.mp4,M006_P0216_trainsample=sync/pairs/M006_P0216/render.mp4,M009_P0217=sync/pairs/M009_P0217/render.mp4,M023_P0370=sync/pairs/M023_P0370/render.mp4,M037_P0485=sync/pairs/M037_P0485/render.mp4,M038_P0485=sync/pairs/M038_P0485/render.mp4,M075_P0845=sync/pairs/M075_P0845/render.mp4,M076_P0845=sync/pairs/M076_P0845/render.mp4,M092_P0989=sync/pairs/M092_P0989/render.mp4,M099_P1067=sync/pairs/M099_P1067/render.mp4,M100_P1067=sync/pairs/M100_P1067/render.mp4,M103_P1174=sync/pairs/M103_P1174/render.mp4,juno_01=sync/drive/juno_01.mp4,juno_02=sync/drive/juno_02.mp4,juno_03=sync/drive/juno_03.mp4,juno_04=sync/drive/juno_04.mp4,muse_01=sync/drive/muse_01.mp4,muse_02=sync/drive/muse_02.mp4,muse_03=sync/drive/muse_03.mp4,muse_04=sync/drive/muse_04.mp4"
python sync/pod/infer.py --variant lora_causvid_s6 --control raw --steps 6 --cfg 1 --shift 8 --lora $L --causvid $CV --clips "$CL" > inf_cv.log 2>&1
python sync/pod/infer.py --variant lora_raw_s20 --control raw --steps 20 --lora $L --clips "$CL" > inf_s20.log 2>&1
echo ALL_DONE
