cd /w
while ! grep -q ALL_DONE run_all.log; do sleep 15; done
L=/w/lora_full/epoch-1.safetensors; CV=/w/models/kijai/Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors
C=M009_P0217=sync/pairs/M009_P0217/render.mp4,juno_02=sync/drive/juno_02.mp4
python sync/pod/infer.py --out sweep --variant sw_causvid_s4 --control raw --steps 4 --cfg 1 --shift 8 --lora $L --causvid $CV --clips $C
python sync/pod/infer.py --out sweep --variant sw_causvid_a05_s8 --control raw --steps 8 --cfg 1 --shift 8 --lora $L --causvid $CV --causvid_alpha 0.5 --clips $C
python sync/pod/infer.py --out sweep --variant sw_raw_s10 --control raw --steps 10 --lora $L --clips $C
python sync/pod/infer.py --out sweep --variant sw_raw_s30 --control raw --steps 30 --lora $L --clips M009_P0217=sync/pairs/M009_P0217/render.mp4
echo SWEEP_DONE
