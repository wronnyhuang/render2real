cd /w
while ! grep -q SWEEP_DONE sweep.log; do sleep 15; done
CV=/w/models/kijai/Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors
C=M009_P0217=sync/pairs/M009_P0217/render.mp4,muse_01=sync/drive/muse_01.mp4
P="--prompts sync/pod/prompts.json --out test"
python sync/pod/infer.py $P --variant t_e1_prompt_s20 --control raw --steps 20 --lora /w/lora_full/epoch-1.safetensors --clips $C
python sync/pod/infer.py $P --variant t_e0_prompt_s20 --control raw --steps 20 --lora /w/lora_full/epoch-0.safetensors --clips $C
python sync/pod/infer.py $P --variant t_e1a06_prompt_s20 --control raw --steps 20 --lora /w/lora_full/epoch-1.safetensors --lora_alpha 0.6 --clips $C
python sync/pod/infer.py $P --variant t_e1_prompt_cv6 --control raw --steps 6 --cfg 1 --shift 8 --lora /w/lora_full/epoch-1.safetensors --causvid $CV --clips $C
echo TEST_DONE
