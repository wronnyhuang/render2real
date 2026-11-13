import json, collections, statistics
D=__import__('os').path.expandvars('$WORK/wan0/')
rows=[json.loads(l) for l in open(D+'latency.jsonl')]
g=collections.defaultdict(list)
for r in rows: g[r['variant']].append(r)
lat3={'M001_P0115','juno_01','muse_01'}
def summ(v,rs,label=None):
    spf=statistics.median(r['s_per_frame'] for r in rs)
    return dict(variant=label or v,gpu=rs[0]['gpu'],steps=rs[0]['steps'],cfg=rs[0]['cfg'],clips=len(rs),clip_ids=[r['clip'] for r in rs] if len(rs)<5 else None,
        frames_per_clip=rs[0]['frames_generated'],resolution='384x384',fps_generated=16,
        median_wall_s_per_clip=round(statistics.median(r['wall_s'] for r in rs),1),
        median_s_per_generated_frame=round(spf,4),ms_per_frame=round(spf*1000,1),fps=round(1/spf,2),
        realtime_factor_vs_16fps=round(16*spf,2),peak_mem_gb=max(r['peak_mem_gb'] for r in rs),
        lora=rs[0]['lora'],causvid=rs[0]['causvid'],control=rs[0]['control'],prompts=('prompt' in v))
summary=[summ(v,rs) for v,rs in g.items()]
for v in ('base_causvid_s6','base_prompt_cv6'):
    summary.append(summ(v,[r for r in g[v] if r['clip'] in lat3],v+'__same3clips'))
note=('s_per_generated_frame = wall time of the whole pipe() call (text encode + VACE/VAE encode of the control video + '
      'all denoising steps + VAE decode) / 129 frames generated at 16 fps; pipeline load time excluded; batch=1, bf16, '
      'no torch.compile, no TeaCache, no FP8, single GPU. realtime_factor = seconds of compute per second of 16 fps video.')
json.dump(dict(note=note,comparison=('all rows from one pod (r2r-wan0), same GPU. zlat_lora_prompt_cv6 = render2real fine-tuned VACE LoRA epoch-1 '
      '+ CausVid, identical to lora_prompt_cv6, on 3 clips (M001_P0115, juno_01, muse_01); "__same3clips" rows give the base variants on those same 3 clips. '
      'The first clip of each process includes CUDA warm-up.'),summary=summary,rows=rows),open(D+'latency.json','w'),indent=1)
for s in summary: print(s['variant'],s['clips'],s['median_s_per_generated_frame'],s['ms_per_frame'],s['fps'],s['peak_mem_gb'])
