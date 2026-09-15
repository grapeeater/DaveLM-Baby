import json, sys, torch
from pathlib import Path
sys.path.insert(0, r'C:\DaveLM-CADAVER\phase2a_t32_causal_ordered_source_memory_v1')
from T32_RUNNER import load_jsonl, _forward_row, T3_DATA
from T32_MEMORY import T32MemoryModel
from baby_vnext.config import BabyVNextConfig

b=Path(r'C:\DaveLM-CADAVER\phase2a_t32_causal_ordered_source_memory_v1'); c=BabyVNextConfig.load(r'C:\DaveLM-CADAVER\baby_vnext_60m_design_v1\BABY_VNEXT_CONFIG.json'); rows=load_jsonl(T3_DATA/'qa_dev.jsonl'); out=[]
for seed in (890001,890002):
    p=torch.load(b/f'run_seed{seed}'/'checkpoints'/'checkpoint_1000.pt',map_location='cpu',weights_only=False); m=T32MemoryModel(c); m.load_state_dict(p['model_state_dict'],strict=True); m.cuda().eval(); masses=[]; sels=[]
    with torch.no_grad():
        for r in rows:
            prompt=[2]+[int(x) for x in r['prompt_token_ids']]; target=[int(x) for x in r['candidate_token_ids'][int(r['correct_index'])]]+[3]; _,st=_forward_row(m,prompt+target[:-1],r,torch.device('cuda'),len(prompt)-1)
            mm=st['memory_mass'][0,len(prompt)-1:].float(); masses.append(float(mm.mean().cpu())); pw=st['pointer_weights'][0]; sels.append(float(pw[int(r['correct_index'])].cpu()))
    out.append({'seed':seed,'mean_answer_memory_attention_mass':sum(masses)/len(masses),'mean_correct_source_pointer_weight':sum(sels)/len(sels),'n':len(rows)})
(b/'T32_MEMORY_TELEMETRY.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
