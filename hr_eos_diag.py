import json,sys
from pathlib import Path
import torch
ROOT=Path(r'C:\\DaveLM-CADAVER'); sys.path[:0]=[str(ROOT),r'C:\\DaveLM-v0.9']
from treatment13_model import Treatment13Model
from fact_supervision_87001_eval_v1.PINNED_PILOT1_BINDING_IMPLEMENTATION import OrthoLocalizer
from tokenizers import Tokenizer
TOK=Tokenizer.from_file(r'C:\\DaveLM-v0.9\\tokenizer\\v0_7\\davelm_tokenizer.json')
CK={'Pilot1':ROOT/'language_pilot_1_early_block_protection_seed8380/pilot_run/checkpoints/seed_8380/latest.pt','HR1':ROOT/'human_readiness_hr1_seed87004_v8/run/checkpoint_500.pt','HR2':ROOT/'human_readiness_hr2_seed87005_v7/run/checkpoint_500.pt'}
PROMPTS=['The dog ran to the park because','Mia found a red ball and','The small cat sat on the mat. It','A boy opened the box and','Tell me one thing about a sunny day.']; DEVICE='cuda'
def load(p):
 sd=dict(torch.load(p,map_location=DEVICE,weights_only=True)['model_state_dict']);v=[sd.pop('localizer.'+k) for k in ('u','q','bs','ba')];m=Treatment13Model();m.load_state_dict(sd,strict=False);m.localizer=OrthoLocalizer(*v);return m.to(DEVICE).eval()
out={}
for name,p in CK.items():
 m=load(p); rows=[]
 for q in PROMPTS:
  ids=[2]+TOK.encode(q).ids
  with torch.inference_mode(): logits=m.base_model(torch.tensor([ids],device=DEVICE))[0,-1].float(); pr=logits.softmax(-1); rank=int((logits>logits[3]).sum())+1
  rows.append({'prompt':q,'eos_probability':float(pr[3]),'eos_rank':rank,'top1_token':int(logits.argmax()),'top1_probability':float(pr.max())})
 out[name]=rows
(ROOT/'generation_pathology_eos.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
for n,rs in out.items(): print(n,'mean_eos_p',sum(r['eos_probability'] for r in rs)/len(rs),'mean_eos_rank',sum(r['eos_rank'] for r in rs)/len(rs))
