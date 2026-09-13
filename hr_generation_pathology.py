import json,math,sys,torch
from pathlib import Path
ROOT=Path(r'C:\\DaveLM-CADAVER'); sys.path[:0]=[str(ROOT),r'C:\\DaveLM-v0.9']; from treatment13_model import Treatment13Model
from fact_supervision_87001_eval_v1.PINNED_PILOT1_BINDING_IMPLEMENTATION import OrthoLocalizer
from tokenizers import Tokenizer
TOK=Tokenizer.from_file(r'C:\\DaveLM-v0.9\\tokenizer\\v0_7\\davelm_tokenizer.json')
CK={'Pilot1':ROOT/'language_pilot_1_early_block_protection_seed8380/pilot_run/checkpoints/seed_8380/latest.pt','HR1':ROOT/'human_readiness_hr1_seed87004_v8/run/checkpoint_500.pt','HR2':ROOT/'human_readiness_hr2_seed87005_v7/run/checkpoint_500.pt'}
PROMPTS=['The dog ran to the park because','Mia found a red ball and','The small cat sat on the mat. It','A boy opened the box and','Tell me one thing about a sunny day.']
def load(p):
 sd=dict(torch.load(p,map_location='cuda',weights_only=True)['model_state_dict']); v=[sd.pop('localizer.'+k) for k in ('u','q','bs','ba')]; m=Treatment13Model(); m.load_state_dict(sd,strict=False); m.localizer=OrthoLocalizer(*v); return m.to('cuda').eval()
def trace(m,p,temp=1.0,topk=None):
 ids=[2]+TOK.encode(p).ids; g=list(ids); rows=[]
 with torch.no_grad():
  for step in range(32):
   z=m.base_model(torch.tensor([g[-256:]],device='cuda'))[0,-1]; pr=(z/temp).softmax(-1); vals,ix=torch.topk(pr,topk) if topk else (pr,torch.arange(pr.numel(),device=pr.device));
   if topk: q=torch.zeros_like(pr); q[ix]=vals; pr=q/q.sum()
   s=sorted(pr.tolist(),reverse=True); top=float(s[0]); second=float(s[1]); ent=float(-(pr.clamp_min(1e-12)*pr.clamp_min(1e-12).log()).sum()); n=int(pr.argmax()); rows.append({'step':step+1,'token_id':n,'token':TOK.decode([n],skip_special_tokens=True),'top1_prob':top,'top1_margin':top-second,'entropy':ent}); g.append(n)
   if n==3: break
 return {'prompt':p,'decoded':TOK.decode(g,skip_special_tokens=True),'rows':rows,'length':len(g)-len(ids),'immediate_eos':len(rows)==1 and rows[0]['token_id']==3}
def main():
 out={}
 for name,p in CK.items():
  m=load(p); out[name]={'greedy':[trace(m,x) for x in PROMPTS],'temp07_top20':[trace(m,x,.7,20) for x in PROMPTS]}
 Path(r'C:\DaveLM-CADAVER\generation_pathology_hr123.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf8'); print(json.dumps(out,indent=2,ensure_ascii=False))
main()
