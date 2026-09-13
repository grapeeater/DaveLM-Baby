import json, sys, math
from pathlib import Path
import torch
ROOT=Path(r'C:\\DaveLM-CADAVER'); sys.path[:0]=[str(ROOT),r'C:\\DaveLM-v0.9']
from treatment13_model import Treatment13Model
from fact_supervision_87001_eval_v1.PINNED_PILOT1_BINDING_IMPLEMENTATION import OrthoLocalizer
from tokenizers import Tokenizer
TOK=Tokenizer.from_file(r'C:\\DaveLM-v0.9\\tokenizer\\v0_7\\davelm_tokenizer.json')
CK={
 'Pilot1':ROOT/'language_pilot_1_early_block_protection_seed8380/pilot_run/checkpoints/seed_8380/latest.pt',
 'HR1':ROOT/'human_readiness_hr1_seed87004_v8/run/checkpoint_500.pt',
 'HR2':ROOT/'human_readiness_hr2_seed87005_v7/run/checkpoint_500.pt'}
PROMPTS=['The dog ran to the park because','Mia found a red ball and','The small cat sat on the mat. It','A boy opened the box and','Tell me one thing about a sunny day.']
DEVICE='cuda'
def load(p):
 sd=dict(torch.load(p,map_location=DEVICE,weights_only=True)['model_state_dict']);v=[sd.pop('localizer.'+k) for k in ('u','q','bs','ba')];m=Treatment13Model();m.load_state_dict(sd,strict=False);m.localizer=OrthoLocalizer(*v);return m.to(DEVICE).eval()
def ban_trigrams(logits, seq):
 if len(seq)<3:return logits
 seen={}
 for i in range(len(seq)-2): seen.setdefault((seq[i],seq[i+1]),set()).add(seq[i+2])
 pref=(seq[-2],seq[-1])
 for t in seen.get(pref,set()): logits[t]=-float('inf')
 return logits
def step_pick(logits,seq,mode,gen):
 if mode=='rep_penalty':
  for t in set(seq): logits[t] = logits[t]/1.2 if logits[t]>0 else logits[t]*1.2
  return int(logits.argmax())
 if mode=='no_repeat_trigram':
  return int(ban_trigrams(logits,seq).argmax())
 if mode=='temp07_top20_argmax':
  p=(logits/.7).softmax(-1); v,ix=torch.topk(p,20); q=torch.full_like(p,0); q[ix]=v; return int(q.argmax())
 if mode=='temp07_top20_sample':
  p=(logits/.7).softmax(-1); v,ix=torch.topk(p,20); q=torch.zeros_like(p); q[ix]=v; q=q/q.sum(); return int(torch.multinomial(q,1,generator=gen))
 return int(logits.argmax())
def run(m,p,mode,seed):
 seq=[2]+TOK.encode(p).ids; start=len(seq); rows=[]; gen=torch.Generator(device=DEVICE).manual_seed(seed)
 with torch.inference_mode():
  for k in range(32):
   logits=m.base_model(torch.tensor([seq[-256:]],device=DEVICE))[0,-1].float().clone()
   lp=logits.log_softmax(-1); pr=lp.exp(); vals=torch.topk(logits,2).values
   n=step_pick(logits,seq,mode,gen); rows.append({'step':k+1,'token_id':n,'token':TOK.decode([n],skip_special_tokens=True),'entropy':float(-(pr*lp.clamp_min(-30)).sum()),'top1_prob':float(pr.max()),'top1_margin':float(vals[0]-vals[1])});seq.append(n)
   if n==3:break
 return {'prompt':p,'mode':mode,'decoded':TOK.decode(seq,skip_special_tokens=True),'length':len(seq)-start,'immediate_eos':len(rows)==1 and rows[0]['token_id']==3,'rows':rows}
def main():
 out={}
 modes=['greedy','temp07_top20_argmax','temp07_top20_sample','rep_penalty','no_repeat_trigram']
 for name,p in CK.items():
  m=load(p);out[name]={mode:[run(m,q,mode,87005+i) for i,q in enumerate(PROMPTS)] for mode in modes}
 (ROOT/'generation_pathology_constraints.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf8')
 # concise stdout only
 for name in out:
  print('===',name,'===')
  for mode in modes:
   print('--',mode,'--')
   for q in out[name][mode]: print(q['prompt'],'=>',q['decoded'],'len',q['length'])
if __name__=='__main__':main()
