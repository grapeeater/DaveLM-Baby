from __future__ import annotations
import json,hashlib,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(r'C:\DaveLM-CADAVER'); OUT=ROOT/'treatment13_distinct_localization_supervision_seed8380'; SRC=ROOT/'treatment13_learned_mapping_row_localization_seed8380'
sys.path.insert(0,str(ROOT)); sys.path.insert(0,r'C:\DaveLM-v0.9')
from treatment13_model import Treatment13Model
from treatment13_distinct_preflight import rowpos,loc_loss
TRAIN=SRC/'treatment13_training_quartet_pool.json'; SCHED=SRC/'treatment13_frozen_schedule.json'; START=SRC/'treatment13_learned_mapping_row_localization_seed8380'/'checkpoints'
START=SRC/'checkpoints'/'learned_mapping_row_localization'/'seed_8380'/'latest.pt'
FINAL=OUT/'checkpoints'/'distinct_localization_supervision'/'seed_8380'/'latest.pt'; METRICS=OUT/'training_metrics.jsonl'; RESULT=OUT/'TRAINING_RESULT.json'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def state(c):
 for k in ('model_state_dict','model_state','model','state_dict'):
  if isinstance(c,dict) and isinstance(c.get(k),dict): return c[k]
 raise RuntimeError('checkpoint')
def main():
 import argparse; ap=argparse.ArgumentParser(); ap.add_argument('--device',default='cuda'); a=ap.parse_args(); assert torch.cuda.is_available(); dev=torch.device(a.device)
 pre=json.loads((OUT/'PREFLIGHT_AUDIT.json').read_text()); lam=float(pre['lambda']); pool=json.loads(TRAIN.read_text()); sch=json.loads(SCHED.read_text()); byq={q['quartet_id']:q for q in pool['quartets']}; assert len(sch['steps_data'])==1000
 torch.manual_seed(8380); model=Treatment13Model().to(dev); model.load_state_dict(state(torch.load(START,map_location=dev))); model.train(); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.05); metrics=[]; t0=time.time()
 for i,s in enumerate(sch['steps_data']):
  docs=[]
  for q in s['quartet_ids']: docs.extend(byq[q]['docs'])
  x=torch.tensor([d['full_document_token_ids'] for d in docs],dtype=torch.long,device=dev); qp=torch.tensor([d['qdp'] for d in docs],device=dev); ap=torch.tensor([d['answer_causal_position'] for d in docs],device=dev); y=torch.tensor([d['target_value_token'] for d in docs],device=dev); opt.zero_grad(set_to_none=True); logits,ex=model(x,qp,ap); rows=torch.arange(len(docs),device=dev); la=F.cross_entropy(logits[rows,ap],y); li=loc_loss(ex['localization_attention'],docs); total=la+lam*li; total.backward(); gn=torch.nn.utils.clip_grad_norm_(model.parameters(),2.0); opt.step();
  with torch.no_grad():
   att=ex['localization_attention']; cats=[]
   for j,d in enumerate(docs):
    s0,s1=rowpos(d); p0=int(att[j,:,0].argmax())+1; p1=int(att[j,:,1].argmax())+1; T={s0,s1}; cats.append('B' if p0!=p1 and {p0,p1}==T else 'S' if p0==p1 else 'E' if sum(z in T for z in (p0,p1))==1 else 'N')
  metrics.append({'step':i+1,'L_total':float(total.item()),'L_answer':float(la.item()),'L_localization':float(li.item()),'answer_accuracy':float((logits[rows,ap].argmax(-1)==y).float().mean().item()),'both_distinct_rate':cats.count('B')/len(cats),'exactly_one_rate':cats.count('E')/len(cats),'slot_collapse_rate':cats.count('S')/len(cats),'neither_rate':cats.count('N')/len(cats),'grad_norm':float(gn.item())})
  if (i+1)%50==0: print(metrics[-1],flush=True)
 METRICS.write_text(''.join(json.dumps(m)+'\n' for m in metrics)); FINAL.parent.mkdir(parents=True,exist_ok=True); torch.save({'model_state_dict':model.state_dict(),'step':1000,'seed':8380,'lambda':lam,'objective':'answer_ce_plus_permutation_invariant_localization_nll'},FINAL); out={'status':'TREATMENT13_DISTINCT_LOCALIZATION_TRAINED','steps':1000,'lambda':lam,'final_checkpoint_sha256':sha(FINAL),'final_step':metrics[-1],'runtime_seconds':time.time()-t0}; RESULT.write_text(json.dumps(out,indent=1)); print(out)
if __name__=='__main__': main()
