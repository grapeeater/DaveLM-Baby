from __future__ import annotations
import json, hashlib, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(r'C:\DaveLM-CADAVER'); OUT=ROOT/'treatment13_nonsource_localization_suppression_seed8380'; SRC=ROOT/'treatment13_learned_mapping_row_localization_seed8380'
sys.path[:0]=[str(ROOT),r'C:\DaveLM-v0.9']
from treatment13_model import Treatment13Model
from treatment13_nonsource_preflight import rowpos, lexisting, lfiller
TRAIN=SRC/'treatment13_training_quartet_pool.json'; SCHED=SRC/'treatment13_frozen_schedule.json'; START=SRC/'checkpoints/learned_mapping_row_localization/seed_8380/latest.pt'; FINAL=OUT/'checkpoints/nonsource_localization_suppression/seed_8380/latest.pt'
def main():
 dev=torch.device('cuda'); assert torch.cuda.is_available(); pre=json.loads((OUT/'CORRECTIVE_PREFLIGHT_AUDIT.json').read_text()); le=float(pre['lambda_existing']); lf=float(pre['lambda_filler']); pool=json.loads(TRAIN.read_text()); sch=json.loads(SCHED.read_text()); qm={q['quartet_id']:q for q in pool['quartets']}; torch.manual_seed(8380); m=Treatment13Model().to(dev); ck=torch.load(START,map_location=dev,weights_only=False); m.load_state_dict(ck['model_state_dict']); m.train(); opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=.05); metrics=[]; t0=time.time()
 for step,s in enumerate(sch['steps_data'],1):
  docs=[]
  for q in s['quartet_ids']: docs.extend(qm[q]['docs'])
  x=torch.tensor([d['full_document_token_ids'] for d in docs],device=dev); qp=torch.tensor([d['qdp'] for d in docs],device=dev); ap=torch.tensor([d['answer_causal_position'] for d in docs],device=dev); y=torch.tensor([d['target_value_token'] for d in docs],device=dev); opt.zero_grad(set_to_none=True); logits,ex=m(x,qp,ap); rows=torch.arange(len(docs),device=dev); La=F.cross_entropy(logits[rows,ap],y); Le=lexisting(ex['localization_attention'],docs); Lf=lfiller(ex['localization_attention'],docs,qp,x.shape[1]); total=La+le*Le+lf*Lf; total.backward(); gn=torch.nn.utils.clip_grad_norm_(m.parameters(),2.0); opt.step()
  with torch.no_grad():
   att=ex['localization_attention']; cats=[]
   for j,d in enumerate(docs):
    s0,s1=rowpos(d); p0=int(att[j,:,0].argmax())+1;p1=int(att[j,:,1].argmax())+1; T={s0,s1}; cats.append('B' if p0!=p1 and {p0,p1}==T else 'S' if p0==p1 else 'E' if sum(z in T for z in (p0,p1))==1 else 'N')
  metrics.append({'step':step,'L_total':float(total.item()),'L_answer':float(La.item()),'L_existing':float(Le.item()),'L_filler':float(Lf.item()),'answer_accuracy':float((logits[rows,ap].argmax(-1)==y).float().mean().item()),'both_distinct_rate':cats.count('B')/len(cats),'exactly_one_rate':cats.count('E')/len(cats),'slot_collapse_rate':cats.count('S')/len(cats),'neither_rate':cats.count('N')/len(cats),'grad_norm':float(gn.item())})
  if step%50==0: print(metrics[-1],flush=True)
 (OUT/'training_metrics.jsonl').write_text(''.join(json.dumps(z)+'\n' for z in metrics)); FINAL.parent.mkdir(parents=True,exist_ok=True); torch.save({'model_state_dict':m.state_dict(),'step':1000,'seed':8380,'lambda_existing':le,'lambda_filler':lf,'objective':'answer_ce_plus_existing_permutation_invariant_localization_nll_plus_nonsource_filler_penalty'},FINAL); out={'status':'TREATMENT_TRAINED','steps':1000,'lambda_existing':le,'lambda_filler':lf,'final_checkpoint_sha256':hashlib.sha256(FINAL.read_bytes()).hexdigest(),'runtime_seconds':time.time()-t0,'final_step':metrics[-1]}; (OUT/'TRAINING_RESULT.json').write_text(json.dumps(out,indent=2)); print(out)
if __name__=='__main__': main()
