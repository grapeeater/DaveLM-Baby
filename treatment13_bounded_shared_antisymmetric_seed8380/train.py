import hashlib,json,random,time,sys
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(r'C:\DaveLM-CADAVER'); SRC=ROOT/'treatment13_learned_mapping_row_localization_seed8380'; OUT=ROOT/'treatment13_bounded_shared_antisymmetric_seed8380'; sys.path.insert(0,str(ROOT)); sys.path.insert(0,r'C:\DaveLM-v0.9')
from bounded_model import make_bounded_model
from treatment13_distinct_localization_supervision_seed8380.treatment13_distinct_preflight import rowpos,loc_loss
CK=SRC/'checkpoints/learned_mapping_row_localization/seed_8380/latest.pt'; TRAIN=SRC/'treatment13_training_quartet_pool.json'; SCHED=SRC/'treatment13_frozen_schedule.json'; FINAL=OUT/'checkpoints/seed_8380/latest.pt'; MET=OUT/'training_metrics.jsonl'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert torch.cuda.is_available(); dev=torch.device('cuda'); pool=json.loads(TRAIN.read_text()); sched=json.loads(SCHED.read_text()); byq={q['quartet_id']:q for q in pool['quartets']}; ck=torch.load(CK,map_location=dev,weights_only=False); state=ck['model_state_dict']; rho=1.5919504165649414
 torch.manual_seed(8380); random.seed(8380); m=make_bounded_model(state,rho,dev); m.train(); assert sum(p.numel() for p in m.localizer.parameters())==642; opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=.05); t0=time.time(); recs=[]
 for step,s in enumerate(sched['steps_data'],1):
  docs=[]
  for qid in s['quartet_ids']: docs.extend(byq[qid]['docs'])
  x=torch.tensor([d['full_document_token_ids'] for d in docs],device=dev); qp=torch.tensor([d['qdp'] for d in docs],device=dev); ap=torch.tensor([d['answer_causal_position'] for d in docs],device=dev); y=torch.tensor([d['target_value_token'] for d in docs],device=dev); opt.zero_grad(set_to_none=True); logits,ex=m(x,qp,ap); ix=torch.arange(len(docs),device=dev); al=logits[ix,ap]; la=F.cross_entropy(al,y); li=loc_loss(ex['localization_attention'],docs); total=la+float(1.0536573711078283)*li; total.backward(); gn=torch.nn.utils.clip_grad_norm_(m.parameters(),2.0); assert torch.isfinite(gn); opt.step()
  with torch.no_grad():
   att=ex['localization_attention']; cats=[]
   for i,d in enumerate(docs):
    s0,s1=rowpos(d); p0=int(att[i,:,0].argmax())+1; p1=int(att[i,:,1].argmax())+1; T={s0,s1}; cats.append('B' if p0!=p1 and {p0,p1}==T else 'S' if p0==p1 else 'E' if sum(z in T for z in (p0,p1))==1 else 'N')
  r={'step':step,'L_total':float(total),'L_answer':float(la),'L_localization':float(li),'answer_accuracy':float((al.argmax(-1)==y).float().mean()),'both_distinct_rate':cats.count('B')/32,'exactly_one_rate':cats.count('E')/32,'slot_collapse_rate':cats.count('S')/32,'grad_norm':float(gn)}; recs.append(r)
  if step%50==0: print(r,flush=True)
 MET.write_text(''.join(json.dumps(r)+'\n' for r in recs)); FINAL.parent.mkdir(parents=True,exist_ok=True); torch.save({'model_state_dict':m.state_dict(),'step':1000,'seed':8380,'rho':rho,'lambda_existing':1.0536573711078283,'objective':'answer_ce_plus_champion_hard_min_localization'},FINAL); print(json.dumps({'steps':1000,'final_checkpoint_sha256':sha(FINAL),'runtime_seconds':time.time()-t0,'train_pool_sha256':sha(TRAIN),'schedule_sha256':sha(SCHED)}))
if __name__=='__main__': main()
