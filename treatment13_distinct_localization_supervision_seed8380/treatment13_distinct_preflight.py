from __future__ import annotations
import hashlib,json,math,sys
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(r'C:\DaveLM-CADAVER'); OUT=ROOT/'treatment13_distinct_localization_supervision_seed8380'; SRC=ROOT/'treatment13_learned_mapping_row_localization_seed8380'; TRAIN=SRC/'treatment13_training_quartet_pool.json'; RET=SRC/'treatment13_retention_quartet_pool.json'; SCHED=SRC/'treatment13_frozen_schedule.json'; PRE=SRC/'treatment13_preflight_result.json'
sys.path.insert(0,str(ROOT)); sys.path.insert(0,r'C:\DaveLM-v0.9')
from treatment13_model import Treatment13Model
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rowpos(d):
 q=int(d['query_slot']); return (int(d['query_key_clause_pos']),int(d['other_key_pos'])) if q==0 else (int(d['other_key_pos']),int(d['query_key_clause_pos']))
def loc_loss(att,docs):
 out=[]
 for i,d in enumerate(docs):
  s0,s1=rowpos(d); x=att[i].clamp_min(1e-12); out.append(-torch.maximum(x[s0-1,0].log()+x[s1-1,1].log(),x[s1-1,0].log()+x[s0-1,1].log()))
 return torch.stack(out).mean()
def main():
 train=json.loads(TRAIN.read_text()); ret=json.loads(RET.read_text()); sch=json.loads(SCHED.read_text()); pre=json.loads(PRE.read_text()); docs=[d for q in train['quartets'] for d in q['docs']]
 assert pre['status']=='TREATMENT13_LOCALIZATION_PREFLIGHT_PASS' and len(docs)==800 and len([d for q in ret['quartets'] for d in q['docs']])==320 and sch['steps']==1000 and sch['batch_size']==32 and len(sch['steps_data'])==1000
 b=docs[:32]; model=Treatment13Model(); model.train(); x=torch.tensor([d['full_document_token_ids'] for d in b]); q=torch.tensor([d['qdp'] for d in b]); a=torch.tensor([d['answer_causal_position'] for d in b]); y=torch.tensor([d['target_value_token'] for d in b]); logits,ex=model(x,q,a); assert tuple(logits.shape)==(32,193,1024) and tuple(ex['localization_attention'].shape)==(32,188,2)
 toy=torch.full((1,10,2),.05); toy[0,2,0]=.8; toy[0,6,1]=.8; toy=toy/toy.sum(1,keepdim=True); td=[{'query_slot':0,'query_key_clause_pos':3,'other_key_pos':7}]; assert torch.allclose(loc_loss(toy,td),loc_loss(toy.flip(-1),td)) and loc_loss(toy,td)<loc_loss(torch.full_like(toy,.1),td)
 la=F.cross_entropy(logits[torch.arange(32),a],y); li=loc_loss(ex['localization_attention'],b); gs=torch.autograd.grad(li,tuple(model.localizer.parameters()),allow_unused=True); assert torch.isfinite(la) and torch.isfinite(li) and li.item()>0 and any(v is not None and torch.isfinite(v).all() and v.abs().sum()>0 for v in gs); lam=float(la.item()/li.item()); assert math.isfinite(lam) and lam>0
 result={'status':'PRETREATMENT_PASS','experiment':'treatment13_distinct_localization_supervision_seed8380','train_docs':800,'retention_docs':320,'schedule_steps':1000,'train_pool_sha256':sha(TRAIN),'retention_pool_sha256':sha(RET),'schedule_sha256':sha(SCHED),'prior_t13_preflight_sha256':sha(PRE),'initial_mean_answer_loss':float(la.item()),'initial_mean_localization_loss':float(li.item()),'lambda':lam,'permutation_invariance_test':True,'gradient_test':True,'optimizer_steps_so_far':0,'forward_inputs':'input_ids,qpos,anspos only','no_retention_behavior_used':True}
 (OUT/'PREFLIGHT_AUDIT.json').write_text(json.dumps(result,indent=1)); print(json.dumps(result,indent=1)); print('PRETREATMENT_PASS')
if __name__=='__main__': main()
