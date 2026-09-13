from __future__ import annotations
import hashlib, json, math, statistics, sys, time
from collections import defaultdict, Counter
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT=Path(r"C:\DaveLM-CADAVER")
SRC=ROOT/"treatment13_learned_mapping_row_localization_seed8380"
OUT=ROOT/"treatment13_orthogonal_shared_unbounded_seed8380"
CK=SRC/"checkpoints/learned_mapping_row_localization/seed_8380/latest.pt"
TRAIN=SRC/"treatment13_training_quartet_pool.json"
RET=SRC/"treatment13_retention_quartet_pool.json"
SCHED=SRC/"treatment13_frozen_schedule.json"
PREF=ROOT/"forensics/orthogonal_shared_unbounded_preflight_seed8380/RESULTS.json"
sys.path.insert(0,str(ROOT)); sys.path.insert(0,r"C:\DaveLM-v0.9")
from treatment13_model import Treatment13Model
from treatment13_config import ROW_VALUE_OFFSET

EXP_HASH={
 str(CK):"cf53eef03adf874255cd54508fc7948dfd1e4926a57537f7e2f828f5f582adab",
 str(TRAIN):"40d78c450f2335aa155031a612d5b8e3311de2071cc5feb97fb430bc67f84df3",
 str(RET):"29dc566e90df7c9a592014d1d0cf60dd5da654d570096c96d0c7287bcd5f9072",
 str(SCHED):"6157dce61d990b2d96ef91470a80a973bd5e96339d406bceb9fdc75e020cea42"}
LAM=1.0536573711078283

def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def state(c): return c['model_state_dict']
def rowpos(d):
 return (int(d['query_key_clause_pos']),int(d['other_key_pos'])) if int(d['query_slot'])==0 else (int(d['other_key_pos']),int(d['query_key_clause_pos']))
def loc_loss(att,docs):
 vals=[]
 for i,d in enumerate(docs):
  s0,s1=rowpos(d); x=att[i].clamp_min(1e-12)
  vals.append(-torch.maximum(x[s0-1,0].log()+x[s1-1,1].log(),x[s1-1,0].log()+x[s0-1,1].log()))
 return torch.stack(vals).mean()
def basis(u):
 n=u.numel(); un=u/torch.linalg.vector_norm(u); p=int(torch.argmax(torch.abs(un)).item()); e=torch.zeros_like(un); e[p]=1.0 if un[p]>=0 else -1.0; w=un-e; den=torch.dot(w,w); I=torch.eye(n,dtype=u.dtype,device=u.device); H=I if float(den.detach().abs())<1e-12 else I-2*torch.outer(w,w)/den; return H[:,[i for i in range(n) if i!=p]]
class OrthoLocalizer(nn.Module):
 def __init__(self,u,q,bs,ba):
  super().__init__(); self.u=nn.Parameter(u.detach().clone()); self.q=nn.Parameter(q.detach().clone()); self.bs=nn.Parameter(bs.detach().clone().reshape(())); self.ba=nn.Parameter(ba.detach().clone().reshape(()))
 def forward(self,h):
  v=basis(self.u)@self.q; S=h@self.u+self.bs; R=h@v+self.ba; return torch.stack((S+R,S-R),-1)
def cat(p0,p1,s0,s1):
 T={s0,s1}
 if p0!=p1 and {p0,p1}==T:return 'BOTH_DISTINCT'
 if p0==p1:return 'SLOT_COLLAPSE'
 if sum(p in T for p in (p0,p1))==1:return 'EXACTLY_ONE'
 return 'NEITHER'
def main():
 OUT.mkdir(parents=True,exist_ok=True); hashes={str(p):sha(p) for p in (CK,TRAIN,RET,SCHED)}
 assert all(hashes[k].lower()==v for k,v in EXP_HASH.items()), 'provenance mismatch'
 pref=json.loads(PREF.read_text()); assert pref['status']=='ORTHOGONAL_SHARED_UNBOUNDED_PREFLIGHT_PASS'
 pool=json.loads(TRAIN.read_text()); sched=json.loads(SCHED.read_text()); byq={q['quartet_id']:q for q in pool['quartets']}; assert len(sched['steps_data'])==1000
 ck=torch.load(CK,map_location='cuda',weights_only=False); torch.manual_seed(8380); dev=torch.device('cuda'); model=Treatment13Model().to(dev); model.load_state_dict(state(ck),strict=True)
 W0=model.localizer.scorer.weight.detach().cpu()[0]; W1=model.localizer.scorer.weight.detach().cpu()[1]; b0=model.localizer.scorer.bias.detach().cpu()[0]; b1=model.localizer.scorer.bias.detach().cpu()[1]; Ws=(W0+W1)/2; Wa=(W0-W1)/2; bs=(b0+b1)/2; ba=(b0-b1)/2; B=basis(Ws); q0=B.T@Wa
 model.localizer=OrthoLocalizer(Ws.to(dev),q0.to(dev),bs.to(dev),ba.to(dev)).to(dev); model.train(); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.05); metrics=[]; t0=time.time()
 for step,s in enumerate(sched['steps_data'],1):
  docs=[]
  for qid in s['quartet_ids']: docs.extend(byq[qid]['docs'])
  x=torch.tensor([d['full_document_token_ids'] for d in docs],dtype=torch.long,device=dev); qp=torch.tensor([d['qdp'] for d in docs],device=dev); ap=torch.tensor([d['answer_causal_position'] for d in docs],device=dev); y=torch.tensor([d['target_value_token'] for d in docs],device=dev); opt.zero_grad(set_to_none=True); out,ex=model(x,qp,ap); ix=torch.arange(len(docs),device=dev); la=F.cross_entropy(out[ix,ap],y); li=loc_loss(ex['localization_attention'],docs); total=la+LAM*li; total.backward(); gn=torch.nn.utils.clip_grad_norm_(model.parameters(),2.0); opt.step()
  with torch.no_grad():
   att=ex['localization_attention']; cs=[]
   for j,d in enumerate(docs):
    s0,s1=rowpos(d); p0=int(att[j,:,0].argmax())+1;p1=int(att[j,:,1].argmax())+1;cs.append(cat(p0,p1,s0,s1))
  m={'step':step,'L_total':float(total),'L_answer':float(la),'L_localization':float(li),'answer_accuracy':float((out[ix,ap].argmax(-1)==y).float().mean()),'both_distinct_rate':cs.count('BOTH_DISTINCT')/len(cs),'exactly_one_rate':cs.count('EXACTLY_ONE')/len(cs),'slot_collapse_rate':cs.count('SLOT_COLLAPSE')/len(cs),'neither_rate':cs.count('NEITHER')/len(cs),'grad_norm':float(gn)}; metrics.append(m)
  if step%50==0: print(m,flush=True)
 (OUT/'training_metrics.jsonl').write_text(''.join(json.dumps(m)+'\n' for m in metrics))
 final=OUT/'checkpoints/orthogonal_shared_unbounded/seed_8380/latest.pt'; final.parent.mkdir(parents=True,exist_ok=True); torch.save({'model_state_dict':model.state_dict(),'step':1000,'seed':8380,'objective':'answer_ce_plus_permutation_invariant_localization_nll','localizer':'orthogonal_shared_core_plus_unbounded_antisymmetric'},final); finalsha=sha(final); print({'training_complete':True,'final_checkpoint_sha256':finalsha},flush=True)
 # Single frozen retention exam begins only after the exact checkpoint is saved.
 model.eval(); ret=json.loads(RET.read_text()); docs=[d for q in ret['quartets'] for d in q['docs']]; rows=[]
 with torch.no_grad():
  for st in range(0,len(docs),32):
   ch=docs[st:st+32]; x=torch.tensor([d['full_document_token_ids'] for d in ch],device=dev); qp=torch.tensor([d['qdp'] for d in ch],device=dev); ap=torch.tensor([d['answer_causal_position'] for d in ch],device=dev); out,ex=model(x,qp,ap); ix=torch.arange(len(ch),device=dev)
   for i,d in enumerate(ch):
    v=out[i,ap[i]].float(); pr=torch.softmax(v,-1); t=int(d['target_value_token']); di=int(d['distractor_value_token']); s0,s1=rowpos(d); a=ex['localization_attention'][i]; p0=int(a[:,0].argmax())+1;p1=int(a[:,1].argmax())+1; c=cat(p0,p1,s0,s1); w=ex['row_weights'][i]; sel=int(w.argmax()); sr=qr=None
    if c=='BOTH_DISTINCT':
     smap={0:s0 if p0==s0 else s1,1:s0 if p1==s0 else s1}; sr=smap[sel] in (s0,s1); qsrc=s0 if int(d['query_slot'])==0 else s1; qr=smap[sel]==qsrc
    rows.append({'doc_id':d['doc_id'],'quartet_id':d['quartet_id'],'member':d['member'],'query_slot':int(d['query_slot']),'orientation':int(d['orientation']),'layout_combo':d['layout_combo'],'target':t,'distractor':di,'predicted':int(v.argmax()),'answer_correct':bool(v.argmax()==t),'target_probability':float(pr[t]),'distractor_probability':float(pr[di]),'margin':float(v[t]-v[di]),'pos0':p0,'pos1':p1,'true0':s0,'true1':s1,'localization_category':c,'row_weights':[float(z) for z in w],'selected_slot':sel,'selected_row_correct':sr,'selected_query_row_correct':qr})
 def summary(v):
  return {'documents':len(v),'answer_exact':sum(r['answer_correct'] for r in v),'answer_accuracy':sum(r['answer_correct'] for r in v)/len(v),'both_distinct':sum(r['localization_category']=='BOTH_DISTINCT' for r in v),'exactly_one':sum(r['localization_category']=='EXACTLY_ONE' for r in v),'slot_collapse':sum(r['localization_category']=='SLOT_COLLAPSE' for r in v)}
 bycat=defaultdict(list); bylay=defaultdict(list); byqslot=defaultdict(list); byori=defaultdict(list); byquart=defaultdict(list)
 for r in rows: bycat[r['localization_category']].append(r); bylay[r['layout_combo']].append(r); byqslot[r['query_slot']].append(r); byori[r['orientation']].append(r); byquart[r['quartet_id']].append(r)
 rev=0; revbc=0; changed=0
 for q in byquart.values():
  bm={r['member']:r for r in q}
  for k in ('k0','k1'):
   a,b=bm.get('o1_'+k),bm.get('o2_'+k)
   if a and b: rev+=1; revbc+=a['answer_correct'] and b['answer_correct']; changed+=a['predicted']!=b['predicted']
 qpat=Counter(); stable=0
 for q in byquart.values():
  cs=[r['localization_category'] for r in q]; qpat['4/4 BOTH_DISTINCT' if all(c=='BOTH_DISTINCT' for c in cs) else '4/4 EXACTLY_ONE' if all(c=='EXACTLY_ONE' for c in cs) else '4/4 SLOT_COLLAPSE' if all(c=='SLOT_COLLAPSE' for c in cs) else 'mixed']+=1; stable+=len({(r['pos0'],r['pos1']) for r in q})==1
 bd=[r for r in rows if r['localization_category']=='BOTH_DISTINCT']; ans_bd=sum(r['answer_correct'] for r in bd); sel_bd=sum(bool(r['selected_row_correct']) for r in bd); qr_bd=sum(bool(r['selected_query_row_correct']) for r in bd)
 result={'status':'FINAL_FROZEN_RETENTION_RESULTS','documents':len(rows),'quartets':len(byquart),'overall':summary(rows),'localization_counts':{k:len(v) for k,v in bycat.items()},'by_localization_category':{k:summary(v) for k,v in bycat.items()},'conditional_both_distinct':{'answer_correct':ans_bd,'answer_total':len(bd),'answer_accuracy':ans_bd/len(bd),'selected_row_correct':sel_bd,'selected_row_total':len(bd),'selected_query_row_correct':qr_bd,'selected_query_row_total':len(bd)},'per_query_slot':{str(k):summary(v) for k,v in sorted(byqslot.items())},'per_orientation':{str(k):summary(v) for k,v in sorted(byori.items())},'per_layout':{k:summary(v) for k,v in sorted(bylay.items())},'complete_quartets':sum(all(r['answer_correct'] for r in q) for q in byquart.values()),'quartet_patterns':dict(qpat),'quartet_coordinate_stability':stable,'reversal_pairs':rev,'reversal_both_correct':revbc,'reversal_both_correct_rate':revbc/rev,'prediction_changed_under_reversal':changed,'residual_wrong_localization_destinations':dict(Counter(r['pos0'] for r in rows if r['localization_category']!='BOTH_DISTINCT')+Counter(r['pos1'] for r in rows if r['localization_category']!='BOTH_DISTINCT')),'checkpoint_sha256':finalsha,'starting_checkpoint_sha256':hashes[str(CK)],'train_pool_sha256':hashes[str(TRAIN)],'retention_pool_sha256':hashes[str(RET)],'schedule_sha256':hashes[str(SCHED)],'optimizer_updates':1000,'retention_optimizer_updates':0,'retention_evaluations':1,'model_eval':True,'torch_no_grad':True,'per_document_rows':rows}
 (OUT/'FINAL_FROZEN_RETENTION_RESULTS.json').write_text(json.dumps(result,indent=1)); (OUT/'REPORT.md').write_text('# Orthogonal shared-core + unbounded antisymmetric treatment\n\n'+json.dumps({k:v for k,v in result.items() if k!='per_document_rows'},indent=2)); print(json.dumps({k:v for k,v in result.items() if k!='per_document_rows'},indent=2)); print('ORTHOGONAL_SHARED_UNBOUNDED_AUTHORITATIVE_TREATMENT_COMPLETE')
if __name__=='__main__': main()
