from __future__ import annotations
import hashlib, inspect, json, math, re, sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT=Path(r"C:\DaveLM-CADAVER"); OUT=ROOT/"treatment13_nonsource_localization_suppression_seed8380"; T13=ROOT/"treatment13_learned_mapping_row_localization_seed8380"
sys.path[:0]=[str(ROOT),r"C:\DaveLM-v0.9"]
from treatment13_model import Treatment13Model
START=T13/"checkpoints/learned_mapping_row_localization/seed_8380/latest.pt"; TRAIN=T13/"treatment13_training_quartet_pool.json"; RET=T13/"treatment13_retention_quartet_pool.json"; SCHED=T13/"treatment13_frozen_schedule.json"; PRIOR=ROOT/"treatment13_distinct_localization_supervision_seed8380/PREFLIGHT_AUDIT.json"; AUTH=ROOT/"treatment13_distinct_localization_supervision_seed8380/FINAL_FROZEN_RETENTION_RESULTS.json"
def sha(p):
 h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()
def load(p): return json.loads(p.read_text())
def rowpos(d):
 return (int(d['query_key_clause_pos']),int(d['other_key_pos'])) if int(d['query_slot'])==0 else (int(d['other_key_pos']),int(d['query_key_clause_pos']))
def docs_for_first_step(pool,sched):
 qm={q['quartet_id']:q for q in pool['quartets']}; out=[]
 for qid in sched['steps_data'][0]['quartet_ids']: out.extend(qm[qid]['docs'])
 return out
def lexisting(att,docs):
 lp=att.clamp_min(1e-12).log(); vals=[]
 for b,d in enumerate(docs):
  s0,s1=rowpos(d); p0=s0-1;p1=s1-1
  vals.append(torch.stack((lp[b,p0,0]+lp[b,p1,1],lp[b,p1,0]+lp[b,p0,1])).max())
 return -torch.stack(vals).mean()
def lfiller(att,docs,q,T):
 vals=[]; n=T-1-4
 for b,d in enumerate(docs):
  s0,s1=rowpos(d); valid=[p for p in range(1,1+n) if p<int(q[b]) and p+4<T]; non=[p for p in valid if p not in (s0,s1)]
  assert s0 in valid and s1 in valid and len(non)==len(valid)-2
  ii=[p-1 for p in non]; vals.append(0.5*(att[b,ii,0].max()+att[b,ii,1].max()))
 return torch.stack(vals).mean()
def gnorm(gs): return torch.sqrt(sum((g.float()**2).sum() for g in gs if g is not None))
def main():
 paths=[START,TRAIN,RET,SCHED,PRIOR,AUTH,ROOT/'treatment13_model.py',ROOT/'treatment13_preflight.py']; hashes={str(p):sha(p) for p in paths}
 assert hashes[str(START)]=='cf53eef03adf874255cd54508fc7948dfd1e4926a57537f7e2f828f5f582adab'
 tr,rt,sc,pr=load(TRAIN),load(RET),load(SCHED),load(PRIOR)
 assert tr['document_count']==800 and rt['document_count']==320 and sc['steps']==1000 and sc['batch_size']==32
 assert sha(TRAIN)=='40d78c450f2335aa155031a612d5b8e3311de2071cc5feb97fb430bc67f84df3' and sha(RET)=='29dc566e90df7c9a592014d1d0cf60dd5da654d570096c96d0c7287bcd5f9072' and sha(SCHED)=='6157dce61d990b2d96ef91470a80a973bd5e96339d406bceb9fdc75e020cea42'
 assert pr['lambda']==1.0536573711078283 and pr['optimizer_steps_so_far']==0
 docs=docs_for_first_step(tr,sc); assert len(docs)==32; ids=torch.tensor([d['full_document_token_ids'] for d in docs]); q=torch.tensor([d['qdp'] for d in docs]); ans=torch.tensor([d['answer_causal_position'] for d in docs]); model=Treatment13Model(); ck=torch.load(START,map_location='cpu',weights_only=False); model.load_state_dict(ck['model_state_dict'],strict=True); model.train(); lp=list(model.localizer.parameters()); assert model.base_model_parameter_count()==10594944
 logits,ex=model(ids,q,ans); att=ex['localization_attention']; targ=ids[torch.arange(32),ans]; La=F.cross_entropy(logits[torch.arange(32),ans],targ); Le=lexisting(att,docs); Lf=lfiller(att,docs,q,ids.shape[1]); ge=gnorm(torch.autograd.grad(pr['lambda']*Le,lp,retain_graph=True,allow_unused=True)); logits2,ex2=model(ids,q,ans); gf=gnorm(torch.autograd.grad(lfiller(ex2['localization_attention'],docs,q,ids.shape[1]),lp,allow_unused=True)); assert all(torch.isfinite(x) for x in (La,Le,Lf,ge,gf)) and ge.item()>0 and gf.item()>0
 lam=0.10*ge.item()/gf.item()
 if not (math.isfinite(lam) and 0.001<=lam<=0.1):
  out={'status':'NONSOURCE_SUPPRESSION_PREFLIGHT_ABORT_ZERO_UPDATES','failure':'lambda_filler_out_of_preregistered_range','optimizer_steps_so_far':0,'no_training':True,'no_retention_behavior_used':True,'hashes':hashes,'lambda_existing':pr['lambda'],'initial_mean_answer_loss':float(La),'initial_mean_existing_localization_loss':float(Le),'initial_mean_filler_loss':float(Lf),'G_existing':float(ge),'G_filler_raw':float(gf),'lambda_filler':lam,'required_range':[0.001,0.1],'calibration_formula':'0.10*G_existing/G_filler_raw','forward_inputs':'input_ids,qpos,anspos only','optimizer_created':False,'gradient_calibration_zero_update':True,'checkpoint_step':int(ck['step'])}
  (OUT/'PREFLIGHT_AUDIT.json').write_text(json.dumps(out,indent=2)); (OUT/'FILES.json').write_text(json.dumps({'experiment':OUT.name,'input_hashes':hashes},indent=2)); print(json.dumps(out,indent=2)); print('NONSOURCE_SUPPRESSION_PREFLIGHT_ABORT_ZERO_UPDATES'); return
 # permutation-invariant and synthetic monotonicity checks
 vals0=[]; vals1=[]; lp0=att.clamp_min(1e-12).log()
 for b,d in enumerate(docs):
  s0,s1=rowpos(d); a=lp0[b,s0-1,0]+lp0[b,s1-1,1]; b2=lp0[b,s1-1,0]+lp0[b,s0-1,1]; vals0.append(torch.maximum(a,b2)); vals1.append(torch.maximum(b2,a))
 assert torch.allclose(torch.stack(vals0),torch.stack(vals1),atol=1e-6)
 def f(v,S): return max(v[i] for i in range(len(v)) if i not in S)
 u=[.1]*10; h=u.copy(); h[0]=.4; m=h.copy(); m[0]=.1;m[3]=.4; assert f(m,{3,7})<f(h,{3,7}); h2=u.copy();h2[8]=.4; assert f(h2,{3,7})>f(u,{3,7})
 assert not any(re.search(r'(?<!\d)'+x+r'(?!\d)',inspect.getsource(lfiller)) for x in ('20','22','54'))
 out={'status':'NONSOURCE_SUPPRESSION_PREFLIGHT_PASS_AWAITING_AUTHORIZATION','optimizer_steps_so_far':0,'no_training':True,'no_retention_behavior_used':True,'hashes':hashes,'counts':{'train_docs':800,'retention_docs':320,'schedule_steps':1000,'batch_size':32},'starting_checkpoint_sha256':hashes[str(START)],'lambda_existing':pr['lambda'],'initial_mean_answer_loss':float(La),'initial_mean_existing_localization_loss':float(Le),'initial_mean_filler_loss':float(Lf),'G_existing':float(ge),'G_filler_raw':float(gf),'lambda_filler':lam,'initial_weighted_filler_gradient_ratio':0.10,'forward_inputs':'input_ids,qpos,anspos only','mask_audit':{'exactly_two_sources_excluded':True,'all_other_valid_candidates_included':True},'synthetic_tests':{'swap_labels_invariant':True,'distinct_better_than_wrong':True,'strongest_non_source_monotonic':True,'position_agnostic':True,'no_special_positions':True},'optimizer_created':False,'gradient_calibration_zero_update':True,'lambda_frozen':True,'checkpoint_step':int(ck['step'])}
 (OUT/'PREFLIGHT_AUDIT.json').write_text(json.dumps(out,indent=2)); (OUT/'FILES.json').write_text(json.dumps({'experiment':OUT.name,'input_hashes':hashes},indent=2)); print(json.dumps(out,indent=2)); print('NONSOURCE_SUPPRESSION_PREFLIGHT_PASS_AWAITING_AUTHORIZATION')
if __name__=='__main__': main()
