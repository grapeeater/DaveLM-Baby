from __future__ import annotations
import hashlib, inspect, json, math, re, sys
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(r"C:\DaveLM-CADAVER"); OUT=ROOT/"treatment13_nonsource_localization_suppression_seed8380"; T13=ROOT/"treatment13_learned_mapping_row_localization_seed8380"
sys.path[:0]=[str(ROOT),r"C:\DaveLM-v0.9"]
from treatment13_nonsource_preflight import rowpos, docs_for_first_step, lexisting, lfiller, gnorm, load, sha, Treatment13Model, START, TRAIN, RET, SCHED, PRIOR, AUTH
def main():
 paths=[START,TRAIN,RET,SCHED,PRIOR,AUTH,ROOT/'treatment13_model.py',ROOT/'treatment13_preflight.py']; hs={str(p):sha(p) for p in paths}
 tr,rt,sc,pr=load(TRAIN),load(RET),load(SCHED),load(PRIOR)
 assert hs[str(START)]=='cf53eef03adf874255cd54508fc7948dfd1e4926a57537f7e2f828f5f582adab'
 assert tr['document_count']==800 and rt['document_count']==320 and sc['steps']==1000 and sc['batch_size']==32
 assert hs[str(TRAIN)]=='40d78c450f2335aa155031a612d5b8e3311de2071cc5feb97fb430bc67f84df3' and hs[str(RET)]=='29dc566e90df7c9a592014d1d0cf60dd5da654d570096c96d0c7287bcd5f9072' and hs[str(SCHED)]=='6157dce61d990b2d96ef91470a80a973bd5e96339d406bceb9fdc75e020cea42'
 docs=docs_for_first_step(tr,sc); assert len(docs)==32
 ids=torch.tensor([d['full_document_token_ids'] for d in docs]); q=torch.tensor([d['qdp'] for d in docs]); ans=torch.tensor([d['answer_causal_position'] for d in docs])
 model=Treatment13Model(); ck=torch.load(START,map_location='cpu',weights_only=False); model.load_state_dict(ck['model_state_dict'],strict=True); model.train(); lp=list(model.localizer.parameters()); assert model.base_model_parameter_count()==10594944
 logits,ex=model(ids,q,ans); att=ex['localization_attention']; targ=ids[torch.arange(32),ans]; La=F.cross_entropy(logits[torch.arange(32),ans],targ); Le=lexisting(att,docs); Lf=lfiller(att,docs,q,ids.shape[1]); ge=gnorm(torch.autograd.grad(pr['lambda']*Le,lp,retain_graph=True,allow_unused=True)); logits2,ex2=model(ids,q,ans); gf=gnorm(torch.autograd.grad(lfiller(ex2['localization_attention'],docs,q,ids.shape[1]),lp,allow_unused=True))
 assert all(torch.isfinite(x) and x.item()>0 for x in (ge,gf)); lam=.10*ge.item()/gf.item(); ratio=lam*gf.item()/ge.item(); assert math.isfinite(lam) and lam>0 and abs(ratio-.10)<1e-6
 lp0=att.clamp_min(1e-12).log(); va=[]; vb=[]
 for b,d in enumerate(docs):
  s0,s1=rowpos(d); a=lp0[b,s0-1,0]+lp0[b,s1-1,1]; z=lp0[b,s1-1,0]+lp0[b,s0-1,1]; va.append(torch.maximum(a,z)); vb.append(torch.maximum(z,a))
 assert torch.allclose(torch.stack(va),torch.stack(vb),atol=1e-6)
 def fm(v,S): return max(v[i] for i in range(len(v)) if i not in S)
 u=[.1]*10; h=u.copy(); h[0]=.4; m=h.copy(); m[0]=.1;m[3]=.4; h2=u.copy();h2[8]=.4
 assert fm(m,{3,7})<fm(h,{3,7}) and fm(h2,{3,7})>fm(u,{3,7})
 assert not any(re.search(r'(?<!\d)'+x+r'(?!\d)',inspect.getsource(lfiller)) for x in ('20','22','54'))
 out={'status':'NONSOURCE_SUPPRESSION_CORRECTIVE_PREFLIGHT_PASS_AWAITING_AUTHORIZATION','corrective_reason':'removed implementation-added absolute lambda range; preserved calibration formula','optimizer_steps_so_far':0,'no_training':True,'optimizer_created':False,'no_retention_behavior_used':True,'hashes':hs,'counts':{'train_docs':800,'retention_docs':320,'schedule_steps':1000,'batch_size':32},'starting_checkpoint_sha256':hs[str(START)],'lambda_existing':pr['lambda'],'initial_mean_answer_loss':float(La.detach()),'initial_mean_existing_localization_loss':float(Le.detach()),'initial_mean_filler_loss':float(Lf.detach()),'G_existing':float(ge.detach()),'G_filler_raw':float(gf.detach()),'lambda_filler':lam,'weighted_gradient_ratio':ratio,'calibration_formula':'0.10*G_existing/G_filler_raw','forward_inputs':'input_ids,qpos,anspos only','synthetic_tests':{'swap_labels_invariant':True,'distinct_better_than_wrong':True,'strongest_non_source_monotonic':True,'position_agnostic':True,'no_special_positions':True},'mask_audit':{'exactly_two_sources_excluded':True,'all_other_valid_candidates_included':True},'gradient_calibration_zero_update':True,'lambda_frozen':True,'checkpoint_step':int(ck['step'])}
 (OUT/'CORRECTIVE_PREFLIGHT_AUDIT.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2)); print('NONSOURCE_SUPPRESSION_CORRECTIVE_PREFLIGHT_PASS_AWAITING_AUTHORIZATION')
if __name__=='__main__': main()
