import hashlib, json, math, sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(r'C:\DaveLM-CADAVER')
T13 = ROOT / 'treatment13_learned_mapping_row_localization_seed8380'
CH = ROOT / 'treatment13_distinct_localization_supervision_seed8380'
DEST = ROOT / 'forensics' / 'SOURCE_RECOGNITION_PREFLIGHT_SEED8380.json'
sys.path[:0] = [str(ROOT), r'C:\DaveLM-v0.9']
from treatment13_model import Treatment13Model
START = T13 / 'checkpoints/learned_mapping_row_localization/seed_8380/latest.pt'
TRAIN = T13 / 'treatment13_training_quartet_pool.json'; SCHED = T13 / 'treatment13_frozen_schedule.json'; RET = T13 / 'treatment13_retention_quartet_pool.json'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rowpos(d): return (int(d['query_key_clause_pos']),int(d['other_key_pos'])) if int(d['query_slot'])==0 else (int(d['other_key_pos']),int(d['query_key_clause_pos']))
def hard(A,s0,s1):
 lp=A.clamp_min(1e-12).log(); u1=lp[0,s0]+lp[1,s1]; u2=lp[0,s1]+lp[1,s0]; return -torch.maximum(u1,u2),int(u1>=u2)
def sr(A,s0,s1): return -0.5*(torch.log(A[0,s0]+A[0,s1])+torch.log(A[1,s0]+A[1,s1]))
def gn(gs): return torch.sqrt(sum((g.float()**2).sum() for g in gs if g is not None))
def main():
 hs={str(p):sha(p) for p in [START,TRAIN,SCHED,RET,CH/'PREFLIGHT_AUDIT.json']}; assert hs[str(START)]=='cf53eef03adf874255cd54508fc7948dfd1e4926a57537f7e2f828f5f582adab' and hs[str(TRAIN)]=='40d78c450f2335aa155031a612d5b8e3311de2071cc5feb97fb430bc67f84df3' and hs[str(SCHED)]=='6157dce61d990b2d96ef91470a80a973bd5e96339d406bceb9fdc75e020cea42'
 pool=json.loads(TRAIN.read_text()); sch=json.loads(SCHED.read_text()); qm={q['quartet_id']:q for q in pool['quartets']}; docs=[]
 for qid in sch['steps_data'][0]['quartet_ids']: docs.extend(qm[qid]['docs'])
 assert len(docs)==32 and sch['steps']==1000
 torch.manual_seed(8380); m=Treatment13Model(); ck=torch.load(START,map_location='cpu',weights_only=False); m.load_state_dict(ck['model_state_dict'],strict=True); m.eval(); pars=list(m.localizer.parameters())
 ids=torch.tensor([d['full_document_token_ids'] for d in docs]); q=torch.tensor([d['qdp'] for d in docs]); ans=torch.tensor([d['answer_causal_position'] for d in docs]); logits,e=m(ids,q,ans); A=e['localization_attention']; ix=torch.arange(32)
 La=F.cross_entropy(logits[ix,ans],ids[ix,ans]); Le=torch.stack([hard(A[b].transpose(0,1),*(x-1 for x in rowpos(d)))[0] for b,d in enumerate(docs)]).mean(); Ls=torch.stack([sr(A[b].transpose(0,1),*(x-1 for x in rowpos(d))) for b,d in enumerate(docs)]).mean(); w=1.0536573711078283
 g1=torch.autograd.grad(w*Le,pars,retain_graph=True,allow_unused=True); ge=gn(g1); logits2,e2=m(ids,q,ans); Ls2=torch.stack([sr(e2['localization_attention'][b].transpose(0,1),*(x-1 for x in rowpos(d))) for b,d in enumerate(docs)]).mean(); g2=torch.autograd.grad(Ls2,pars,allow_unused=True); gs=gn(g2); lam=.10*ge.item()/gs.item(); ratio=lam*gs.item()/ge.item(); cos=float(sum((a*b).sum() for a,b in zip(g1,g2))/(ge*gs)); assert all(torch.isfinite(x) and x.item()>0 for x in (ge,gs)) and math.isfinite(lam) and lam>0 and abs(ratio-.10)<1e-6
 states={'A_early_moderate':[[0.,3.,1.,2.,-3.],[0.,1.,3.,2.,-3.]],'B_sharp_wandering':[[-8.,8.,-8.,-8.,-8.],[-8.,-8.,-8.,-8.,8.]],'C_duplicate':[[-8.,8.,-8.,-8.,-8.],[-8.,8.,-8.,-8.,-8.]],'D_diffuse':[[-8.,8.,-8.,-8.,-8.],[-8.,-8.,2.,-8.,2.]],'E_swapped':[[-8.,-8.,8.,-8.,-8.],[-8.,8.,-8.,-8.,-8.]]}; synth={}
 for name,zv in states.items():
  z=torch.tensor(zv,dtype=torch.float64,requires_grad=True); P=torch.softmax(z,1); lh,ass=hard(P,1,2); ls=sr(P,1,2); synth[name]={'hard_loss':float(lh),'src_rec_loss':float(ls),'hard_grad':torch.autograd.grad(lh,z,retain_graph=True)[0].tolist(),'src_rec_grad':torch.autograd.grad(ls,z)[0].tolist(),'selected_assignment':ass}
 result={'status':'SOURCE_RECOGNITION_CORRECTIVE_PREFLIGHT_PASS_AWAITING_AUTHORIZATION','checkpoint_sha256':hs[str(START)],'train_pool_sha256':hs[str(TRAIN)],'schedule_sha256':hs[str(SCHED)],'retention_pool_sha256':hs[str(RET)],'optimizer_steps_so_far':0,'optimizer_created':False,'retention_evaluated':False,'forward_inputs':'input_ids,qpos,anspos only','G_existing':float(ge),'G_src':float(gs),'lambda_src_rec':lam,'weighted_gradient_ratio':ratio,'gradient_cosine_existing_vs_source_rec':cos,'initial_mean_answer_loss':float(La.detach()),'initial_mean_existing_loss':float(Le.detach()),'initial_mean_source_rec_loss':float(Ls.detach()),'synthetic_states':synth,'auxiliary_distinct_signal':True,'no_unauthorized_lambda_bounds':True}
 DEST.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2)); print('SOURCE_RECOGNITION_CORRECTIVE_PREFLIGHT_PASS_AWAITING_AUTHORIZATION')
if __name__=='__main__': main()
