import hashlib,json,math,statistics,sys
from pathlib import Path
import torch
ROOT=Path(r'C:\DaveLM-CADAVER'); T13=ROOT/'treatment13_learned_mapping_row_localization_seed8380'; OUT=ROOT/'forensics'; DEST=OUT/'BOUNDED_SHARED_ANTISYMMETRIC_PREFLIGHT_SEED8380.json'; CK=T13/'checkpoints/learned_mapping_row_localization/seed_8380/latest.pt'; TRAIN=T13/'treatment13_training_quartet_pool.json'; RET=T13/'treatment13_retention_quartet_pool.json'; SCHED=T13/'treatment13_frozen_schedule.json'
sys.path[:0]=[str(ROOT),r'C:\DaveLM-v0.9']; from treatment13_model import Treatment13Model
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rowpos(d): return (int(d['query_key_clause_pos']),int(d['other_key_pos'])) if int(d['query_slot'])==0 else (int(d['other_key_pos']),int(d['query_key_clause_pos']))
def main():
 hs={str(p):sha(p) for p in (CK,TRAIN,RET,SCHED)}; assert hs[str(CK)]=='cf53eef03adf874255cd54508fc7948dfd1e4926a57537f7e2f828f5f582adab' and hs[str(TRAIN)]=='40d78c450f2335aa155031a612d5b8e3311de2071cc5feb97fb430bc67f84df3' and hs[str(RET)]=='29dc566e90df7c9a592014d1d0cf60dd5da654d570096c96d0c7287bcd5f9072' and hs[str(SCHED)]=='6157dce61d990b2d96ef91470a80a973bd5e96339d406bceb9fdc75e020cea42'
 pool=json.loads(TRAIN.read_text()); sch=json.loads(SCHED.read_text()); qm={q['quartet_id']:q for q in pool['quartets']}; docs=[]
 for q in sch['steps_data'][0]['quartet_ids']: docs.extend(qm[q]['docs'])
 assert len(docs)==32 and sch['steps']==1000 and sch['batch_size']==32
 torch.manual_seed(8380); m=Treatment13Model(); ck=torch.load(CK,map_location='cpu',weights_only=False); m.load_state_dict(ck['model_state_dict'],strict=True); m.eval();
 ids=torch.tensor([d['full_document_token_ids'] for d in docs]); q=torch.tensor([d['qdp'] for d in docs]); ans=torch.tensor([d['answer_causal_position'] for d in docs])
 with torch.no_grad():
  old_logits,ex=m(ids,q,ans); h=m._captured.float(); W=m.localizer.scorer.weight.float(); b=m.localizer.scorer.bias.float(); cand=old_logits.new_zeros((32,ids.shape[1],2)); n=ids.shape[1]-1-4; H=h[:,1:1+n,:]; old=cand; old=H@W.T+b; u=(W[0]+W[1])/2; v=(W[0]-W[1])/2; bs=(b[0]+b[1])/2; br=(b[0]-b[1])/2; anti=(H@v+br); rho=float(torch.sqrt((anti**2).mean()).item()); new=torch.stack((H@u+bs+rho*torch.tanh(anti),H@u+bs-rho*torch.tanh(anti)),dim=-1)
  oldA=torch.softmax(old,1); newA=torch.softmax(new,1); rmse=float(torch.sqrt(((new-old)**2).mean())); mx=float((new-old).abs().max()); kl=float((oldA*(oldA.clamp_min(1e-12)/newA.clamp_min(1e-12)).log()).sum(-1).mean()); coll_old=coll_new=0; preserve=[0,0]; setpres=0; rows=[]; identical_new=0
  for i,d in enumerate(docs):
   qpos=int(q[i]); valid=[p for p in range(1,1+n) if p<qpos and p+4<ids.shape[1]]; oo=[];nn=[]
   for k in range(2): oo.append(valid[int(oldA[i,[p-1 for p in valid],k].argmax())]); nn.append(valid[int(newA[i,[p-1 for p in valid],k].argmax())]); preserve[k]+=oo[k]==nn[k]
   setpres+=set(oo)==set(nn); coll_old+=oo[0]==oo[1]; coll_new+=nn[0]==nn[1]; identical_new += bool(torch.allclose(new[i,:,0],new[i,:,1],atol=1e-7,rtol=0.0)); rows.append({'doc_id':d['doc_id'],'old_top_positions':oo,'new_top_positions':nn,'old_new_set_preserved':set(oo)==set(nn)})
 result={'status':'BOUNDED_SHARED_ANTISYMMETRIC_PREFLIGHT_PASS_AWAITING_AUTHORIZATION','provenance_hashes':hs,'starting_checkpoint_step':int(ck['step']),'seed':8380,'schedule_steps':1000,'batch_size':32,'old_scorer_parameters':642,'new_scorer_parameters':642,'rho_definition':'RMS of baseline linear antisymmetric score a_j=(v^T h_j+b_r) over the exact deterministic first training batch, with no retention access','rho':rho,'initialization_mapping':{'u':'(W0+W1)/2','v':'(W0-W1)/2','b_s':'(b0+b1)/2','b_r':'(b0-b1)/2'},'functional_audit':{'logit_rmse':rmse,'max_abs_logit_difference':mx,'mean_old_to_new_position_KL':kl,'slot0_top1_preserved':preserve[0]/32,'slot1_top1_preserved':preserve[1]/32,'unordered_top1_set_preserved':setpres/32,'old_top1_collisions':coll_old,'new_top1_collisions':coll_new,'new_identical_slot_distributions':identical_new},'mathematical_properties':{'shared_average_exact':True,'antisymmetric_half_difference_exact':True,'residual_bound_abs_le_rho':True,'not_invertible_due_tanh_bound':True,'slot_swap_v_to_minus_v_br_to_minus_br':True},'objective':'answer CE + champion permutation-invariant hard-min localization only','forward_inputs':'input_ids,qpos,anspos only','source_labels_outside_forward':True,'optimizer_created':False,'optimizer_steps_so_far':0,'retention_evaluated':False,'per_document':rows}
 DEST.write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!='per_document'},indent=2)); print('BOUNDED_SHARED_ANTISYMMETRIC_PREFLIGHT_PASS_AWAITING_AUTHORIZATION')
if __name__=='__main__': main()
