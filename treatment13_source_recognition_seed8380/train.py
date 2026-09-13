import json,hashlib,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(r'C:\DaveLM-CADAVER'); OUT=ROOT/'treatment13_source_recognition_seed8380'; SRC=ROOT/'treatment13_learned_mapping_row_localization_seed8380'; START=SRC/'checkpoints/learned_mapping_row_localization/seed_8380/latest.pt'; TRAIN=SRC/'treatment13_training_quartet_pool.json'; SCHED=SRC/'treatment13_frozen_schedule.json'; CK=OUT/'checkpoints/source_recognition/seed_8380/latest.pt'
sys.path[:0]=[str(ROOT),r'C:\DaveLM-v0.9']; from treatment13_model import Treatment13Model
def rowpos(d): return (int(d['query_key_clause_pos']),int(d['other_key_pos'])) if int(d['query_slot'])==0 else (int(d['other_key_pos']),int(d['query_key_clause_pos']))
def hard(A,s0,s1):
 lp=A.clamp_min(1e-12).log(); return -torch.maximum(lp[0,s0]+lp[1,s1],lp[0,s1]+lp[1,s0])
def sr(A,s0,s1): return -0.5*(torch.log(A[0,s0]+A[0,s1])+torch.log(A[1,s0]+A[1,s1]))
def main():
 p=json.loads((Path(r'C:\DaveLM-CADAVER\forensics\SOURCE_RECOGNITION_PREFLIGHT_SEED8380.json')).read_text()); le=1.0536573711078283; ls=float(p['lambda_src_rec']); pool=json.loads(TRAIN.read_text()); sch=json.loads(SCHED.read_text()); qm={q['quartet_id']:q for q in pool['quartets']}; torch.manual_seed(8380); m=Treatment13Model().to('cuda'); c=torch.load(START,map_location='cuda',weights_only=False); m.load_state_dict(c['model_state_dict']); m.train(); opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=.05); mets=[]; t=time.time()
 for step,s in enumerate(sch['steps_data'],1):
  docs=[]
  for q in s['quartet_ids']: docs.extend(qm[q]['docs'])
  x=torch.tensor([d['full_document_token_ids'] for d in docs],device='cuda'); q=torch.tensor([d['qdp'] for d in docs],device='cuda'); a=torch.tensor([d['answer_causal_position'] for d in docs],device='cuda'); y=torch.tensor([d['target_value_token'] for d in docs],device='cuda'); opt.zero_grad(set_to_none=True); logits,e=m(x,q,a); ix=torch.arange(len(docs),device='cuda'); La=F.cross_entropy(logits[ix,a],y); A=e['localization_attention']; Le=torch.stack([hard(A[i].T,*[z-1 for z in rowpos(d)]) for i,d in enumerate(docs)]).mean(); Ls=torch.stack([sr(A[i].T,*[z-1 for z in rowpos(d)]) for i,d in enumerate(docs)]).mean(); total=La+le*Le+ls*Ls; total.backward(); gn=torch.nn.utils.clip_grad_norm_(m.parameters(),2.0); opt.step(); mets.append({'step':step,'L_total':float(total),'L_answer':float(La),'L_existing':float(Le),'L_src_rec':float(Ls),'answer_accuracy':float((logits[ix,a].argmax(-1)==y).float().mean()),'grad_norm':float(gn)})
  if step%50==0: print(mets[-1],flush=True)
 CK.parent.mkdir(parents=True,exist_ok=True); torch.save({'model_state_dict':m.state_dict(),'step':1000,'seed':8380,'lambda_existing':le,'lambda_src_rec':ls,'objective':'answer_ce_plus_existing_localization_plus_source_set_recognition'},CK); out={'status':'TRAINED','steps':1000,'checkpoint_sha256':hashlib.sha256(CK.read_bytes()).hexdigest(),'runtime_seconds':time.time()-t,'final_step':mets[-1]}; (OUT/'training_metrics.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in mets)); (OUT/'TRAINING_RESULT.json').write_text(json.dumps(out,indent=2)); print(out)
if __name__=='__main__': main()
