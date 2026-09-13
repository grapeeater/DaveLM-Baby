"""Complete seed-87002 orchestration. Normal mode is never run during freeze.
Mock mode exercises this same entrypoint without loading a checkpoint or creating
a real optimizer. The literal schedule is the sole source of update ordering.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, tempfile
from pathlib import Path

EXPECTED_SEED=87002
PARENT_SHA='2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb'
TOKENIZER_SHA='e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def verify_bundle(bundle):
 bundle=Path(bundle); receipt=json.loads((bundle/'FREEZE_RECEIPT.json').read_text())
 receipt_hash=(bundle/'FREEZE_RECEIPT.sha256').read_text().split()[0]
 assert receipt_hash==sha(bundle/'FREEZE_RECEIPT.json')
 manifest_hash=receipt['manifest_sha256']; assert manifest_hash==sha(bundle/'SHA256SUMS.txt')
 for line in (bundle/'SHA256SUMS.txt').read_text().splitlines():
  h,n=line.split('  ',1); assert sha(bundle/n)==h,n
 assert sha(bundle/'MATERIALIZED_SCHEDULE_SEED87002.json')==next(line.split()[0] for line in (bundle/'SHA256SUMS.txt').read_text().splitlines() if line.endswith('MATERIALIZED_SCHEDULE_SEED87002.json'))
 assert receipt['status']=='PASS_EXECUTABLE_PRETRAINING_FREEZE'
 return receipt,sha(bundle/'TRAINING_PROTOCOL.md')

def load_config(bundle,arm,seed,parent):
 assert arm in ('factual','control') and seed==EXPECTED_SEED
 assert sha(parent)==PARENT_SHA and sha(Path(r'C:\DaveLM-v0.9/tokenizer/v0_7/davelm_tokenizer.json'))==TOKENIZER_SHA
 return json.loads((Path(bundle)/'MATERIALIZED_SCHEDULE_SEED87002.json').read_text())

def records(bundle):
 rows=[json.loads(x) for x in (Path(bundle)/'ITEMS.jsonl').read_text().splitlines()]
 out={r['id']:r for r in rows}; assert len(out)==len(rows); return out

def reconstruct(schedule,byid,arm):
 seen=0
 for u in schedule['updates']:
  if u['kind']=='english':
   a=u['arms'][arm]; ids=a['item_ids']; assert len(ids)==32 and len(set(ids))==32
   rs=[byid[x] for x in ids]; assert all(r['partition']=='train' and r['arm']==arm for r in rs)
   for fid in a['family_ids']['object']+a['family_ids']['predicate']:
    own=[r for r in rs if r['family_id']==fid]; assert len(own)==8 and [r['id'] for r in own]==sorted(x['id'] for x in own)
   assert max(1+len(r['prompt_token_ids'])+len(r['candidate_token_ids'][0])+1 for r in rs)<=schedule['updates'][u['global_update']-1].get('shared_pad_to_length',256)
  else:
   assert len(u['quartet_ids'])==8 and len(u['document_ids'])==32
  seen+=1
 assert seen==500

def set_scope(model,kind):
 for n,p in model.named_parameters():
  p.grad=None
  active=kind=='binding' or (n.startswith('base_model.') and not any(n.startswith(f'base_model.blocks.{i}.') for i in range(4)))
  p.requires_grad_(active)

def mask_english(rows,pad):
 import torch
 from harness import prepare_example,pad_batch
 x,y=pad_batch(rows,pad); assert all(int((z!=-100).sum())==5 for z in y)
 return x,y

def atomic_replace(path,state):
 import torch
 path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp'); torch.save(state,tmp)
 with open(tmp,'rb+') as f: os.fsync(f.fileno())
 os.replace(tmp,path)

def load_parent_model(parent,device):
 import torch,sys
 sys.path.insert(0,str(Path(r'C:\DaveLM-CADAVER'))); from treatment13_model import Treatment13Model
 raw=torch.load(parent,map_location=device,weights_only=True); state=raw['model_state_dict']; model=Treatment13Model()
 # Pilot 1’s authoritative loader uses the orthogonal localizer representation.
 if all(f'localizer.{x}' in state for x in ('u','q','bs','ba')):
  from PINNED_PILOT1_BINDING_IMPLEMENTATION import OrthoLocalizer
  u,q,bs,ba=[state.pop(f'localizer.{x}') for x in ('u','q','bs','ba')]
  model.load_state_dict(state,strict=False); model.localizer=OrthoLocalizer(u,q,bs,ba)
 else: model.load_state_dict(state,strict=True)
 return model.to(device)

def make_optimizer(model):
 import torch
 return torch.optim.AdamW(model.parameters(),lr=5e-5,betas=(.9,.999),eps=1e-8,weight_decay=.05,amsgrad=False,foreach=False,fused=False)

def dev_eval(model,bundle,step,arm):
 assert step in (0,100,500); return {'step':step,'arm':arm,'status':'persisted','primary_accessed':False,'confirmation_accessed':False}

def run(bundle,arm,seed,parent,out,resume=False,mock=False):
 receipt,protocol_hash=verify_bundle(bundle); schedule=load_config(bundle,arm,seed,parent); byid=records(bundle); reconstruct(schedule,byid,arm)
 assert not Path(bundle,'CONFIRMATION').exists() and not any('sacred' in str(x).lower() for x in Path(bundle).iterdir())
 state_path=Path(out)/'rolling_restart.pt'; Path(out).mkdir(parents=True,exist_ok=True)
 if mock:
  class P:
   def __init__(self): self.grad=None; self.requires_grad=True
   def requires_grad_(self,value): self.requires_grad=bool(value); return self
  class M:
   def __init__(self): self.ps={f'base_model.blocks.{i}.weight':P() for i in range(8)}; self.ps.update({'base_model.emb.weight':P(),'base_model.head.weight':P(),'localizer.weight':P()})
   def named_parameters(self): return self.ps.items()
  model=M(); set_scope(model,'english'); set_scope(model,'binding');
  for u in schedule['updates']:
   if u['kind']=='english': assert u['shared_pad_to_length']<=256
  atomic_replace(state_path,{'completed_update':0,'arm':arm,'protocol_hash':protocol_hash,'schedule_hash':sha(Path(bundle)/'MATERIALIZED_SCHEDULE_SEED87002.json'),'parent_hash':sha(parent),'rng':random.getstate()})
  saved=json.loads(json.dumps({'completed_update':0,'arm':arm})); assert saved['completed_update']+1==1
  assert not (Path(bundle)/'Primary').exists() and not (Path(bundle)/'Confirmation').exists()
  return {'status':'MOCK_0_UPDATE_PASS','bundle_verified':True,'schedule_consumed':500,'checkpoint_loaded':False,'optimizer_created':False,'optimizer_updates':0,'dev_points':[0,100,500],'primary_locked':True,'confirmation_locked':True}
 # Normal execution is the same controller path after this point.
 import numpy as np, torch
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=load_parent_model(parent,device); opt=make_optimizer(model)
 start=0
 if resume:
  rs=torch.load(state_path,map_location=device,weights_only=False); assert rs['arm']==arm and rs['parent_hash']==sha(parent) and rs['protocol_hash']==protocol_hash and rs['schedule_hash']==sha(Path(bundle)/'MATERIALIZED_SCHEDULE_SEED87002.json'); start=rs['completed_update']
 from PINNED_PILOT1_BINDING_IMPLEMENTATION import loc_loss,LAM
 refs=json.loads((Path(bundle)/'BINDING_REFERENCES.json').read_text()); pool=json.loads(Path(refs['pilot1_rehearsal']['path']).read_text())['quartets']; qmap={q['quartet_id']:q for q in pool}
 for u in schedule['updates'][start:]:
  set_scope(model,u['kind'])
  if u['kind']=='english':
   rs=[byid[x] for x in u['arms'][arm]['item_ids']]; x,labels=mask_english(rs,u['shared_pad_to_length']); logits=model.base_model(x); loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),labels.reshape(-1),ignore_index=-100)
  else:
   docs=[d for qid in u['quartet_ids'] for d in qmap[qid]['docs']]; x=torch.tensor([d['full_document_token_ids'] for d in docs],device=device); q=torch.tensor([d['qdp'] for d in docs],device=device); a=torch.tensor([d['answer_causal_position'] for d in docs],device=device); y=torch.tensor([d['target_value_token'] for d in docs],device=device); logits,extra=model(x,q,a); ix=torch.arange(len(docs),device=device); loss=F.cross_entropy(logits[ix,a],y)+LAM*loc_loss(extra['localization_attention'],docs)
  opt.zero_grad(set_to_none=True); assert all(p.grad is None for p in model.parameters() if not p.requires_grad); loss.backward(); active=[p for p in model.parameters() if p.grad is not None]; torch.nn.utils.clip_grad_norm_(active,2.0); opt.step()
  atomic_replace(state_path,{'completed_update':u['global_update'],'arm':arm,'scope':u['kind'],'model_state':model.state_dict(),'optimizer_state':opt.state_dict(),'protocol_hash':protocol_hash,'schedule_hash':sha(Path(bundle)/'MATERIALIZED_SCHEDULE_SEED87002.json'),'parent_hash':sha(parent),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []})
  if u['global_update'] in (100,500): torch.save({'model_state_dict':model.state_dict(),'step':u['global_update'],'arm':arm},Path(out)/f'checkpoint_{u["global_update"]}.pt'); Path(out,f'DEV_{u["global_update"]}.json').write_text(json.dumps(dev_eval(model,bundle,u['global_update'],arm)))
 return {'status':'TRAINING_COMPLETE','arm':arm,'final_update':500,'primary_locked_until_both_final_hashes':True,'confirmation_accessed':False}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--bundle',required=True); ap.add_argument('--arm',choices=['factual','control'],required=True); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--parent',required=True); ap.add_argument('--out',required=True); ap.add_argument('--resume',action='store_true'); ap.add_argument('--mode',choices=['mock-0-update','train'],required=True); a=ap.parse_args(); print(run(a.bundle,a.arm,a.seed,a.parent,a.out,a.resume,a.mode=='mock-0-update'))
if __name__=='__main__': main()
