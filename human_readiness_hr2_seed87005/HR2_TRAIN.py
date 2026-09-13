import argparse,json,random,hashlib
from pathlib import Path
import torch
import torch.nn.functional as F
import sys
ROOT=Path(r'C:\\DaveLM-CADAVER'); sys.path.insert(0,str(ROOT)); sys.path.insert(0,r'C:\\DaveLM-v0.9')
from treatment13_model import Treatment13Model
R=ROOT/'human_readiness_hr2_seed87005'; DEVICE='cuda'
import torch.nn as nn
def basis(u):
 un=u/torch.linalg.vector_norm(u); p=int(torch.argmax(torch.abs(un))); e=torch.zeros_like(un); e[p]=1. if un[p]>=0 else -1.; w=un-e; H=torch.eye(un.numel(),device=u.device,dtype=u.dtype)-2*torch.outer(w,w)/torch.dot(w,w); return H[:,[i for i in range(un.numel()) if i!=p]]
class OrthoLocalizer(nn.Module):
 def __init__(self,u,q,bs,ba): super().__init__(); self.u=nn.Parameter(u); self.q=nn.Parameter(q); self.bs=nn.Parameter(bs); self.ba=nn.Parameter(ba)
 def forward(self,h):
  v=basis(self.u)@self.q; S=h@self.u+self.bs; Q=h@v+self.ba; return torch.stack((S+Q,S-Q),-1)
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def atomic_save(obj,path):
 import os,tempfile
 fd,tmp=tempfile.mkstemp(dir=path.parent); os.close(fd)
 try: torch.save(obj,tmp); os.replace(tmp,path)
 finally:
  if os.path.exists(tmp): os.unlink(tmp)
def dev_loss(m,rows,device):
 m.eval(); vals=[]
 with torch.no_grad():
  for j in range(0,min(len(rows),128),32):
   rs=rows[j:j+32]; L=max(len(r['token_ids']) for r in rs)+2; x=torch.zeros((len(rs),L),dtype=torch.long,device=device); y=torch.full((len(rs),L),-100,dtype=torch.long,device=device)
   for i,r in enumerate(rs):
    z=[2]+r['token_ids']+[3]; x[i,:len(z)-1]=torch.tensor(z[:-1],device=device); y[i,1:len(z)]=torch.tensor(z[1:],device=device)
   vals.append(float(F.cross_entropy(m.base_model(x).reshape(-1,1024),y.reshape(-1),ignore_index=-100)))
 return sum(vals)/len(vals)
def set_scope(m,binding):
 for p in m.parameters(): p.requires_grad_(binding)
 if not binding:
  for p in m.base_model.parameters(): p.requires_grad_(True)
  for i in range(4):
   for p in m.base_model.blocks[i].parameters(): p.requires_grad_(False)
  for p in m.localizer.parameters(): p.requires_grad_(False)
  for mod in (m.wq,m.wk,m.wv,m.wo):
   for p in mod.parameters(): p.requires_grad_(False)
def loc_loss(att,docs):
 vals=[]
 for i,d in enumerate(docs):
  s0,s1=(int(d['query_key_clause_pos']),int(d['other_key_pos'])) if int(d['query_slot'])==0 else (int(d['other_key_pos']),int(d['query_key_clause_pos']))
  x=att[i].clamp_min(1e-12); vals.append(-torch.maximum(x[s0-1,0].log()+x[s1-1,1].log(),x[s1-1,0].log()+x[s0-1,1].log()))
 return torch.stack(vals).mean()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--arm',choices=['factual','control'],required=True); ap.add_argument('--seed',type=int,default=87004); ap.add_argument('--output',required=True); ap.add_argument('--resume',action='store_true'); args=ap.parse_args(); assert args.seed==87004
 proto=json.loads((R/'HR1_PROTOCOL.json').read_text()); assert sha(Path(proto['parent_path']))==proto['parent_sha256']; assert torch.cuda.is_available()
 random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed); torch.use_deterministic_algorithms(True)
 m=Treatment13Model().to(DEVICE); raw=torch.load(proto['parent_path'],map_location=DEVICE,weights_only=False); sd=dict(raw['model_state_dict']); u=sd.pop('localizer.u'); qv=sd.pop('localizer.q'); bs0=sd.pop('localizer.bs'); ba0=sd.pop('localizer.ba'); sd.pop('localizer.scorer.weight',None); sd.pop('localizer.scorer.bias',None); m.load_state_dict(sd,strict=False); m.localizer=OrthoLocalizer(u,qv,bs0,ba0).to(DEVICE)
 tr=[json.loads(x) for x in (R/'ENGLISH_TRAIN.jsonl').read_text(encoding='utf-8').splitlines()]; dv=[json.loads(x) for x in (R/'ENGLISH_DEV.jsonl').read_text(encoding='utf-8').splitlines()]; sched=json.loads((R/'ENGLISH_SCHEDULE.json').read_text())['batches']; bind=json.loads((Path(proto['binding_pool_path'])).read_text())['quartets']; bs=json.loads((R/'BINDING_SCHEDULE.json').read_text())['batches']; byid={d['doc_id']:d for q in bind for d in q['docs']}; rec={d['id']:d for d in tr}; opt=torch.optim.AdamW(m.parameters(),lr=5e-5,betas=(.9,.999),eps=1e-8,weight_decay=.05,amsgrad=False,foreach=False,fused=False); out=Path(args.output); out.mkdir(parents=True,exist_ok=True); start=1; devs={'0':dev_loss(m,dv,DEVICE)}
 if args.resume:
  st=torch.load(out/'restart.pt',map_location=DEVICE,weights_only=False); assert st['arm']==args.arm and st['parent_sha256']==proto['parent_sha256'] and st['protocol_sha256']==sha(R/'HR1_PROTOCOL.json') and st['schedule_sha256']==sha(R/'ENGLISH_SCHEDULE.json'); m.load_state_dict(st['model_state']); opt.load_state_dict(st['optimizer_state']); random.setstate(st['python_rng']); torch.set_rng_state(st['torch_rng'].cpu()); start=int(st['completed_update'])+1; devs=st.get('dev_loss',devs)
 for u in range(start,501):
  isb=u%10==0; set_scope(m,isb); opt.zero_grad(set_to_none=True)
  if not isb:
   rows=[rec[x] for x in sched[u-1-((u-1)//10)]['record_ids']]; L=max(len(x['token_ids']) for x in rows)+2; x=torch.zeros((64,L),dtype=torch.long,device=DEVICE); y=torch.full((64,L),-100,dtype=torch.long,device=DEVICE)
   for i,r in enumerate(rows):
    z=[2]+r['token_ids']+[3]; x[i,:len(z)-1]=torch.tensor(z[:-1],device=DEVICE); y[i,1:len(z)]=torch.tensor(z[1:],device=DEVICE)
   logits=m.base_model(x); ce=F.cross_entropy(logits.reshape(-1,1024),y.reshape(-1),ignore_index=-100); probs=logits.softmax(-1); mask=torch.zeros_like(probs,dtype=torch.bool); [mask.scatter_(2,x.roll(k,dims=1).unsqueeze(-1),True) for k in range(1,9)]; valid=y.ne(-100); ul=(-torch.log((1-probs.clamp_max(.999999)).clamp_min(1e-6))*mask).sum(-1)[valid].mean(); loss=ce+0.1*ul
  else:
   docs=[d for q in bs[u//10-1]['quartet_ids'] for d in bind[q]['docs']]; x=torch.tensor([d['full_document_token_ids'] for d in docs],device=DEVICE); q=torch.tensor([d['qdp'] for d in docs],device=DEVICE); a=torch.tensor([d['answer_causal_position'] for d in docs],device=DEVICE); y=torch.tensor([d['target_value_token'] for d in docs],device=DEVICE); logits,ex=m(x,q,a); ix=torch.arange(len(docs),device=DEVICE); loss=F.cross_entropy(logits[ix,a],y)+1.0536573711078283*loc_loss(ex['localization_attention'],docs)
  loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.grad is not None],2.0); opt.step()
  if u in (100,500): devs[str(u)]=dev_loss(m,dv,DEVICE); torch.save({'model_state_dict':m.state_dict(),'seed':args.seed,'updates':u,'arm':args.arm,'parent_sha256':proto['parent_sha256']},out/f'checkpoint_{u}.pt')
  atomic_save({'completed_update':u,'arm':args.arm,'parent_sha256':proto['parent_sha256'],'protocol_sha256':sha(R/'HR1_PROTOCOL.json'),'schedule_sha256':sha(R/'ENGLISH_SCHEDULE.json'),'model_state':m.state_dict(),'optimizer_state':opt.state_dict(),'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),'dev_loss':devs},out/'restart.pt')
 (out/'DEV_RESULTS.json').write_text(json.dumps({'dev_loss':devs},indent=2),encoding='utf-8')
if __name__=='__main__': main()








