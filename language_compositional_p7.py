from __future__ import annotations
import hashlib,json,sys,time,itertools
from pathlib import Path
import torch,torch.nn as nn,torch.nn.functional as F
from tokenizers import Tokenizer
sys.path.insert(0,r'C:\DaveLM-CADAVER');sys.path.insert(0,r'C:\DaveLM-v0.9')
from treatment13_model import Treatment13Model
ROOT=Path(r'C:\DaveLM-CADAVER');PARENT=ROOT/'language_sentencebound_p5/latest.pt';OUT=ROOT/'language_compositional_p7';TOK=Path(r'C:\DaveLM-v0.9/tokenizer/v0_7/davelm_tokenizer.json');DEVICE='cuda:0';SEED=8380;STEPS=500;BATCH=64
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def basis(u):
 un=u/torch.linalg.vector_norm(u);p=int(torch.argmax(torch.abs(un)));e=torch.zeros_like(un);e[p]=1. if un[p]>=0 else -1.;w=un-e;H=torch.eye(un.numel(),device=u.device,dtype=u.dtype)-2*torch.outer(w,w)/torch.dot(w,w);return H[:,[i for i in range(un.numel()) if i!=p]]
class OrthoLocalizer(nn.Module):
 def __init__(self,u,q,bs,ba):super().__init__();self.u=nn.Parameter(u);self.q=nn.Parameter(q);self.bs=nn.Parameter(bs);self.ba=nn.Parameter(ba)
 def forward(self,h):
  v=basis(self.u)@self.q;S=h@self.u+self.bs;R=h@v+self.ba;return torch.stack((S+R,S-R),-1)
def main():
 OUT.mkdir(exist_ok=True);tok=Tokenizer.from_file(str(TOK));names=['Lily','Mia','Sam','Tom','Nora','Owen','Zoe','Alex'];objs=['ball','book','toy','hat','kite','car','box','fish'];cols=['red','blue','green','yellow'];spec=[('found','found a','picked up'),('saw','saw a','smiled at'),('picked','picked up the','carried'),('carried','carried the','put down'),('liked','liked the','played with'),('put','put the on the table','looked at'),('looked','looked for the','searched for'),('played','played with the','had fun with')];train=[];held=[]
 for ni,n in enumerate(names):
  for oi,o in enumerate(objs):
   for ci,c in enumerate(cols):
    target=held if (ni+oi+ci)%5==0 else train
    for verb,first,second in spec:
     if verb=='put':s=f'{n} put the {c} {o} on the table. {n} looked at the {c} {o}.'
     else:s=f'{n} {first} {c} {o}. {n} {second} the {c} {o}.'
     target.append(s)
 text=''.join(s+'\n' for s in train);(OUT/'TRAIN_TEXT.txt').write_text(text,encoding='utf-8');seqs=[[2]+tok.encode(s).ids+[3] for s in train if len(tok.encode(s).ids)<62];raw=torch.load(PARENT,map_location='cpu',weights_only=True);sd=raw['model_state_dict'];m=Treatment13Model();m.localizer=OrthoLocalizer(*[sd['localizer.'+n].clone() for n in ('u','q','bs','ba')]);m.load_state_dict(sd,strict=True);m.to(DEVICE).eval();panel=[held[i] for i in [0,7,14,21,28,35,42,49,56,63,70,77]];rec={'parent_sha256':sha(PARENT),'tokenizer_sha256':sha(TOK),'synthetic_train_sha256':sha(OUT/'TRAIN_TEXT.txt'),'train_examples':len(seqs),'heldout_examples':len(held),'heldout_prompts':panel,'updates':STEPS,'seed':SEED,'lr':5e-5,'varied_relations':8}
 for p in m.parameters():p.requires_grad_(False)
 for p in m.base_model.parameters():p.requires_grad_(True)
 opt=torch.optim.AdamW([p for p in m.base_model.parameters() if p.requires_grad],lr=5e-5,weight_decay=.05);g=torch.Generator().manual_seed(SEED+9000);logs=[];m.train();t=time.time()
 for step in range(1,STEPS+1):
  ix=torch.randint(0,len(seqs),(BATCH,),generator=g);batch=[seqs[int(i)] for i in ix];x=torch.zeros((BATCH,63),dtype=torch.long);y=torch.full((BATCH,63),-100,dtype=torch.long)
  for j,z in enumerate(batch):x[j,:len(z)-1]=torch.tensor(z[:-1]);y[j,:len(z)-1]=torch.tensor(z[1:])
  x=x.to(DEVICE);y=y.to(DEVICE);loss=F.cross_entropy(m.base_model(x).reshape(-1,1024),y.reshape(-1),ignore_index=-100);opt.zero_grad(set_to_none=True);loss.backward();gn=torch.nn.utils.clip_grad_norm_(m.base_model.parameters(),2.0);opt.step();logs.append({'step':step,'loss':float(loss),'grad_norm':float(gn)});
  if step%100==0:print(json.dumps(logs[-1]),flush=True)
 m.eval()
 def gen(p):
  ids=[2]+tok.encode(p).ids
  with torch.no_grad():
   for _ in range(22):
    n=int(m.base_model(torch.tensor([ids[-256:]],device=DEVICE))[:,-1,:].argmax());ids.append(n)
    if n==3:break
  return tok.decode(ids,skip_special_tokens=True)
 rec['elapsed_seconds']=time.time()-t;rec['generations']=[{'prompt':p,'output':gen(p)} for p in panel];torch.save({'model_state_dict':m.state_dict(),'parent':str(PARENT),'updates':STEPS,'seed':SEED},OUT/'latest.pt');(OUT/'metrics.jsonl').write_text(''.join(json.dumps(z)+'\n' for z in logs),encoding='utf-8');(OUT/'RESULTS.json').write_text(json.dumps(rec,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(rec,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
