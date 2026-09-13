from __future__ import annotations
import hashlib,json,math,random,sys,time
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer
ROOT=Path(r'C:\DaveLM-CADAVER'); PILOT=ROOT/'language_pilot_0_tinystories_seed8380'; OUT=ROOT/'language_continuation_p0_v1'; CK=PILOT/'pilot_run/checkpoints/seed_8380/latest.pt'; TOK=Path(r'C:\DaveLM-v0.9/tokenizer/v0_7/davelm_tokenizer.json')
sys.path.insert(0,str(ROOT)); sys.path.insert(0,r'C:\DaveLM-v0.9')
from treatment13_model import Treatment13Model
SEED=8380; UPDATES=1000; DEVICE='cuda:0'
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def basis(u):
 un=u/torch.linalg.vector_norm(u); p=int(torch.argmax(torch.abs(un))); e=torch.zeros_like(un); e[p]=1. if un[p]>=0 else -1.; w=un-e; H=torch.eye(un.numel(),device=u.device,dtype=u.dtype)-2*torch.outer(w,w)/torch.dot(w,w); return H[:,[i for i in range(un.numel()) if i!=p]]
class OrthoLocalizer(nn.Module):
 def __init__(self,u,q,bs,ba): super().__init__(); self.u=nn.Parameter(u); self.q=nn.Parameter(q); self.bs=nn.Parameter(bs); self.ba=nn.Parameter(ba)
 def forward(self,h):
  v=basis(self.u)@self.q; S=h@self.u+self.bs; R=h@v+self.ba; return torch.stack((S+R,S-R),-1)
def load():
 raw=torch.load(CK,map_location='cpu',weights_only=True); sd=dict(raw['model_state_dict']); u,q,bs,ba=[sd[f'localizer.{n}'].clone() for n in ('u','q','bs','ba')]; m=Treatment13Model(); m.localizer=OrthoLocalizer(u,q,bs,ba); m.load_state_dict(sd,strict=True); return m
def eval_ppl(m,stream):
 g=torch.Generator().manual_seed(SEED+1001); vals=[]; m.eval()
 with torch.no_grad():
  for _ in range(40):
   st=torch.randint(0,len(stream)-257,size=(64,),generator=g); off=torch.arange(256); x=stream[st[:,None]+off[None,:]].to(DEVICE); y=stream[st[:,None]+off[None,:]+1].to(DEVICE); vals.append(float(F.cross_entropy(m.base_model(x).reshape(-1,1024),y.reshape(-1))))
 return sum(vals)/len(vals)
def gen(m,tok,prompt,n=48):
 ids=[tok.token_to_id('<bos>')]+tok.encode(prompt).ids; m.eval()
 with torch.no_grad():
  for _ in range(n): ids.append(int(m.base_model(torch.tensor([ids[-256:]],device=DEVICE))[:,-1,:].argmax()))
 return tok.decode(ids,skip_special_tokens=True)
def main():
 OUT.mkdir(exist_ok=True); tok=Tokenizer.from_file(str(TOK)); stories=[json.loads(x)['text'] for x in (PILOT/'language_train.jsonl').read_text(encoding='utf-8').splitlines()]; dev=[json.loads(x)['text'] for x in (PILOT/'language_dev.jsonl').read_text(encoding='utf-8').splitlines()];
 enc=lambda xs: torch.tensor([z for s in xs for z in tok.encode(s).ids],dtype=torch.long)
 tr,dv=enc(stories),enc(dev); m=load().to(DEVICE); m.eval(); pre=eval_ppl(m,dv); prompts=['Hello, my name is','The dog is','Two plus two is','Tell me something.','Why did the dog run?','I am happy because']
 rec={'parent':str(CK),'parent_sha256':sha(CK),'tokenizer_sha256':sha(TOK),'train_sha256':sha(PILOT/'language_train.jsonl'),'dev_sha256':sha(PILOT/'language_dev.jsonl'),'updates':UPDATES,'lr':3e-4,'weight_decay':.05,'batch':64,'context':256,'seed':SEED,'device':DEVICE,'pre_ppl':pre,'pre_generations':{p:gen(m,tok,p) for p in prompts}}
 for p in m.parameters(): p.requires_grad_(False)
 for p in m.base_model.parameters(): p.requires_grad_(True)
 opt=torch.optim.AdamW([p for p in m.base_model.parameters() if p.requires_grad],lr=3e-4,weight_decay=.05); g=torch.Generator().manual_seed(SEED+2000); logs=[]; t=time.time(); m.train()
 for step in range(1,UPDATES+1):
  st=torch.randint(0,len(tr)-257,size=(64,),generator=g); off=torch.arange(256); x=tr[st[:,None]+off[None,:]].to(DEVICE); y=tr[st[:,None]+off[None,:]+1].to(DEVICE); loss=F.cross_entropy(m.base_model(x).reshape(-1,1024),y.reshape(-1)); opt.zero_grad(set_to_none=True); loss.backward(); gn=torch.nn.utils.clip_grad_norm_(m.base_model.parameters(),2.0); opt.step(); logs.append({'step':step,'loss':float(loss),'grad_norm':float(gn)});
  if step%100==0: print(json.dumps(logs[-1]),flush=True)
 m.eval(); post=eval_ppl(m,dv); rec.update({'post_ppl':post,'post_generations':{p:gen(m,tok,p) for p in prompts},'elapsed_seconds':time.time()-t}); torch.save({'model_state_dict':m.state_dict(),'parent':str(CK),'updates':UPDATES,'seed':SEED},OUT/'latest.pt'); (OUT/'metrics.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in logs),encoding='utf-8'); (OUT/'RESULTS.json').write_text(json.dumps(rec,indent=2,ensure_ascii=False),encoding='utf-8'); print(json.dumps(rec,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
