
from __future__ import annotations
import argparse, hashlib, json, math, os, random, sys
from array import array
from pathlib import Path
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

ARCH=Path(r"C:\DaveLM-CADAVER\baby_vnext_60m_design_v1")
PARENT=Path(r"C:\DaveLM-CADAVER\baby_vnext_phase1g_language_v1_run_seed610001\checkpoints\best.pt")
TOK=Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")
sys.path.insert(0,str(ARCH))

def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def u16(p):
 a=array('H');
 with open(p,'rb') as f: a.fromfile(f,Path(p).stat().st_size//2)
 if sys.byteorder!='little': a.byteswap()
 return torch.tensor(a,dtype=torch.long)
def u32(p,n):
 a=array('I');
 with open(p,'rb') as f: a.fromfile(f,n)
 if sys.byteorder!='little': a.byteswap()
 return torch.tensor(a,dtype=torch.long)
def seed_all(s):
 random.seed(s); torch.manual_seed(s)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
 torch.use_deterministic_algorithms(True)
def load_model():
 from baby_vnext.config import BabyVNextConfig
 from baby_vnext.binding import BabyVNextWithBinding
 payload=torch.load(PARENT,map_location='cpu',weights_only=False)
 if payload.get('schema')!='baby_vnext_phase1_model_v1' or payload.get('completed_update')!=6000: raise RuntimeError('parent schema/update mismatch')
 m=BabyVNextWithBinding(BabyVNextConfig.load(ARCH/'BABY_VNEXT_CONFIG.json'))
 m.load_state_dict(payload['model_state_dict'],strict=True)
 return m,payload['model_state_dict']
def rows(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def find_span(prompt,cand):
 # candidate answer ids include trailing period; anchor all name subtokens only
 c=list(map(int,cand)); p=list(map(int,prompt));
 for drop in (1,0):
  q=c[:-drop] if drop else c
  if not q: continue
  variants=[q]
  if len(q)>1: variants.append(q[1:])  # sentence-start occurrence omits leading-space token
  for v in variants:
   for i in range(len(p)-len(v)+1):
    if p[i:i+len(v)]==v: return list(range(i+1,i+1+len(v))) # +1 for BOS
 raise RuntimeError('candidate mention span not found')
def pointer_margin(m,row,dev):
 prompt=[2]+list(map(int,row['prompt_ids'])); P=len(prompt)
 hs=m.base_model.forward_hidden(torch.tensor([prompt],device=dev))
 q=F.normalize(hs[0,P-1],dim=-1)
 sims=[]; scores=[]
 for cand in row['cand_ids']:
  span=find_span(row['prompt_ids'],cand)
  k=F.normalize(hs[0,span].mean(0),dim=-1); sims.append((q*k).sum())
  c=list(map(int,cand)); x=torch.tensor([prompt+c],device=dev)
  logits=m.base_model(x); lp=F.log_softmax(logits[0,P-1:P-1+len(c)],dim=-1); idx=torch.tensor(c,device=dev)
  scores.append(lp[torch.arange(len(c),device=dev),idx].mean())
 ci=int(row['correct_index']); other=max(v for j,v in enumerate(scores) if j!=ci)
 ptr=F.cross_entropy(torch.stack(sims).view(1,-1),torch.tensor([ci],device=dev))
 mar=torch.clamp(torch.tensor(1.,device=dev)-(scores[ci]-other),min=0.)
 return ptr,mar,scores,sims
@torch.no_grad()
def eval_rows(m,rs,dev):
 m.eval(); n=0; pc=0; fc=0; margins=[]
 for r in rs:
  p,ma,s,si=pointer_margin(m,r,dev); ci=int(r['correct_index']); n+=1; pc+=int(int(torch.argmax(torch.stack(si)))==ci); fc+=int(s[ci]>=max(v for j,v in enumerate(s) if j!=ci)); margins.append(float(s[ci]-max(v for j,v in enumerate(s) if j!=ci)))
 return {'n':n,'pointer_retrieval':pc/n,'native_forced_choice':fc/n,'mean_margin':sum(margins)/n}
@torch.no_grad()
def lang_ce(m,stream,starts,dev,ctx=256,mb=8):
 m.eval(); tot=0.; cnt=0; off=torch.arange(ctx)
 for i in range(0,len(starts),mb):
  s=starts[i:i+mb]; x=stream[s[:,None]+off[None,:]].to(dev); y=stream[s[:,None]+off[None,:]+1].to(dev); out=m(x); z=out[0] if isinstance(out,tuple) else out; tot+=float(F.cross_entropy(z.reshape(-1,z.shape[-1]),y.reshape(-1),reduction='sum')); cnt+=y.numel()
 return tot/cnt
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--bundle',type=Path,required=True); args=ap.parse_args()
 if not torch.cuda.is_available(): raise RuntimeError('GPU unavailable')
 dev=torch.device('cuda'); seed_all(args.seed); args.out.mkdir(parents=True,exist_ok=False)
 m,parent_sd=load_model();
 for n,p in m.named_parameters(): p.requires_grad_(n.startswith('base_model.')); p.grad=None
 m.to(dev); m.train(); tok=Tokenizer.from_file(str(TOK));
 corp=rows(args.bundle/'data'/'corpus.json'); train=corp['train']; dv=corp['dev'];
 stream=u16(r'C:\DaveLM-CADAVER\baby_vnext_phase1g_language_v1\data\LANGUAGE_TRAIN_STREAM.u16'); starts=u32(r'C:\DaveLM-CADAVER\baby_vnext_phase1g_language_v1\data\EVAL_WINDOW_STARTS.u32',1600)[1280:];
 devstream=u16(r'C:\DaveLM-CADAVER\baby_vnext_phase1g_language_v1\data\LANGUAGE_DEV_STREAM.u16'); devstarts=u32(r'C:\DaveLM-CADAVER\baby_vnext_phase1g_language_v1\data\EVAL_WINDOW_STARTS.u32',1600)[:1280]
 opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=5e-5,betas=(.9,.999),eps=1e-8,weight_decay=.05,foreach=False,fused=False)
 base=eval_rows(m,dv,dev); base['dev_ce']=lang_ce(m,devstream,devstarts,dev); base['train_ce']=lang_ce(m,stream,starts,dev); base['update']=0
 (args.out/'evaluation_0000.json').write_text(json.dumps(base,indent=2)); metrics=[]; qidx=list(range(len(train))); rng=random.Random(args.seed+101)
 def atomic(path,obj):
  t=path.with_suffix(path.suffix+'.tmp'); torch.save(obj,t); os.replace(t,path)
 for u in range(1,501):
  opt.zero_grad(set_to_none=True)
  if u%10==0:
   ids=[((u//10-1)*8+j)%len(starts) for j in range(8)]; loss=0.; off=torch.arange(256); ss=starts[ids]
   for j in range(8):
    s=ss[j:j+1]; x=stream[s[:,None]+off[None,:]].to(dev); y=stream[s[:,None]+off[None,:]+1].to(dev); out=m(x); z=out[0] if isinstance(out,tuple) else out; loss=loss+F.cross_entropy(z.reshape(-1,z.shape[-1]),y.reshape(-1))/8
   loss.backward(); kind='lang'; raw=float(loss.detach().cpu())
  else:
   batch=[qidx[(u*32+j)%len(qidx)] for j in range(32)]; loss=0.;
   for j in batch:
    p,ma,_,_=pointer_margin(m,train[j],dev); loss=loss+(ma+p)/32
   loss.backward(); kind='qa'; raw=float(loss.detach().cpu())
  gn=torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],2.0)
  if not torch.isfinite(gn): raise RuntimeError('nonfinite grad')
  if any(p.grad is not None for n,p in m.named_parameters() if not n.startswith('base_model.')): raise RuntimeError('frozen binding gradient')
  opt.step(); metrics.append({'update':u,'kind':kind,'loss':raw,'grad_norm':float(gn)})
  if u in (100,300,500):
   ev=eval_rows(m,dv,dev); ev.update({'update':u,'dev_ce':lang_ce(m,devstream,devstarts,dev),'train_ce':lang_ce(m,stream,starts,dev)}); (args.out/f'evaluation_{u:04d}.json').write_text(json.dumps(ev,indent=2)); atomic(args.out/f'checkpoint_{u:04d}.pt',{'schema':'baby_vnext_phase2a_t3_model_v1','update':u,'model_state_dict':m.state_dict()})
 (args.out/'training_metrics.jsonl').write_text('\n'.join(json.dumps(x) for x in metrics)+'\n'); (args.out/'FINAL_STATUS.json').write_text(json.dumps({'seed':args.seed,'update':500,'classification':'T3_FAIL_NO_REPRESENTATION'},indent=2))
if __name__=='__main__': main()
