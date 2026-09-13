from __future__ import annotations
import hashlib,json,sys,time,itertools
from pathlib import Path
import torch, torch.nn.functional as F
from tokenizers import Tokenizer
sys.path.insert(0,r'C:\DaveLM-CADAVER');sys.path.insert(0,r'C:\DaveLM-v0.9')
from language_continuation_p0_v1 import load
ROOT=Path(r'C:\DaveLM-CADAVER');PARENT=ROOT/'language_consistency_p1/latest.pt';PILOT=ROOT/'language_pilot_0_tinystories_seed8380';OUT=ROOT/'language_mixed_p2';TOK=Path(r'C:\DaveLM-v0.9/tokenizer/v0_7/davelm_tokenizer.json');SEED=8380;DEVICE='cuda:0';STEPS=500
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 OUT.mkdir(exist_ok=True);tok=Tokenizer.from_file(str(TOK)); stories=[json.loads(x)['text'] for x in (PILOT/'language_train.jsonl').read_text(encoding='utf-8').splitlines()]; syn=[]
 for n,o,c,a in itertools.product(['Lily','Mia','Sam','Tom','Nora','Owen','Zoe','Alex'],['ball','book','toy','hat','kite','car','box','fish'],['red','blue','green','yellow'],['played with','looked at','carried','found']):syn.append(f'{n} found a {c} {o}. {n} {a} the {c} {o}.')
 enc=lambda xs:[z for s in xs for z in ([2]+tok.encode(s).ids+[3])];tr=torch.tensor(enc(stories),dtype=torch.long);co=torch.tensor(enc(syn),dtype=torch.long);m=load().to(DEVICE);m.eval();panel=['Zoe found a yellow kite.','Alex found a green book.','Nora found a red car.','Owen found a blue box.','Mia found a green fish.','Sam found a yellow hat.','Lily found a blue car.','Tom found a red kite.'];rec={'parent_sha256':sha(PARENT),'tokenizer_sha256':sha(TOK),'tiny_train_sha256':sha(PILOT/'language_train.jsonl'),'synthetic_sha256':hashlib.sha256((''.join(s+'\n' for s in syn)).encode()).hexdigest(),'updates':STEPS,'schedule':'9 TinyStories batches then 1 consistency batch','seed':SEED,'panel_prompts':panel}
 for p in m.parameters():p.requires_grad_(False)
 for p in m.base_model.parameters():p.requires_grad_(True)
 opt=torch.optim.AdamW([p for p in m.base_model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.05);gt=torch.Generator().manual_seed(SEED+4000);gc=torch.Generator().manual_seed(SEED+4001);logs=[];m.train();t=time.time()
 for step in range(1,STEPS+1):
  src=co if step%10==0 else tr;g=gc if step%10==0 else gt;st=torch.randint(0,len(src)-257,size=(64,),generator=g);off=torch.arange(256);x=src[st[:,None]+off[None,:]].to(DEVICE);y=src[st[:,None]+off[None,:]+1].to(DEVICE);loss=F.cross_entropy(m.base_model(x).reshape(-1,1024),y.reshape(-1));opt.zero_grad(set_to_none=True);loss.backward();gn=torch.nn.utils.clip_grad_norm_(m.base_model.parameters(),2.0);opt.step();logs.append({'step':step,'scope':'consistency' if step%10==0 else 'tinystories','loss':float(loss),'grad_norm':float(gn)});
  if step%100==0:print(json.dumps(logs[-1]),flush=True)
 m.eval()
 def gen(p):
  ids=[2]+tok.encode(p).ids
  with torch.no_grad():
   for _ in range(28):ids.append(int(m.base_model(torch.tensor([ids[-256:]],device=DEVICE))[:,-1,:].argmax()))
  return tok.decode(ids,skip_special_tokens=True)
 rec['elapsed_seconds']=time.time()-t;rec['generations']=[{'prompt':p,'output':gen(p)} for p in panel];torch.save({'model_state_dict':m.state_dict(),'parent':str(PARENT),'updates':STEPS,'seed':SEED},OUT/'latest.pt');(OUT/'metrics.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in logs),encoding='utf-8');(OUT/'RESULTS.json').write_text(json.dumps(rec,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(rec,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
