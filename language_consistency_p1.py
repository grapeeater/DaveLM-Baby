from __future__ import annotations
import hashlib,json,sys,time,itertools
from pathlib import Path
import torch, torch.nn.functional as F
from tokenizers import Tokenizer
sys.path.insert(0,r'C:\DaveLM-CADAVER'); sys.path.insert(0,r'C:\DaveLM-v0.9')
from language_continuation_p0_v1 import load
ROOT=Path(r'C:\DaveLM-CADAVER'); PARENT=ROOT/'language_continuation_p0_v1/latest.pt'; OUT=ROOT/'language_consistency_p1'; TOK=Path(r'C:\DaveLM-v0.9/tokenizer/v0_7/davelm_tokenizer.json'); DEV=ROOT/'language_pilot_0_tinystories_seed8380/language_dev.jsonl'; SEED=8380; DEVICE='cuda:0'; STEPS=500
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 OUT.mkdir(exist_ok=True); tok=Tokenizer.from_file(str(TOK)); names=['Lily','Mia','Sam','Tom','Nora','Owen','Zoe','Alex']; objs=['ball','book','toy','hat','kite','car','box','fish']; colors=['red','blue','green','yellow']; acts=['played with','looked at','carried','found']; ex=[]
 for i,(n,o,c,a) in enumerate(itertools.product(names,objs,colors,acts)):
  if i%5: ex.append(f'{n} found a {c} {o}. {n} {a} the {c} {o}.')
 held=[f'{n} found a {c} {o}.' for i,(n,o,c,a) in enumerate(itertools.product(names,objs,colors,acts)) if not i%5]
 text=''.join(s+'\n' for s in ex); (OUT/'TRAIN_TEXT.txt').write_text(text,encoding='utf-8'); corpus_sha=sha(OUT/'TRAIN_TEXT.txt'); stream=[]
 for s in ex: stream += [2]+tok.encode(s).ids+[3]
 tr=torch.tensor(stream,dtype=torch.long); m=load().to(DEVICE); m.eval();
 panel=[held[i] for i in [0,7,14,21,28,35,42,49,56,63,70,77]]; prompts=[{'prompt':p,'expected':p.replace(' found a ',' ').replace('.','.',1)} for p in panel]
 rec={'parent_sha256':sha(PARENT),'tokenizer_sha256':sha(TOK),'synthetic_train_sha256':corpus_sha,'train_examples':len(ex),'heldout_examples':len(held),'updates':STEPS,'seed':SEED,'lr':3e-4,'weight_decay':.05,'batch':64,'context':256,'panel_prompts':panel}
 for p in m.parameters(): p.requires_grad_(False)
 for p in m.base_model.parameters(): p.requires_grad_(True)
 opt=torch.optim.AdamW([p for p in m.base_model.parameters() if p.requires_grad],lr=3e-4,weight_decay=.05); g=torch.Generator().manual_seed(SEED+3000); logs=[]; m.train();t=time.time()
 for step in range(1,STEPS+1):
  st=torch.randint(0,len(tr)-257,size=(64,),generator=g); off=torch.arange(256); x=tr[st[:,None]+off[None,:]].to(DEVICE); y=tr[st[:,None]+off[None,:]+1].to(DEVICE); loss=F.cross_entropy(m.base_model(x).reshape(-1,1024),y.reshape(-1)); opt.zero_grad(set_to_none=True);loss.backward();gn=torch.nn.utils.clip_grad_norm_(m.base_model.parameters(),2.0);opt.step();logs.append({'step':step,'loss':float(loss),'grad_norm':float(gn)});
  if step%100==0: print(json.dumps(logs[-1]),flush=True)
 m.eval();
 def gen(p):
  ids=[tok.token_to_id('<bos>')]+tok.encode(p).ids
  with torch.no_grad():
   for _ in range(24):ids.append(int(m.base_model(torch.tensor([ids[-256:]],device=DEVICE))[:,-1,:].argmax()))
  return tok.decode(ids,skip_special_tokens=True)
 rec['elapsed_seconds']=time.time()-t;rec['generations']=[{'prompt':p,'output':gen(p)} for p in panel];torch.save({'model_state_dict':m.state_dict(),'parent':str(PARENT),'updates':STEPS,'seed':SEED},OUT/'latest.pt');(OUT/'metrics.jsonl').write_text(''.join(json.dumps(z)+'\n' for z in logs),encoding='utf-8');(OUT/'RESULTS.json').write_text(json.dumps(rec,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(rec,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
