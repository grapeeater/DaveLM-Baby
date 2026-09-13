from __future__ import annotations
import hashlib, json, shutil, sys
from pathlib import Path
import torch
import torch.nn as nn
from tokenizers import Tokenizer
ROOT=Path(r"C:\DaveLM-CADAVER"); TREAT=ROOT/"treatment13_orthogonal_shared_unbounded_seed8380"; CK=TREAT/"checkpoints/orthogonal_shared_unbounded/seed_8380/latest.pt"; TOK=Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json"); OUT=ROOT/"forensics/post_graduation_raw_generation_probe_seed8380"; ARCH=OUT/"archive/graduating_checkpoint_frozen.pt"; sys.path.insert(0,str(ROOT)); sys.path.insert(0,r"C:\DaveLM-v0.9")
from treatment13_model import Treatment13Model
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def basis(u):
 un=u/torch.linalg.vector_norm(u); p=int(torch.argmax(torch.abs(un))); e=torch.zeros_like(un); e[p]=1. if un[p]>=0 else -1.; w=un-e; H=torch.eye(un.numel(),device=u.device,dtype=u.dtype)-2*torch.outer(w,w)/torch.dot(w,w); return H[:,[i for i in range(un.numel()) if i!=p]]
class OrthoLocalizer(nn.Module):
 def __init__(self,u,q,bs,ba): super().__init__(); self.u=nn.Parameter(u); self.q=nn.Parameter(q); self.bs=nn.Parameter(bs); self.ba=nn.Parameter(ba)
 def forward(self,h):
  v=basis(self.u)@self.q; S=h@self.u+self.bs; R=h@v+self.ba; return torch.stack((S+R,S-R),-1)
def main():
 OUT.mkdir(parents=True,exist_ok=True); ARCH.parent.mkdir(parents=True,exist_ok=True); assert sha(CK).lower()=="fed298748e62def6f1751ef1bb7df0cdec51d53fac6e4c86e8c00a95dccdd430"
 if not ARCH.exists(): shutil.copy2(CK,ARCH)
 ah=sha(ARCH); assert ah==sha(CK); tok=Tokenizer.from_file(str(TOK)); bos=tok.token_to_id('<bos>'); eos=tok.token_to_id('<eos>'); assert tok.get_vocab_size()==1024 and bos==2 and eos==3
 prompts=["Hello","Hello, my name is","What is your name?","The dog is","Two plus two is","If red means apple and blue means banana, what does red mean?","Tell me something.","he womputeld"]; ents=[]
 for p in prompts:
  e=tok.encode(p); assert tok.decode(e.ids)==p; ents.append((p,e.ids,e.tokens))
 dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); raw=torch.load(CK,map_location=dev,weights_only=False); base=Treatment13Model().to(dev); sd=dict(raw['model_state_dict']); u=sd.pop('localizer.u'); q=sd.pop('localizer.q'); bs=sd.pop('localizer.bs'); ba=sd.pop('localizer.ba'); sd.pop('localizer.scorer.weight',None); sd.pop('localizer.scorer.bias',None); missing,unexpected=base.load_state_dict(sd,strict=False); assert set(missing)=={'localizer.scorer.weight','localizer.scorer.bias'} and not unexpected; base.localizer=OrthoLocalizer(u,q,bs,ba).to(dev); base.eval(); [p.requires_grad_(False) for p in base.parameters()]
 settings={'decoder':'greedy argmax','max_new_tokens':32,'temperature':1.0,'top_k':0,'bos_id':bos,'eos_id':eos,'context_size':base.base_model.context_size,'model_interface':'Treatment13Model.base_model(tokens)','runs_per_prompt':1}; rows=[]
 with torch.inference_mode():
  for p,ids,toks in ents:
   g=[bos]+list(ids); stopped=False
   for _ in range(settings['max_new_tokens']):
    z=torch.tensor([g[-settings['context_size']:]],dtype=torch.long,device=dev); nxt=int(torch.argmax(base.base_model(z)[:,-1,:],dim=-1)); g.append(nxt)
    if nxt==eos: stopped=True; break
   rows.append({'prompt':p,'prompt_ids':ids,'prompt_tokens':toks,'generated_ids':g,'generated_tokens':[tok.id_to_token(i) for i in g],'raw_decoded_output':tok.decode(g,skip_special_tokens=True),'stopped_on_eos':stopped,'new_tokens':len(g)-1-len(ids)})
 result={'status':'POST_GRADUATION_RAW_GENERATION_PROBE_COMPLETE','checkpoint_path':str(CK),'checkpoint_sha256':sha(CK),'archival_copy_path':str(ARCH),'archival_copy_sha256':ah,'tokenizer_path':str(TOK),'tokenizer_sha256':sha(TOK),'vocabulary_size':tok.get_vocab_size(),'tokenizer_type':'byte-level BPE with byte fallback','probe_set_frozen_before_outputs':True,'generation_settings':settings,'optimizer_created':False,'optimizer_steps':0,'backward_passes':0,'parameter_modifications':0,'checkpoint_overwrites':0,'training':False,'retention_evaluations':0,'outputs':rows}
 (OUT/'RAW_RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); (OUT/'REPORT.md').write_text('# Post-graduation raw generation probe\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n\nNo cleanup, rerolls, or postprocessing were performed.\n',encoding='utf-8'); print(json.dumps(result,ensure_ascii=False,indent=2)); print('POST_GRADUATION_RAW_GENERATION_PROBE_COMPLETE')
if __name__=='__main__': main()
