import json,sys,torch
from pathlib import Path
sys.path.insert(0,r'C:\DaveLM-CADAVER');sys.path.insert(0,r'C:\DaveLM-v0.9')
from language_compositional_p7 import OrthoLocalizer
from treatment13_model import Treatment13Model
from tokenizers import Tokenizer
P=Path(__file__).resolve().parent;raw=torch.load(P/'latest.pt',map_location='cpu',weights_only=True);sd=raw['model_state_dict'];m=Treatment13Model();m.localizer=OrthoLocalizer(*[sd['localizer.'+n].clone() for n in ('u','q','bs','ba')]);m.load_state_dict(sd,strict=True);m.to('cuda:0').eval();t=Tokenizer.from_file(r'C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json');prompts=['Zoe discovered a purple umbrella.','Alex noticed a small bird.','Nora carried a heavy basket.','Owen placed the orange cup on the shelf.','Mia searched for her missing key.','Sam enjoyed the quiet garden.','Lily opened the old door.','Tom shared a sweet apple.'];rows=[]
for p in prompts:
 ids=[2]+t.encode(p).ids
 with torch.no_grad():
  for _ in range(24):
   n=int(m.base_model(torch.tensor([ids[-256:]],device='cuda:0'))[:,-1,:].argmax());ids.append(n)
   if n==3:break
 out=t.decode(ids,skip_special_tokens=True);rows.append({'prompt':p,'output':out,'response':out[len(p):].strip()})
(P/'NATURAL_TRANSFER_RESULTS.json').write_text(json.dumps({'prompts_frozen_before_scoring':prompts,'rows':rows},indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(rows,indent=2,ensure_ascii=False))
