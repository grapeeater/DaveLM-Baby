from pathlib import Path
import json,sys,torch
sys.path.insert(0,r'C:\DaveLM-CADAVER');sys.path.insert(0,r'C:\DaveLM-v0.9')
from language_storybound_p3 import OrthoLocalizer
from treatment13_model import Treatment13Model
from tokenizers import Tokenizer
P=Path(__file__).resolve().parent;CK=P/'latest.pt';TOK=Tokenizer.from_file(r'C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json');raw=torch.load(CK,map_location='cpu',weights_only=True);sd=raw['model_state_dict'];m=Treatment13Model();m.localizer=OrthoLocalizer(*[sd['localizer.'+n].clone() for n in ('u','q','bs','ba')]);m.load_state_dict(sd,strict=True);m.to('cuda:0').eval();prompts=['Once upon a time, there was a little girl named Zoe.','Once upon a time, there was a little boy named Alex.','A cat found a blue ball.','A dog found a red hat.','Mia went to the park with her mother.','Leo saw a bird in the yard.','The little girl dropped her book.','Sam was sad because his toy broke.'];g=torch.Generator(device='cuda').manual_seed(8380);rows=[]
for p in prompts:
 ids=[2]+TOK.encode(p).ids
 with torch.no_grad():
  for _ in range(40):
   logits=m.base_model(torch.tensor([ids[-256:]],device='cuda'))[:,-1,:].float()/0.8; probs=torch.softmax(logits,dim=-1);n=int(torch.multinomial(probs,1,generator=g));ids.append(n)
   if n==3:break
 rows.append({'prompt':p,'sampled':TOK.decode(ids,skip_special_tokens=True)})
(P/'SAMPLED_PANEL.json').write_text(json.dumps({'temperature':0.8,'seed':8380,'rows':rows},indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(rows,indent=2,ensure_ascii=False))
