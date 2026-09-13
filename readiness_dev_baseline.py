import sys,json
from pathlib import Path
import torch
ROOT=Path(r'C:\DaveLM-CADAVER');sys.path[:0]=[str(ROOT),r'C:\DaveLM-v0.9'];sys.path.insert(0,str(ROOT/'fact_supervision_87001_eval_v1'))
from treatment13_model import Treatment13Model
from PINNED_PILOT1_BINDING_IMPLEMENTATION import OrthoLocalizer
from tokenizers import Tokenizer
CK=ROOT/'language_pilot_1_early_block_protection_seed8380/pilot_run/checkpoints/seed_8380/latest.pt'; B=ROOT/'human_test_readiness_v1';TOK=Tokenizer.from_file(r'C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json')
def load(d):
 sd=dict(torch.load(CK,map_location=d,weights_only=True)['model_state_dict']);v=[sd.pop('localizer.'+k) for k in ('u','q','bs','ba')];m=Treatment13Model();m.load_state_dict(sd,strict=False);m.localizer=OrthoLocalizer(*v);return m.to(d).eval()
def main():
 d=torch.device('cuda');m=load(d);rows=[json.loads(x) for x in (B/'DEV_ITEMS.jsonl').read_text(encoding='utf8').splitlines()];out=[]
 with torch.no_grad():
  for r in rows:
   ids=[2]+TOK.encode(r['prompt']).ids; gen=list(ids)
   for _ in range(32):
    n=int(m.base_model(torch.tensor([gen[-256:]],device=d))[0,-1].argmax());gen.append(n)
    if n==3:break
   out.append({'id':r['id'],'task':r['task'],'prompt':r['prompt'],'decoded':TOK.decode(gen,skip_special_tokens=True),'response_ids':gen[len(ids):],'immediate_eos':len(gen)==len(ids)+1 and gen[-1]==3})
 (B/'DEV_BASELINE_RESULTS.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf8');print({'n':len(out),'immediate_eos':sum(x['immediate_eos'] for x in out),'nonempty':sum(not x['immediate_eos'] for x in out)})
main()
