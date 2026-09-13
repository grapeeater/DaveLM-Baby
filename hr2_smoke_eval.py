import json,sys,torch
from pathlib import Path
ROOT=Path(r'C:\\DaveLM-CADAVER'); sys.path[:0]=[str(ROOT),r'C:\\DaveLM-v0.9']; sys.path.insert(0,str(ROOT/'fact_supervision_87001_eval_v1'))
from treatment13_model import Treatment13Model
from fact_supervision_87001_eval_v1.PINNED_PILOT1_BINDING_IMPLEMENTATION import OrthoLocalizer,binding_eval
from tokenizers import Tokenizer
CK=ROOT/'human_readiness_hr2_seed87005_v7/run/checkpoint_500.pt'; TOK=Tokenizer.from_file(r'C:\\DaveLM-v0.9\\tokenizer\\v0_7\\davelm_tokenizer.json')
PROMPTS=['The dog ran to the park because','Mia found a red ball and','The small cat sat on the mat. It','A boy opened the box and','Tell me one thing about a sunny day.']
def load(d):
 sd=dict(torch.load(CK,map_location=d,weights_only=True)['model_state_dict']); v=[sd.pop('localizer.'+k) for k in ('u','q','bs','ba')]; m=Treatment13Model(); m.load_state_dict(sd,strict=False); m.localizer=OrthoLocalizer(*v); return m.to(d).eval()
def main():
 d=torch.device('cuda'); m=load(d); rows=[]
 with torch.no_grad():
  for p in PROMPTS:
   ids=[2]+TOK.encode(p).ids; g=list(ids)
   for _ in range(32):
    n=int(m.base_model(torch.tensor([g[-256:]],device=d))[0,-1].argmax()); g.append(n)
    if n==3: break
   rows.append({'prompt':p,'decoded':TOK.decode(g,skip_special_tokens=True),'immediate_eos':len(g)==len(ids)+1 and g[-1]==3,'length':len(g)-len(ids)})
 refs=json.loads((ROOT/'fact_supervision_87001_corrected_v8/BINDING_REFERENCES.json').read_text()); bind={}
 for k in ('pilot0_dev','pilot1_dev'):
  qs=json.loads(Path(refs[k]['path']).read_text())['quartets']; docs=[x for q in qs for x in q['docs']]; z=binding_eval(m,docs,d); bind[k]=z['overall']
 out={'checkpoint_sha256':'pending','prompts':rows,'binding':bind,'sealed_final_accessed':False}; (ROOT/'human_readiness_hr2_seed87005_v7/run/SMOKE_EVAL.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)); print(json.dumps(out,indent=2,ensure_ascii=False))
main()

