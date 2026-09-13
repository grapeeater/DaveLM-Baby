import sys,json,math
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(r'C:\DaveLM-CADAVER'); B=ROOT/'fact_supervision_87001_eval_v1'; sys.path.insert(0,str(ROOT));sys.path.insert(0,r'C:\DaveLM-v0.9');sys.path.insert(0,str(B))
from treatment13_model import Treatment13Model
from PINNED_PILOT1_BINDING_IMPLEMENTATION import OrthoLocalizer,lang_eval,generate
from tokenizers import Tokenizer
TOK=Path(r'C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json')
def load(p,dev):
 sd=dict(torch.load(p,map_location=dev,weights_only=True)['model_state_dict']); vals=[sd.pop('localizer.'+k) for k in ('u','q','bs','ba')]; m=Treatment13Model();m.load_state_dict(sd,strict=False);m.localizer=OrthoLocalizer(*vals);return m.to(dev)
def main():
 dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); tok=Tokenizer.from_file(str(TOK)); src=ROOT/'language_pilot_1_early_block_protection_seed8380'; tr=[json.loads(x) for x in (src/'language_train.jsonl').read_text(encoding='utf-8').splitlines()]; dv=[json.loads(x) for x in (src/'language_dev.jsonl').read_text(encoding='utf-8').splitlines()]; enc=lambda a:torch.tensor(tok.encode_batch([x['text'] for x in a])[0].ids if False else [i for x in a for i in tok.encode(x['text']).ids],dtype=torch.long)
 stream=torch.tensor([i for x in dv for i in tok.encode(x['text']).ids],dtype=torch.long); prompts=[json.loads(x)['prompt'] for x in (B/'NATURALISTIC.jsonl').read_text(encoding='utf-8').splitlines()]
 ck={'Pilot1 parent':ROOT/'language_pilot_1_early_block_protection_seed8380/pilot_run/checkpoints/seed_8380/latest.pt','Factual':ROOT/'fact_supervision_87001_corrected_v8/run_factual/checkpoint_500.pt','Control':ROOT/'fact_supervision_87001_corrected_v8/run_control/checkpoint_500.pt'}; out={}
 for n,p in ck.items():
  m=load(p,dev); loss=lang_eval(m,stream,dev,88003); gens=[]
  for q in prompts: gens.append(generate(m,tok,q))
  out[n]={'loss':loss,'perplexity':math.exp(loss),'naturalistic':gens,'immediate_eos':sum(g['stopped_on_eos'] and len(g['generated_ids'])==1 for g in gens)}
 (B/'DIAGNOSTICS.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:{'loss':v['loss'],'perplexity':v['perplexity'],'immediate_eos':v['immediate_eos']} for k,v in out.items()},indent=2))
if __name__=='__main__':main()

