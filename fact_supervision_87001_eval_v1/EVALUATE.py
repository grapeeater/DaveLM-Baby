import os,sys,json,hashlib,math
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(r'C:\DaveLM-CADAVER'); B=ROOT/'fact_supervision_87001_eval_v1'; sys.path.insert(0,str(ROOT)); sys.path.insert(0,r'C:\DaveLM-v0.9')
from treatment13_model import Treatment13Model
from PINNED_PILOT1_BINDING_IMPLEMENTATION import OrthoLocalizer,binding_eval,lang_eval
TOK=Path(r'C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(path,dev):
 raw=torch.load(path,map_location=dev,weights_only=True); sd=dict(raw['model_state_dict']); m=Treatment13Model(); keys=['u','q','bs','ba']; vals=[sd.pop('localizer.'+k) for k in keys]; m.load_state_dict(sd,strict=False); m.localizer=OrthoLocalizer(*vals); return m.to(dev)
@torch.no_grad()
def item_score(m,row,dev,correct_index):
 cands=row['candidate_token_ids']; out=[]; prefix=[2]+row['prompt_token_ids']
 for c in cands:
  ids=prefix+c+[3]; x=torch.tensor([ids],device=dev); logits=m.base_model(x)[0]
  lp=logits[len(prefix)-1:len(prefix)-1+len(c)+1].log_softmax(-1); idx=torch.tensor(c+[3],device=dev); vals=lp[torch.arange(len(idx),device=dev),idx].detach().cpu().tolist(); out.append({'candidate_ll':sum(vals[:-1]),'eos_ll':sum(vals),'tokens':vals})
 margin=out[0]['candidate_ll']-out[1]['candidate_ll']; signed=margin if correct_index==0 else -margin; return {'id':row['id'],'family_id':row['family_id'],'stratum':row['stratum'],'correct_index':correct_index,'scores':out,'margin':signed,'correct':signed>0,'tie':signed==0}
@torch.no_grad()
def greedy(m,row,dev,correct_index):
 ids=[2]+row['prompt_token_ids']; target=row['candidate_token_ids'][correct_index]+[3]; gen=list(ids)
 for _ in range(32):
  x=torch.tensor([gen[-256:]],device=dev); n=int(m.base_model(x)[0,-1].argmax()); gen.append(n)
  if n==3: break
 return {'exact':gen[len(ids):]==target,'generated_ids':gen[len(ids):]}
def aggregate(rows):
 byf={}
 for r in rows: byf.setdefault(r['family_id'],[]).append(r)
 families=sum(all(x['correct'] for x in v) for v in byf.values()); rev=0; totalrev=0
 for v in byf.values():
  mp={(x.get('assignment'),x.get('query'),x.get('fact_order')):x for x in v}
  for k,x in list(mp.items()):
   y=mp.get((1-k[0],k[1],k[2])) if k[0] in (0,1) else None
   if y is not None: totalrev+=1; rev+=int(x['correct'] and y['correct'])
 return {'items':len(rows),'correct':sum(x['correct'] for x in rows),'accuracy':sum(x['correct'] for x in rows)/len(rows),'ties':sum(x['tie'] for x in rows),'families_complete':families,'families_total':len(byf),'reversal_pairs':totalrev,'reversal_both_correct':rev,'reversal_rate':rev/totalrev if totalrev else None,'mean_margin':sum(x['margin'] for x in rows)/len(rows)}
def main():
 dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); rows=[json.loads(x) for x in (B/'ITEMS.jsonl').read_text().splitlines()]; factual_keys={(r['family_id'],r['assignment'],r['query'],r['fact_order']):r['correct_index'] for r in rows if r['partition']=='primary' and r['arm']=='factual' and r['frame']=='cloze'}
 ck={'Pilot1 parent':ROOT/'language_pilot_1_early_block_protection_seed8380/pilot_run/checkpoints/seed_8380/latest.pt','Factual':ROOT/'fact_supervision_87001_corrected_v8/run_factual/checkpoint_500.pt','Control':ROOT/'fact_supervision_87001_corrected_v8/run_control/checkpoint_500.pt'}
 out=B/'RAW_RESULTS.jsonl'; out.write_text('')
 allres={}
 for name,path in ck.items():
  m=load(path,dev); res=[]
  primary=[r for r in rows if r['partition']=='primary' and r['frame']=='cloze' and (r['arm']=='control' if name=='Control' else r['arm']=='factual')]
  for r in primary:
   ci=factual_keys[(r['family_id'],r['assignment'],r['query'],r['fact_order'])]; z=item_score(m,r,dev,ci); z['greedy']=greedy(m,r,dev,ci); z['checkpoint']=name; res.append(z)
  with out.open('a') as f:
   for z in res:f.write(json.dumps(z)+'\n')
  allres[name]=aggregate(res)
 # binding pools and TinyStories diagnostics
 refs=json.loads((B/'BINDING_REFERENCES.json').read_text()); bind={}
 for nm,key in [('pilot0_dev','pilot0_dev'),('pilot1_dev','pilot1_dev')]: bind[nm]=json.loads(Path(refs[key]['path']).read_text())['quartets']
 for name,path in ck.items():
  m=load(path,dev); allres[name]['binding']={nm:binding_eval(m,[d for q in qs for d in q['docs']],dev)['overall'] for nm,qs in bind.items()}
  allres[name]['greedy_exact']=sum(1 for z in [json.loads(x) for x in out.read_text().splitlines() if json.loads(x).get('checkpoint')==name] for _ in [0] if z['greedy']['exact'])
 (B/'AGGREGATES.json').write_text(json.dumps(allres,indent=2)); print(json.dumps(allres,indent=2))
if __name__=='__main__':main()








