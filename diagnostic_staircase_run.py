"""Literal frozen diagnostic evaluator. Inference only, no optimizer or autograd."""
import json,hashlib,sys,os,math,statistics as st
from pathlib import Path
from collections import defaultdict,Counter
R=Path(r'C:\DaveLM-CADAVER');B=R/'hr3_diagnostic_staircase_v1';O=R/'hr3_diagnostic_staircase_v1_results'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def verify():
 assert sha(B/'FREEZE_RECEIPT.json')==(B/'FREEZE_RECEIPT.sha256').read_text().split()[0]
 assert sha(B/'SHA256SUMS.txt')==read(B/'FREEZE_RECEIPT.json')['manifest_sha256']
 for l in (B/'SHA256SUMS.txt').read_text().splitlines():
  h,p=l.split('  ',1);assert sha(B/p)==h,p
 assert sha(Path(__file__))==sha(B/'EVALUATOR.py')
def summary(rs):
 pairs=defaultdict(list); fam=defaultdict(list)
 for r in rs:
  pairs[r['pair_id']].append(r);fam[r['family_id']].append(r)
 assert all(len(v)==2 for v in pairs.values())
 return {'items':len(rs),'correct':sum(r['correct'] for r in rs),'ties':sum(r['margin']==0 for r in rs),'exact_answer_prefix':sum(r['exact_answer_prefix'] for r in rs),'exact_answer_then_eos':sum(r['exact_then_eos'] for r in rs),'full_vocab_first_top1_correct':sum(r['full_top1_correct'] for r in rs),'first_top1_eos':sum(r['top1_id']==3 for r in rs),'first_top1_outside_candidates':sum(r['top1_id'] not in r['candidate_first_ids'] for r in rs),'reversals':sum(all(r['correct'] for r in v) for v in pairs.values()),'reversal_pairs':len(pairs),'complete_families':sum(all(r['correct'] for r in v) for v in fam.values()),'family_count':len(fam),'margin':{'mean':st.mean(r['margin'] for r in rs),'min':min(r['margin'] for r in rs),'median':st.median(r['margin'] for r in rs),'max':max(r['margin'] for r in rs)},'first_logit_margin_mean':st.mean(r['first_logit_margin'] for r in rs),'first_candidate_mass_mean':st.mean(r['first_candidate_mass'] for r in rs),'sequence_candidate_mass_mean':st.mean(r['sequence_candidate_mass'] for r in rs),'eos_probability_mean':st.mean(r['eos_probability'] for r in rs),'families':{f:{'correct':sum(r['correct'] for r in v),'total':len(v)} for f,v in fam.items()}}
def main():
 verify();p=read(B/'PROTOCOL.json');sys.path.insert(0,str(Path(p['runtime_bundle'])/'sources'))
 import hr3_block3_runtime as rt
 import torch
 from tokenizers import Tokenizer
 rt.verify_integrity(Path(p['runtime_bundle']));rt.configure_runtime(87006);device=torch.device('cuda')
 assert sha(Path(p['tokenizer']))==p['tokenizer_sha256']; tok=Tokenizer.from_file(p['tokenizer'])
 for c in p['checkpoints'].values():assert sha(Path(c['path']))==c['sha256']
 O.mkdir(exist_ok=True); provenance={'bundle_receipt':sha(B/'FREEZE_RECEIPT.json'),'source':sha(Path(__file__)),'runtime':{'python':sys.version,'torch':torch.__version__},'checkpoints':p['checkpoints']}
 if (O/'PROVENANCE.json').exists():assert read(O/'PROVENANCE.json')==provenance
 else:rt.atomic_json(provenance,O/'PROVENANCE.json')
 out=O/'RAW.jsonl';done={}
 if out.exists():
  for l in out.read_text(encoding='utf-8').splitlines():
   r=json.loads(l);key=(r['checkpoint'],r['stage'],r['id']);assert key not in done;done[key]=r
 def persist(r):
  with out.open('a',encoding='utf-8',newline='\n') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
  done[(r['checkpoint'],r['stage'],r['id'])]=r
 registry=list(p['checkpoints']);registry.remove('Factual87002');registry=['Factual87002']+registry
 for name in registry:
  cp=Path(p['checkpoints'][name]['path']);model=rt.load_model(cp,device,Path(p['runtime_bundle'])).eval()
  stages=[('single_fact',read(B/'SINGLE_FACT_ITEMS.json'))]
  if name=='Factual87002':stages=[('training',read(B/'FACTUAL_TRAIN.json')),('factual_dev',read(B/'FACTUAL_DEV.json'))]+stages
  with torch.inference_mode():
   for stage,rs in stages:
    for i,r in enumerate(rs):
     if (name,stage,r['id']) in done:continue
     prefix=[2]+r['prompt_token_ids'];ci=r['correct_index'];scores=[];details=[];first=None
     for cand in r['candidate_token_ids']:
      seq=prefix+cand+[3];logits=model.base_model(torch.tensor([seq],device=device))[0];lp=logits.log_softmax(-1);start=len(prefix)-1
      ts=[]
      for j,t in enumerate(cand+[3]):
       z=logits[start+j];ts.append({'token_id':t,'log_probability':float(lp[start+j,t]),'logit':float(z[t]),'full_vocab_rank':1+int((z>z[t]).sum())})
      scores.append(sum(t['log_probability'] for t in ts[:-1]));details.append(ts)
      if first is None:
       z=logits[start];fids=[c[0] for c in r['candidate_token_ids']];assert len(set(fids))==2
       first={'top1_id':int(z.argmax()),'candidate_first_ids':fids,'full_top1_correct':int(z.argmax())==fids[ci],'first_logit_margin':float(z[fids[ci]]-z[fids[1-ci]]),'first_candidate_mass':float(lp[start,fids].exp().sum()),'eos_probability':float(lp[start,3].exp()),'eos_rank':1+int((z>z[3]).sum())}
     gen=rt.greedy_ids(model,r['prompt_token_ids'],device,32)[len(prefix):]; target=r['candidate_token_ids'][ci]; margin=scores[ci]-scores[1-ci]
     pair=r.get('pair_id',f"{r['family_id']}:q{r['query']}:o{r['fact_order']}")
     persist({'checkpoint':name,'stage':stage,'id':r['id'],'family_id':r['family_id'],'pair_id':pair,'assignment':r['assignment'],'stratum':r['stratum'],'prompt':r['prompt'],'candidates':r['candidates'],'correct_index':ci,'candidate_scores':scores,'token_details':details,'margin':margin,'correct':margin>0,'sequence_candidate_mass':sum(math.exp(x) for x in scores),'generated_ids':gen,'generated_text':tok.decode(gen,skip_special_tokens=True),'exact_answer_prefix':gen[:len(target)]==target,'exact_then_eos':gen==target+[3],**first})
     if (i+1)%96==0:print(name,stage,i+1,flush=True)
    print(name,stage,'complete',flush=True)
  assert sha(cp)==p['checkpoints'][name]['sha256'];del model;torch.cuda.empty_cache()
 groups=defaultdict(list)
 for r in done.values():groups[(r['checkpoint'],r['stage'])].append(r)
 agg={}
 for (name,stage),rs in groups.items():
  val=summary(rs); strata=defaultdict(list)
  for r in rs:strata[r['stratum']].append(r)
  val['strata']={k:summary(v) for k,v in strata.items()}
  if stage=='single_fact':
   clear=val['correct']>=56 and val['reversals']>=24 and all(v['correct']>=12 for v in val['strata'].values())
   chance=28<=val['correct']<=36 and val['reversals']<=4
   val['prospective_flag']='CLEAR_ON_BATTERY' if clear else 'APPROXIMATELY_CHANCE' if chance else 'INCONCLUSIVE_OR_MIXED'
  agg.setdefault(name,{})[stage]=val
 rt.atomic_json(agg,O/'RESULTS.json');verify()
 rt.atomic_json({'status':'STAIRCASE_COMPLETE','checkpoint_hashes_reverified':True,'optimizer_created':False,'weight_updates':0,'final_accessed':False,'sacred_accessed':False,'raw_sha256':sha(out)},O/'COMPLETION.json')
 print(json.dumps({n:{s:{k:v[k] for k in ['items','correct','reversals','reversal_pairs','complete_families','exact_answer_then_eos','full_vocab_first_top1_correct']} for s,v in ss.items()} for n,ss in agg.items()},indent=2))
if __name__=='__main__':main()
