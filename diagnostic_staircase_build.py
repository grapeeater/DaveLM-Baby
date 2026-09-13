import json,hashlib,itertools,re,shutil
from pathlib import Path
from collections import Counter,defaultdict
from tokenizers import Tokenizer
R=Path(r'C:\DaveLM-CADAVER'); O=R/'hr3_diagnostic_staircase_v1'; V=R/'fact_supervision_87001_corrected_v8'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l]
def write(n,v):
 p=O/n;assert not p.exists();p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
def chain(p):
 assert sha(p/'FREEZE_RECEIPT.json')==(p/'FREEZE_RECEIPT.sha256').read_text().split()[0]
 rec=read(p/'FREEZE_RECEIPT.json');assert sha(p/'SHA256SUMS.txt')==rec['manifest_sha256']
 for line in (p/'SHA256SUMS.txt').read_text().splitlines():
  h,n=line.split('  ',1); assert sha(p/n)==h,n
def validate(r):
 fact,cue=r['prompt'].split('\n'); a=re.fullmatch(r'(Alex|Owen|Mia|Nora) (found|carried) the (red ball|blue book)\.',fact)
 b=re.fullmatch(r'The (red ball|blue book) was (found|carried) by (Alex|Owen|Mia|Nora)\.',fact)
 assert bool(a)^bool(b)
 name,pred,obj=a.groups() if a else (b[3],b[2],b[1])
 q=re.fullmatch(r'Who (found|carried) the (red ball|blue book)\? Answer:',cue)
 z=re.fullmatch(r'The person who (found|carried) the (red ball|blue book) was',cue)
 assert bool(q)^bool(z); assert (q or z).groups()==(pred,obj)
 assert not any(n in cue for n in ['Alex','Owen','Mia','Nora'])
 assert r['candidates'][r['correct_index']]==' '+name+'.'
 return name,pred,obj
def main():
 O.mkdir(exist_ok=False);chain(V)
 ck={
 'Pilot1':(R/'language_pilot_1_early_block_protection_seed8380/pilot_run/checkpoints/seed_8380/latest.pt','2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb'),
 'HR1_aligned':(R/'human_readiness_hr1_causal_aligned_seed87006_execution/run_factual/checkpoint_500.pt','eec87f940822938911e34126ba855630473c94867f3e2d267a97e3d288b5652d'),
 'HR3':(R/'human_readiness_hr3_causal_seed87006_execution_v7/checkpoint_500.pt','23726ce76ff789bbe05d44340952a1763b7231584b0c91ff8e15f5c64e6ef93d'),
 'Factual87002':(V/'run_factual/checkpoint_500.pt','3ae30ae847d3129d7715173ddbfbbaf6b7490772dc850e361a5a2d785b991123')}
 for p,h in ck.values():assert sha(p)==h
 pilot0=R/'language_pilot_0_tinystories_seed8380/pilot_run/checkpoints/seed_8380/latest.pt';assert sha(pilot0)=='769bd01efd28e7888064c0d5fe1dd0d85a4344e2039aef04bcbf7d9e906f47f5'
 rec0=R/'language_pilot_0_tinystories_seed8380/pilot_run/RESULTS.json';rec1=R/'language_pilot_1_early_block_protection_seed8380/pilot_run/RESULTS.json'
 assert read(rec0)['final_checkpoint_sha256']==sha(pilot0); assert read(rec1)['final_checkpoint_sha256']==ck['Pilot1'][1]
 write('PARENT_RECONCILIATION.json',{'Pilot0':{'path':str(pilot0),'hash':sha(pilot0),'record':str(rec0),'record_hash':sha(rec0)},'Pilot1':{'path':str(ck['Pilot1'][0]),'hash':ck['Pilot1'][1],'record':str(rec1),'record_hash':sha(rec1)},'both_starting_hashes':[read(p)['starting_checkpoint_sha256'] for p in [rec0,rec1]],'finding':'769bd01 is Pilot0; 2281d20 is Pilot1. They are distinct descendants of the same Graduate parent, not two versions of Pilot1.'})
 tokpath=Path(r'C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json');assert sha(tokpath)=='e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b'; tok=Tokenizer.from_file(str(tokpath))
 allrows=rows(V/'ITEMS.jsonl'); train=[r for r in allrows if r['partition']=='train' and r['arm']=='factual'];dev=[r for r in allrows if r['partition']=='dev' and r['arm']=='factual']
 assert len(train)==384 and len(dev)==96
 for split,rs in [('train',train),('dev',dev)]:
  fam=defaultdict(list)
  for r in rs:fam[r['family_id']].append(r)
  assert all(len(v)==8 and {(r['assignment'],r['query'],r['fact_order']) for r in v}==set(itertools.product(range(2),repeat=3)) for v in fam.values())
  write('FACTUAL_'+split.upper()+'.json',rs)
 from importlib.util import spec_from_file_location,module_from_spec
 sp=spec_from_file_location('v8mask',V/'harness.py'); hm=module_from_spec(sp);sp.loader.exec_module(hm)
 for r in train+dev:
  x,y=hm.prepare_example(r);seq=[2]+r['prompt_token_ids']+r['candidate_token_ids'][r['correct_index']]+[3]
  assert x==seq[:-1] and len(y)==len(x)
  assert [(i,t) for i,t in enumerate(y) if t!=-100]==[(len(r['prompt_token_ids'])+j,t) for j,t in enumerate(r['candidate_token_ids'][r['correct_index']]+[3])]
 single=[]
 for pairi,names in enumerate([('Alex','Owen'),('Mia','Nora')]):
  for pred in ['found','carried']:
   fid=f'SF:{pairi}:{pred}'
   for obj,assignment,voice,cuekind in itertools.product(['red ball','blue book'],range(2),['active','passive'],['cloze','qa']):
    name=names[assignment]; fact=f'{name} {pred} the {obj}.' if voice=='active' else f'The {obj} was {pred} by {name}.'
    cue=f'The person who {pred} the {obj} was' if cuekind=='cloze' else f'Who {pred} the {obj}? Answer:'
    prompt=fact+'\n'+cue; cands=[' '+n+'.' for n in names]
    r={'id':f'{fid}:{obj}:{voice}:{cuekind}:a{assignment}','family_id':fid,'assignment':assignment,'query':obj,'fact_order':voice,'frame':cuekind,'stratum':voice+':'+cuekind,'prompt':prompt,'prompt_token_ids':tok.encode(prompt).ids,'candidates':cands,'candidate_token_ids':[tok.encode(c).ids for c in cands],'correct_index':assignment,'pair_id':f'{fid}:{obj}:{voice}:{cuekind}'}
    validate(r); single.append(r)
 assert len(single)==64 and len({r['prompt'].casefold() for r in single})==64
 for r in single+train+dev:
  assert all(len(c)==4 for c in r['candidate_token_ids'])
  for c,ids in zip(r['candidates'],r['candidate_token_ids']):
   assert tok.encode(r['prompt']+c).ids==r['prompt_token_ids']+ids
   assert tok.decode(r['prompt_token_ids']+ids)==r['prompt']+c
   assert 1+len(r['prompt_token_ids'])+len(ids)+1<=256
 neg={}
 for label,change in [('wrong_key',lambda r:r.update(correct_index=1-r['correct_index'])),('wrong_predicate',lambda r:r.update(prompt=r['prompt'].replace('Who found','Who carried'))),('missing_object',lambda r:r.update(prompt=r['prompt'].replace('Who found the red ball','Who found the blue book'))),('extra_fact',lambda r:r.update(prompt=r['prompt'].replace('\n',' Owen went home.\n')))]:
  base=next(r for r in single if r['frame']=='qa' and 'found the red ball?' in r['prompt']); r=dict(base);change(r)
  try:validate(r);raise RuntimeError('negative test accepted '+label)
  except AssertionError:neg[label]='rejected'
 write('SINGLE_FACT_ITEMS.json',single)
 # Explicit nonsacred allowlist; never open sealed final material for leakage checks.
 auditpaths=[R/'language_pilot_1_early_block_protection_seed8380/language_train.jsonl',R/'language_pilot_1_early_block_protection_seed8380/language_dev.jsonl',V/'ITEMS.jsonl',R/'human_test_readiness_v2_seed87010/DEV_FACTS.jsonl',R/'human_test_readiness_v2_seed87010/DEV_GENERATION.jsonl',R/'human_test_readiness_v2_seed87010/DEV_INSTRUCTIONS.jsonl',R/'human_test_readiness_v2_seed87010/DEV_CONTINUITY.jsonl']
 norm=lambda x:' '.join(x.casefold().split()); audits={}
 for p in auditpaths:
  source=rows(p);texts=[r.get('text',r.get('prompt','')) for r in source]; normalized=[norm(t) for t in texts]
  hits=[r['id'] for r in single if any(norm(r['prompt']) in t for t in normalized)]
  exact=[r['id'] for r in single if norm(r['prompt']) in set(normalized)]
  audits[str(p)]={'sha256':sha(p),'complete_prompt_substring_hits':hits,'whole_record_exact_hits':exact}
  assert not exact,(str(p),exact)
 write('PREFLIGHT.json',{'passed':True,'single_items':64,'families':4,'items_per_family':16,'reversal_pairs':32,'items_per_voice_cue':16,'candidate_lengths':[4],'correct_names':dict(Counter(r['candidates'][r['correct_index']] for r in single)),'correct_index':dict(Counter(r['correct_index'] for r in single)),'position_control':'active actor first, passive actor after object; equal counts. One fact intentionally contains only the correct candidate. Absent alternative and identity copying are limitations, not two-fact binding evidence.','query_leakage':False,'independent_final_text_parse':'PASS','negative_tests':neg,'training_alignment_checks':len(train+dev),'overlap_audit':audits,'limits':'Known familiar vocabulary; full prompts distinct from allowlisted sources. Constituent facts, semantics, and templates can overlap; no unseen-vocabulary/general-transfer claim. FINAL excluded.'})
 protocol={'status':'FROZEN_DIAGNOSTIC_STAIRCASE','stage1':'Evaluate factual87002 on all 384 factual training rows, plus 96 factual DEV rows as diagnostic. No control prompts or Confirmation scoring. No suitable frozen training evaluator exists; preserve training target semantics via validated harness alignment.',
 'stage2':'64 single-fact actor-retrieval items, 4 families of 16, 32 actor reversal pairs; two name pairs, two predicates, two objects, active/passive fact wording, cloze/QA cues. No random sampling.',
 'checkpoints':{n:{'path':str(p),'sha256':h} for n,(p,h) in ck.items()},'runtime_bundle':str(R/'human_readiness_hr3_block3_causal_seed87006_v7'),'tokenizer':str(tokpath),'tokenizer_sha256':sha(tokpath),
 'scoring':{'candidate':'sum all 4 candidate log probabilities, EOS excluded; positive signed margin wins, zero fails; no length normalization or prior subtraction','exact_answer':'raw greedy first four tokens equal correct candidate; separately exact candidate then immediate EOS; greedy max32 normal EOS','full_vocab_top1':'first answer position argmax equals correct first candidate token; distinct first token IDs make signed correct-vs-alternative first-logit margin meaningful','mass':'first-step candidate-token mass and full four-token candidate-sequence mass are both absolute probabilities, not candidate renormalization','reversal':'both members correct; every pair counted once','families':'all members correct','raw':'save per-token full-vocabulary ranks, log probabilities, logits; EOS probability/rank; all greedy IDs/text; persist each item before aggregation'},
 'diagnostic_flags':{'clearly_acquired_on_this_battery':'per checkpoint: restricted correct >=56/64, reversal >=24/32, and >=12/16 restricted correct in EACH active/passive x cloze/QA stratum. These are prospective descriptive staircase flags, not readiness gates or inferential population claims.','approximately_chance':'28..36/64 restricted correct and <=4/32 successful reversals','otherwise':'inconclusive or mixed; no automatic two-fact objective-treatment authorization','mixed_models':'acquisition in factual descendant cannot establish acquisition in Pilot1 or HR descendants; report separately and stop for parent/next-diagnostic decision','uncertainty':'report raw counts and family/format profiles, no IID item CI due dependence and only four lexical families'},
 'future_ab_constraints':'Corrected-HR1/Pilot1 mask blocks0-3 frozen, identical arms; strict counterbalancing; full-vocabulary answer CE retained, never naked margin; candidate membership preserved; any coefficient calibrated once on deterministic training-only step0 gradient norms before DEV; freeze paired binary/reversal statistics and effect sizes; readiness gates unchanged.',
 'no_optimizer':True,'no_training':True,'final_access':False,'sacred_access':False}
 write('PROTOCOL.json',protocol)
 write('RECONSTRUCTION.json',{'v8_receipt_sha256':sha(V/'FREEZE_RECEIPT.json'),'training_rows_per_arm':384,'training_families_per_arm':48,'object_families':24,'predicate_families':24,'family_size':8,'family_transformations':'assignment x query x fact order, 2x2x2','format':'two-fact declarative cloze; factual entailed; control omits queried description and presents each identical prefix with both targets','parent':protocol['checkpoints']['Pilot1'],'training':'500 updates, 450 English/50 binding, 9:1; 32 English records per batch (2 complete families from each stratum); AdamW lr5e-5 wd0.05 betas0.9/0.999 eps1e-8 clip2; blocks0-3 protected on English, full T13 on binding','objective':'four answer tokens plus final EOS, context labels ignored, correct causal alignment; pinned binding CE + hard-min localizer','dev500':read(V/'run_factual/DEV_500.json'),'dev_limitation':'This file stores no measured DEV score. No per-update loss file exists in preserved run_factual directory. Checkpoint metadata will be recorded mechanically at inference load.','historical_acquisition_report':str(R/'fact_supervision_87001_eval_v1/REPORT.md'),'hr3_distinction':'HR3 data are isolated TinyStories sentences, not two-fact QA. V8 is a different explicit two-fact cloze treatment.'})
 shutil.copyfile(Path(__file__),O/'BUILD_SOURCE.py');shutil.copyfile(R/'diagnostic_staircase_run.py',O/'EVALUATOR.py');shutil.copyfile(V/'harness.py',O/'PINNED_TRAINING_HARNESS.py')
 payload=sorted(O.iterdir());(O/'SHA256SUMS.txt').write_text(''.join(sha(p)+'  '+p.name+'\n' for p in payload),encoding='utf-8',newline='\n')
 write('FREEZE_RECEIPT.json',{'status':'FROZEN_DIAGNOSTIC_STAIRCASE','manifest_sha256':sha(O/'SHA256SUMS.txt'),'model_behavior_accessed_during_build':False})
 (O/'FREEZE_RECEIPT.sha256').write_text(sha(O/'FREEZE_RECEIPT.json')+'  FREEZE_RECEIPT.json\n',encoding='utf-8')
 for p in O.iterdir():p.chmod(0o444)
 chain(O);print(json.dumps({'receipt':sha(O/'FREEZE_RECEIPT.json'),'manifest':sha(O/'SHA256SUMS.txt'),'preflight':'PASS'}))
if __name__=='__main__':main()
