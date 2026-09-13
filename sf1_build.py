import json,hashlib,random,re,itertools,shutil,sys,importlib.util
from pathlib import Path
from collections import Counter,defaultdict
R=Path(r'C:\DaveLM-CADAVER');O=R/'single_fact_acquisition_sf1_seed87011';H=R/'human_readiness_hr3_block3_causal_seed87006_v7';V=R/'fact_supervision_87001_corrected_v8'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(n,v):
 p=O/n;assert not p.exists();p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
def validate(r):
 context,cue=r['prompt'].split('\n');facts=re.findall(r'(Alex|Owen|Mia|Nora) (found|carried) the ([^.]+)\.',context)
 passive=re.fullmatch(r'The ([^.]+) was (found|carried) by (Alex|Owen|Mia|Nora)\.',context)
 if passive:facts=[(passive[3],passive[2],passive[1])]
 card=re.findall(r'(?:A card for the [^.]+ says|A card says) (Alex|Owen|Mia|Nora)\.',context)
 if cue=='Copy the name on the card:':assert len(card)==1;answer=card[0]
 else:
  q=re.fullmatch(r'The person who (found|carried) the (.+) was',cue) or re.fullmatch(r'Who (found|carried) the (.+)\? Answer:',cue)
  assert q and len(facts)==1;assert facts[0][1:]==q.groups();answer=facts[0][0]
 assert r['candidates'][r['correct_index']]==' '+answer+'.'
 assert not any(n in cue for n in ['Alex','Owen','Mia','Nora'])
 # Exhaustively consume legal clauses so extra/unrecognized assertions fail.
 legal=[]
 if passive:legal=[passive[0]]
 else:
  legal=re.findall(r'(?:Alex|Owen|Mia|Nora) (?:found|carried) the [^.]+\.|A card(?: for the [^.]+)? says (?:Alex|Owen|Mia|Nora)\.',context)
 assert ' '.join(legal)==context
 return answer
def main():
 O.mkdir(exist_ok=False);(O/'sources').mkdir();(O/'data').mkdir()
 for name in ['hr3_block3_runtime.py','treatment13_model.py','treatment13_config.py','PINNED_PILOT1_BINDING_IMPLEMENTATION.py']:shutil.copyfile(H/'sources'/name,O/'sources'/name)
 shutil.copyfile(V/'harness.py',O/'sources/PINNED_MASKING.py')
 for name in ['binding_rehearsal.json','binding_dev_pilot0.json','binding_dev_pilot1.json','ENGLISH_DEV.jsonl']:shutil.copyfile(H/'data'/name,O/'data'/name)
 shutil.copyfile(R/'sf1_engine.py',O/'ENGINE.py');shutil.copyfile(Path(__file__),O/'BUILD.py')
 sys.path.insert(0,str(O/'sources'));import hr3_block3_runtime as rt
 rt.verify_integrity(H)
 from tokenizers import Tokenizer
 tokpath=Path(r'C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json');assert sha(tokpath)==rt.TOKENIZER_SHA256;tok=Tokenizer.from_file(str(tokpath))
 parent=R/'language_pilot_1_early_block_protection_seed8380/pilot_run/checkpoints/seed_8380/latest.pt';assert sha(parent)==rt.PARENT_SHA256
 panels={n:[] for n in ['TRAIN','HELDOUT','ALTERNATE','COPY','COMPETING']};pairs=[('Alex','Owen'),('Mia','Nora')];objects=[['small drum','wooden boat'],['soft scarf','round plate']]
 def add(panel,fid,names,obj,pred,a,prompt,sub,pairkey,ci=None):
  cand=[' '+n+'.' for n in names];r={'id':f'{panel}:{pairkey}:a{a}','family_id':panel+':'+fid,'pair_id':panel+':'+pairkey,'subgroup':sub,'arm':'factual','prompt':prompt,'prompt_token_ids':tok.encode(prompt).ids,'candidates':cand,'candidate_token_ids':[tok.encode(c).ids for c in cand],'correct_index':a if ci is None else ci,'assignment':a,'actor':names[a],'object':obj,'predicate':pred}
  validate(r);panels[panel].append(r)
 for g,names in enumerate(pairs):
  for pred in ['found','carried']:
   fid=f'g{g}:{pred}'
   for panel,objs in [('TRAIN',objects[g]),('HELDOUT',objects[1-g])]:
    for obj,a in itertools.product(objs,range(2)):
     fact=f'{names[a]} {pred} the {obj}.';cue=f'The person who {pred} the {obj} was'
     add(panel,fid,names,obj,pred,a,fact+'\n'+cue,'familiar_cloze',f'{fid}:{obj}')
     if panel=='HELDOUT':
      for voice,c in [('passive','cloze'),('active','qa'),('passive','qa')]:
       fact2=fact if voice=='active' else f'The {obj} was {pred} by {names[a]}.'
       cue2=cue if c=='cloze' else f'Who {pred} the {obj}? Answer:'
       add('ALTERNATE',fid+':'+voice+':'+c,names,obj,pred,a,fact2+'\n'+cue2,voice+':'+c,f'{fid}:{obj}:{voice}:{c}')
     else:
      if pred=='found':add('COPY',f'g{g}',names,obj,pred,a,f'A card for the {obj} says {names[a]}.\nCopy the name on the card:','isolated_card_copy',f'g{g}:{obj}')
      for order,c in itertools.product(range(2),['fact','copy']):
       card=f'A card says {names[1-a]}.';context=fact+' '+card if order==0 else card+' '+fact
       add('COMPETING',fid+f':o{order}:{c}',names,obj,pred,a,context+'\n'+(cue if c=='fact' else 'Copy the name on the card:'),f'fact_order{order}:{c}',f'{fid}:{obj}:o{order}:{c}',a if c=='fact' else 1-a)
 expected={'TRAIN':16,'HELDOUT':16,'ALTERNATE':48,'COPY':8,'COMPETING':64}
 allrows=sum(panels.values(),[]);assert len({r['prompt'] for r in allrows})==len(allrows)
 lengthmax=0
 for n,rs in panels.items():
  assert len(rs)==expected[n];pf=defaultdict(list)
  for r in rs:
   pf[r['pair_id']].append(r);assert all(len(x)==4 for x in r['candidate_token_ids'])
   for c,ids in zip(r['candidates'],r['candidate_token_ids']):
    assert tok.encode(r['prompt']+c).ids==r['prompt_token_ids']+ids;assert tok.decode(r['prompt_token_ids']+ids)==r['prompt']+c
   lengthmax=max(lengthmax,len(r['prompt_token_ids'])+6)
  for pp in pf.values():
   assert len(pp)==2 and {x['correct_index'] for x in pp}=={0,1};assert len(pp[0]['prompt_token_ids'])==len(pp[1]['prompt_token_ids'])
  assert Counter(r['correct_index'] for r in rs)=={0:len(rs)//2,1:len(rs)//2}
  assert len(set(Counter(r['candidates'][r['correct_index']] for r in rs).values()))==1
  write(n+'.json',rs)
 assert lengthmax<=256
 trcomb={(r['actor'],r['object']) for r in panels['TRAIN']};tecomb={(r['actor'],r['object']) for r in panels['HELDOUT']};assert not trcomb&tecomb
 neg={}
 for n in ['wrong_key','wrong_predicate','wrong_object','extra_claim']:
  r=dict(panels['TRAIN'][0])
  if n=='wrong_key':r['correct_index']=1-r['correct_index']
  elif n=='wrong_predicate':r['prompt']=r['prompt'].replace('who found','who carried')
  elif n=='wrong_object':r['prompt']=r['prompt'].replace('who found the small drum','who found the wooden boat')
  else:r['prompt']=r['prompt'].replace('\n',' Owen went home.\n')
  try:validate(r)
  except AssertionError:neg[n]='REJECTED';continue
  raise AssertionError(n)
 # Freeze literal 200-update schedule; two presentations of every training record per English batch.
 rng=random.Random(87012);brng=random.Random(87013);qs=read(O/'data/binding_rehearsal.json')['quartets'];assert len(qs)==80 and all(len(q['docs'])==4 for q in qs)
 placements=[]
 for _ in range(2):q=list(range(80));brng.shuffle(q);placements+=q
 schedule=[];bi=0
 for u in range(1,201):
  if u%10:
   ids=sorted(r['id'] for r in panels['TRAIN'])*2;rng.shuffle(ids)
   schedule.append({'update':u,'kind':'english','ids':ids,'pad':max(len(r['prompt_token_ids'])+6 for r in panels['TRAIN'])})
  else:
   ix=placements[bi*8:bi*8+8];bi+=1;schedule.append({'update':u,'kind':'binding','quartets':ix,'documents':[d['doc_id'] for i in ix for d in qs[i]['docs']]})
 write('SCHEDULE.json',schedule)
 # Outcome-blind decoded-text audit. No FINAL/sacred material is included.
 auditpaths=[R/'language_pilot_1_early_block_protection_seed8380/language_train.jsonl',R/'language_pilot_1_early_block_protection_seed8380/language_dev.jsonl',V/'ITEMS.jsonl',R/'hr3_diagnostic_staircase_v1/SINGLE_FACT_ITEMS.json']+[R/'human_test_readiness_v2_seed87010'/n for n in ['DEV_FACTS.jsonl','DEV_GENERATION.jsonl','DEV_INSTRUCTIONS.jsonl','DEV_CONTINUITY.jsonl']]
 audit={};norm=lambda x:' '.join(x.casefold().split())
 for path in auditpaths:
  ar=read(path) if path.suffix=='.json' else [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l]
  texts=[norm(r.get('text',r.get('prompt',''))) for r in ar]
  exact=[r['id'] for r in allrows if norm(r['prompt']) in set(texts)]
  substring=[r['id'] for r in allrows if any(norm(r['prompt']) in t for t in texts)]
  assert not exact and not substring,(path,exact,substring)
  audit[str(path)]={'sha256':sha(path),'whole_prompt_exact':exact,'whole_prompt_substring':substring}
 external={str(Path(m.__file__).resolve()):sha(Path(m.__file__)) for m in list(sys.modules.values()) if getattr(m,'__file__',None) and str(Path(m.__file__).resolve()).startswith(r'C:\DaveLM-v0.9') and Path(m.__file__).is_file()}
 write('PROTOCOL.json',{'name':'SF1','seed':87011,'parent':str(parent),'parent_sha256':rt.PARENT_SHA256,'tokenizer':str(tokpath),'tokenizer_sha256':rt.TOKENIZER_SHA256,'external_hashes':external,
 'hypothesis':'Repeated balanced single-fact answer supervision is sufficient to acquire the trained ordinary-English-to-name response interface while retaining nonsacred binding. This is an engineering acquisition test, not readiness or a discrimination-loss comparison.',
 'rationale':'Earlier two-fact answer CE failed even on its own training distribution; isolated-sentence HR runs had no explicit factual training. Reduce contextual selection complexity to one actor and one relation, retain correct response CE and Pilot1 scope. Alternative objectives/scopes are not manipulated.',
 'curriculum':'16 training records, four complete actor-pair/relation families, two objects x two actor assignments each; one familiar active-voice declarative cloze. 180 English updates, 32 records each, every training record twice per update (360 presentations each). No separate sentence-training mixture; measure language cost explicitly.',
 'schedule':'200 updates, 20 cycles of nine factual English then one binding. Frozen English order seed87012; binding quartet order seed87013, two permutations of 80 quartets ->20 batches of eight quartets. Training stochastic seed87011. No resampling during execution.',
 'objective':'Pinned prepare_example/pad_batch: BOS+prompt+four candidate tokens+EOS; x=sequence[:-1], labels ignored for context then candidate+EOS shifted correctly. Mean full-vocabulary CE over five response/EOS tokens per row. No contrastive term, prior subtraction or length normalization. Binding exact pinned answer CE + LAM1.0536573711078283 localization.',
 'scope':'Pinned v8 harness set_scope: English train every base_model parameter except blocks0-3; all specialized parameters frozen. Binding all parameters active. Frozen grad=None, optimizer moments/step and parameter values checked unchanged on every English update.',
 'optimizer':'fresh AdamW lr5e-5 betas(.9,.999) eps1e-8 wd.05 amsgrad=False foreach=False fused=False; continuous state, no scheduler; clip currently gradient-bearing parameters to2.0. float32 deterministic GPU; no autocast/TF32; no DataLoader; seed Python/torch CPU/GPU87011; Pythonhashseed87011.',
 'gate':'At fixed update200: 16/16 candidate correctness, 8/8 both-correct reversals, 4/4 complete families, 16/16 greedy exact candidate then EOS. No best-checkpoint selection. These are own-training-family acquisition criteria, not population generalization or readiness.',
 'scoring':'Candidate likelihood sum four tokens, excludeEOS, positive margin wins/ties fail; raw first-step candidate mass and full candidate-sequence mass; normal greedymax32 EOS, no repair. Reversal both-correct counts pairs once. Save every prompt/score/generation.',
 'monitoring':'At0/100/200 evaluate training acquisition, identical first128 TinyStoriesDEV sentence rows via pinned aligned_dev_loss, and BOTH frozen nonsacred binding pools separately. Human-readiness and FINAL batteries are not evaluated.',
 'safety_gates':'Each binding pool answer>=76/80,BD>=76/80,collapse=0; no averaging. Language loss <=actual update0 loss+0.25 nats (PPL ratio<=exp(.25)); stop at100 or200 if exceeded. This is a new prospective SF1 cost guard, not a change to readiness gates.',
 'evaluation_order':'Only if endpoint acquisition AND language/binding checks pass: HELDOUT16, ALTERNATE48 (passive cloze16,activeQA16,passiveQA16), COPY8 isolated card-name control, COMPETING64 same factual/card context with fact-vs-copy queries and card-before/after fact. All frozen before training. Transfer diagnostic and raw subgroup reporting; no additional acquisition/readiness claim from them.',
 'interpretation':'Training gate failure: stop/preserve and diagnose no automatic objective/scope escalation. Training success alone can be name copying. Heldout success establishes tested familiar actor/object recombination only; alternate results cue/wording specific. Competing fact/copy cues must switch answers in identical contexts to provide evidence beyond copying sole name or fixed mention. Do not call SF1 HUMAN_TEST_READY.',
 'persistence':'atomic fsync rolling restart after every successful update, including model/optimizer/RNG and parent/protocol/schedule/receipt hashes. One rolling plus permanent100/200. Resume only latest valid committed; committed+1 next; evaluations raw durably per item and independently resumed; commit optimizer update before scheduled evaluation. Corrupt provenance stops. Final checkpoint hash committed before transfer. No update500/seed87003/replication.',
 'counts':expected,'negative_tests':neg,'overlap_policy':'zero complete prompt overlap/substrings in listed nonsacred material; constituent words/clauses may overlap; no unseen vocabulary claim. Train actor/object combinations disjoint from heldout; alternate shares heldout semantics intentionally. Copy shares training actors/objects intentionally.','final_access':False,'sacred_access':False})
 sp=importlib.util.spec_from_file_location('sf1engine',O/'ENGINE.py');e=importlib.util.module_from_spec(sp);sp.loader.exec_module(e);rt.configure_runtime(87011);checks=e.preflight(O)
 import platform,torch,tokenizers
 assert platform.python_version()=='3.12.14' and torch.__version__=='2.12.0+rocm7.14.0' and tokenizers.__version__=='0.23.1'
 write('PREFLIGHT.json',{'status':'PASS','counts':expected,'max_completed_tokens':lengthmax,'all_prompt_unique':len(allrows),'independent_semantics':'PASS','negative':neg,'actor_correct_counts':{n:dict(Counter(r['candidates'][r['correct_index']] for r in rs)) for n,rs in panels.items()},'actor_object_disjoint':True,'overlap':audit,'mechanical':checks,'runtime':{'python':platform.python_version(),'torch':torch.__version__,'tokenizers':tokenizers.__version__,'gpu':torch.cuda.get_device_name(0)},'parent_loaded':False,'optimizer_created':False})
 write('SOURCE_PROVENANCE.json',{'predecessor':str(H),'receipt':sha(H/'FREEZE_RECEIPT.json'),'masking_scope_source':str(V/'harness.py'),'masking_scope_sha256':sha(V/'harness.py'),'staircase_report_sha256':sha(R/'hr3_diagnostic_staircase_v1_results/REPORT.md'),'autopsy_report_sha256':sha(R/'hr3_context_autopsy_v1/AUTOPSY_REPORT.md'),'training_definition_delta':'Two-fact/isolated sentence tasks replaced by bounded 16-item single-fact response curriculum; scope/objective/optimizer policies use verified prior implementations. No architecture or tokenizer changes.'})
 files=sorted(p for p in O.rglob('*') if p.is_file());(O/'SHA256SUMS.txt').write_text(''.join(sha(p)+'  '+p.relative_to(O).as_posix()+'\n' for p in files),encoding='utf-8',newline='\n')
 write('RECEIPT.json',{'status':'SF1_FROZEN_PREFLIGHT_PASS','manifest_sha256':sha(O/'SHA256SUMS.txt'),'protocol_sha256':sha(O/'PROTOCOL.json'),'preflight_sha256':sha(O/'PREFLIGHT.json')})
 (O/'RECEIPT.sha256').write_text(sha(O/'RECEIPT.json')+'  RECEIPT.json\n',encoding='utf-8')
 for p in O.rglob('*'):
  if p.is_file():p.chmod(0o444)
 e.verify(O);print(json.dumps({'status':'SF1_FROZEN','receipt':sha(O/'RECEIPT.json'),'manifest':sha(O/'SHA256SUMS.txt'),'counts':expected}))
if __name__=='__main__':main()
