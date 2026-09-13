import ast,hashlib,json,platform,stat,sys,time
from pathlib import Path
ROOT=Path(r'C:\DaveLM-CADAVER'); F=ROOT/'post_p7_language_report_card_v3d_seed8391'; OUT=ROOT/'post_p7_v3d_stage1_execution_retry'
TOK=Path(r'C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json'); TH='e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b'
CK={
'graduate':(ROOT/'treatment13_orthogonal_shared_unbounded_seed8380/checkpoints/orthogonal_shared_unbounded/seed_8380/latest.pt','fed298748e62def6f1751ef1bb7df0cdec51d53fac6e4c86e8c00a95dccdd430'),
'pilot0':(ROOT/'language_pilot_0_tinystories_seed8380/pilot_run/checkpoints/seed_8380/latest.pt','769bd01efd28e7888064c0d5fe1dd0d85a4344e2039aef04bcbf7d9e906f47f5'),
'pilot1':(ROOT/'language_pilot_1_early_block_protection_seed8380/pilot_run/checkpoints/seed_8380/latest.pt','2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb'),
'p5':(ROOT/'language_sentencebound_p5/latest.pt','da009100412fba2adcf58823aefd37cc9486ef7aff39672361ce6111773c6a3b'),
'p6':(ROOT/'language_compositional_p6/latest.pt','e708f00bdab4c4325382f3ceb724ad377360daf8ff2f6c14278cc17e692a9930'),
'p7':(ROOT/'archive/DAVELM_P7_MILESTONE_seed8380/davelm_p7_latest.pt','d41ed1186cc945aa05dbd2ba3086fac08ff3b97035c9532e4fa70e78b149a20e')}
REF={'pilot0_dev':(ROOT/'language_pilot_0_tinystories_seed8380/binding_dev.json','30bfbcbe1b11d3d2d427a631562f43b24dbb97f6e595164516d40553792edc4e'),'pilot1_dev':(ROOT/'language_pilot_1_early_block_protection_seed8380/binding_dev.json','3b6774b2cadeb7818d59c8a4f4f0b69b2cd85744efce91af562d9bd6f2a54ea1')}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert not OUT.exists(); assert sha(TOK)==TH
 for p,h in CK.values(): assert sha(p)==h
 for p,h in REF.values(): assert sha(p)==h
 sums=(F/'SHA256SUMS.txt').read_text().splitlines(); assert all(sha(F/n)==h for h,n in (x.split('  ') for x in sums)); assert all(not p.stat().st_mode&stat.S_IWRITE for p in F.iterdir())
 rows=[json.loads(x) for x in (F/'ITEMS.jsonl').read_text(encoding='utf8').splitlines()]; assert len(rows)==288
 pri=[r for r in rows if r['section']=='query_only_prior']; ctrl=[r for r in rows if r['section'] in ('near_distribution','counterfactual','surface_form','distractor')]; nat=[r for r in rows if r['section']=='naturalistic']; assert len(pri)==16 and len(ctrl)==256 and len(nat)==16
 sys.path.insert(0,r'C:\DaveLM-v0.9'); sys.path.insert(0,str(ROOT)); import torch,tokenizers
 tok=tokenizers.Tokenizer.from_file(str(TOK)); from v0_8_2.model import build_model
 tree=ast.parse((ROOT/'language_pilot_0_tinystories_seed8380/run.py').read_text()); ns={'torch':torch,'nn':torch.nn,'defaultdict':__import__('collections').defaultdict}; names={'basis','OrthoLocalizer','rowpos','binding_eval'}; exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names],type_ignores=[]),'binding','exec'),ns)
 device='cuda:0'; models={}; full_models={}
 from treatment13_model import Treatment13Model
 for k,(p,h) in CK.items():
  raw=torch.load(p,map_location='cpu',weights_only=True); state=raw['model_state_dict']; model=build_model('untied'); base={n[len('base_model.'):]:v for n,v in state.items() if n.startswith('base_model.')}; model.load_state_dict(base,strict=True); model.to(device).eval(); models[k]=model
  full=Treatment13Model(); full.localizer=ns['OrthoLocalizer'](*(state[f'localizer.{name}'].clone() for name in ('u','q','bs','ba'))); full.load_state_dict(state,strict=True); full.to(device).eval(); full_models[k]=full
 OUT.mkdir(); json.dump({'status':'PASS','tokenizer_sha256':TH,'checkpoint_hashes':{k:h for k,(_,h) in CK.items()},'reference_hashes':{k:h for k,(_,h) in REF.items()}},open(OUT/'PRECHECK.json','w'),indent=2)
 from collections import defaultdict,Counter
 scores=[]; gens=[]
 for k,m in models.items():
  for r in ctrl+pri:
   if 'candidate_token_ids' not in r: continue
   p=r['prompt_token_ids']; vals=[]
   for c in r['candidate_token_ids']:
    ids=[2]+p+c; logits=m(torch.tensor([ids],device=device))[0,len(p):len(p)+len(c),:].detach().float().cpu(); vals.append(float(sum(torch.log_softmax(logits.double(),-1)[range(len(c)),torch.tensor(c)].tolist())))
   ci=r.get('correct_index'); scores.append({'checkpoint':k,'item_id':r['item_id'],'section':r['section'],'candidate_log_likelihoods':vals,'margin':(vals[ci]-vals[1-ci]) if ci is not None else None,'correct':(vals[ci]>vals[1-ci]) if ci is not None else None,'tie':(vals[0]==vals[1]),'record':r})
  for r in nat:
   ids=[2]+r['prompt_token_ids']; new=[]
   for _ in range(32):
    n=int(m(torch.tensor([ids],device=device))[0,-1,:].argmax()); new.append(n); ids.append(n)
    if n in (3,5,18,35,901): break
   gens.append({'checkpoint':k,'item_id':r['item_id'],'prompt':r['prompt'],'generated_token_ids':new,'continuation':tok.decode(new,skip_special_tokens=True)})
 bind={}
 for k,m in models.items():
  bind[k]={}
  for ref,(p,h) in REF.items():
   docs=[d for q in json.loads(p.read_text())['quartets'] for d in q['docs']]; res=ns['binding_eval'](full_models[k],docs,torch.device(device)); bd=[x for x in res['rows'] if x['localization_category']=='BOTH_DISTINCT']; res['queried_row_given_BD']={'correct':sum(x['selected_query_row_correct'] for x in bd),'n':len(bd)}; res['answer_given_BD']={'correct':sum(x['answer_correct'] for x in bd),'n':len(bd)}; bind[k][ref]=res
 json.dump(scores,open(OUT/'RAW_SCORES.json','w'),indent=2); json.dump(gens,open(OUT/'RAW_GENERATIONS.json','w'),indent=2); json.dump(bind,open(OUT/'BINDING_REFERENCE_RESULTS.json','w'),indent=2)
 agg={}
 for k in models:
  agg[k]={}
  for s in ('near_distribution','counterfactual','surface_form','distractor','query_only_prior'):
   z=[x for x in scores if x['checkpoint']==k and x['section']==s]; agg[k][s]={'n':len(z),'correct':sum(x['correct'] is True for x in z),'ties':sum(x['tie'] for x in z),'mean_margin':sum(x['margin'] or 0 for x in z)/len(z)}
 json.dump(agg,open(OUT/'SUMMARY.json','w'),indent=2); json.dump({'status':'COMPLETE','no_training':True,'sacred_accessed':False,'checkpoint_hashes_after':{k:sha(p) for k,(p,_) in CK.items()}},open(OUT/'POST_RECEIPT.json','w'),indent=2)
 print(json.dumps(agg,indent=2))
if __name__=='__main__': main()
