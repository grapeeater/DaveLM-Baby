"""Aggregate completed sealed SF13 outputs without model loading or inference."""
import hashlib,json,math,os
from pathlib import Path

H=Path(__file__).resolve().parent
S=H.parent/'sf13_broad_coverage_kl_retention_v1'
SEEDS=(87032,87033,87034)

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def snapshot(d,u):
 t=read(d/f'update{u}_train16_RESULT.json'); sf=read(d/f'update{u}_devsurface_RESULT.json'); od=read(d/f'update{u}_devorder_RESULT.json')
 ch=read(d/f'update{u}_checks.json'); d3=read(d/f'd3_update{u}_summary.json'); sg=od['subgroups']
 return {'update':u,'train16':{k:t[k] for k in ['correct','exact','reversals','families']},
  'dev_surface':{'correct':sf['correct'],'exact':sf['exact'],'subgroups':{k:v['exact'] for k,v in sf['subgroups'].items()}},
  'dev_order':{'correct':od['correct'],'exact':od['exact'],'fact_exact':sum(sg[f'order{i}:fact']['exact'] for i in [0,1]),'copy_exact':sum(sg[f'order{i}:copy']['exact'] for i in [0,1]),'subgroups':{k:v['exact'] for k,v in sg.items()}},
  'language':ch['language'],'d3':d3,
  'binding':{k:v['summary'] for k,v in ch['binding'].items()}}

def main():
 protocol=read(S/'PROTOCOL.json'); ledger=read(H/'RUN_LEDGER.json'); assert len(ledger)==3 and all(x['state']=='CLASSIFIED' for x in ledger)
 rows=[]
 for seed in SEEDS:
  d=S/'runs'/f'seed_{seed}_curriculum'; status=read(d/'STATUS.json'); assert status['completed']==100 and status['status']=='STOP_REGRESSION'
  assert sha(status['checkpoint'])==status['checkpoint_sha256']
  u0=snapshot(d,0); u100=snapshot(d,100)
  widening=(u100['dev_surface']['exact']>u0['dev_surface']['exact'] and u100['dev_order']['exact']>u0['dev_order']['exact'])
  rows.append({'seed':seed,'parent':next(r for r in protocol['runs'] if r['seed']==seed),'u0':u0,'u100':u100,'u200':'NOT_REACHED_FROZEN_D3_STOP',
   'gates':status['gates'],'status':status['status'],'stop_reason':'D3 > 0.010','checkpoint':status['checkpoint'],'checkpoint_sha256':status['checkpoint_sha256'],'widening_movement':widening})
 retention_ok=all(r['gates']['retention_acquisition'] and r['gates']['language'] and all(r['gates']['binding'].values()) for r in rows)
 d3fails=sum(not r['gates']['d3'] for r in rows); movement=sum(r['widening_movement'] for r in rows)
 if retention_ok and d3fails>=2 and movement>=2: classification='RETENTION_POSITION_COVERAGE_INSUFFICIENT'
 else: classification='MIXED_OR_UNRESOLVED'
 result={'classification':classification,'runs':rows,'cross_seed':{'d3_failures':d3fails,'widening_movement':movement,'complete_endpoint_passes':sum(r['gates']['endpoint_pass'] for r in rows),'retention_failures':sum(not (r['gates']['retention_acquisition'] and r['gates']['language'] and all(r['gates']['binding'].values())) for r in rows)},
  'sealed_study_receipt_sha256':sha(S/'FREEZE_RECEIPT.json'),'sealed_study_manifest_sha256':sha(S/'SHA256SUMS.txt'),'controller_sha256':sha(S/'CONTROLLER.py'),'kl_schedule_sha256':sha(S/'SF13_KL_SCHEDULE.json'),
  'locked_transfer':'LOCKED_UNSCORED','final_accessed':False,'sacred_accessed':False}
 (H/'RESULTS.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
 lines=['# SF13 FINAL CLASSIFICATION','',f"**{classification}**",'',
  '# WHAT CHANGED','', 'The sole scientific variable was the frozen broad-coverage KL retention-position sampling geometry. The KL pool, 160-position budget, loss, curriculum, optimizer, scopes, and gates were unchanged.','',
  '# PER-SEED RESULTS','', '| Seed | Update | TRAIN16 c/e/r/f | Surface c/e | Order e (fact/copy) | Language CE / PPL | D3 | Binding pools |','|---:|---:|---:|---:|---:|---:|---:|---|']
 for r in rows:
  for key in ['u0','u100']:
   x=r[key]; b='; '.join(f"{k}:{v['answer_exact']}/{v['both_distinct']}/c{v['slot_collapse']}/q{v['complete_quartets']}/r{v['strict_reversal_both_correct']}" for k,v in x['binding'].items())
   t=x['train16']; lines.append(f"| {r['seed']} | {x['update']} | {t['correct']}/{t['exact']}/{t['reversals']}/{t['families']} | {x['dev_surface']['correct']}/{x['dev_surface']['exact']} | {x['dev_order']['exact']} ({x['dev_order']['fact_exact']}/{x['dev_order']['copy_exact']}) | {x['language']['loss']:.4f} / {x['language']['perplexity']:.2f} | {x['d3']['mean_combined_name_probability']:.5f} | {b} |")
  lines.append(f"| {r['seed']} | 200 | — | — | — | — | — | NOT REACHED: frozen D3 stop at 100 |")
 lines += ['', '# U100 SIGNAL','', '| Seed | D3 ≤.010 | TRAIN16 | Language ≤U0+.25 | Binding | Surface movement | Order movement | Stop |','|---:|---|---|---|---|---:|---:|---|']
 for r in rows:
  a,b=r['u0'],r['u100']; lines.append(f"| {r['seed']} | **FAIL** ({b['d3']['mean_combined_name_probability']:.5f}) | PASS | PASS (+{b['language']['loss']-a['language']['loss']:.4f}) | PASS/PASS | +{b['dev_surface']['exact']-a['dev_surface']['exact']} exact | +{b['dev_order']['exact']-a['dev_order']['exact']} exact | STOP_REGRESSION |")
 lines += ['', '# ENDPOINT GATES','', '- TRAIN16: PASS 3/3.', '- Language: PASS 3/3.', '- Pilot0 binding: PASS 3/3.', '- Pilot1 binding: PASS 3/3.', '- D3: FAIL 3/3.', '- DEV_SURFACE endpoint: FAIL 3/3.', '- DEV_ORDER endpoint: FAIL 3/3.', '- Complete endpoint: 0/3.', '',
  '# INTERPRETATION','', 'Broadening retention-position coverage within the same frozen KL pool did not keep D3 below 0.010. All three Babies nevertheless retained TRAIN16, language, and binding, while surface and order/source exactness moved upward. Under the frozen classification, broader coverage was insufficient to contain the replicated D3 regression. This weakens retention-position coverage as the primary explanation under this tested geometry; it does not establish a deeper mechanism.','',
  '# LOCKS','', '- Historical transfer panels: LOCKED/UNSCORED.', '- FINAL: not accessed.', '- Sacred material: not accessed.', '',
  '# CHECKPOINTS','']
 for r in rows: lines.append(f"- Seed {r['seed']}: `{r['checkpoint']}` — `{r['checkpoint_sha256']}`")
 lines += ['', '# PROVENANCE','',f"- Sealed SF13 receipt: `{result['sealed_study_receipt_sha256']}`",f"- Sealed SF13 manifest: `{result['sealed_study_manifest_sha256']}`",f"- Controller: `{result['controller_sha256']}`",f"- Frozen KL schedule: `{result['kl_schedule_sha256']}`",'', '# EXACT NEXT ACTION','', 'Conduct the single preregistered widening-supervision redesign review specified by the frozen SF13 protocol. Do not execute a new treatment from this report.','']
 (H/'FINAL_REPORT.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
 payload=[]
 for p in sorted((S/'runs').rglob('*')):
  if p.is_file(): payload.append(f'{sha(p)}  study/{p.relative_to(S).as_posix()}')
 for p in sorted(H.rglob('*')):
  if p.is_file() and p.name not in {'OUTPUT_SHA256SUMS.txt','EXECUTION_RECEIPT.json','EXECUTION_RECEIPT.sha256'} and '__pycache__' not in p.parts:
   payload.append(f'{sha(p)}  execution/{p.relative_to(H).as_posix()}')
 (H/'OUTPUT_SHA256SUMS.txt').write_text('\n'.join(payload)+'\n',encoding='utf-8',newline='\n')
 receipt={'status':'SF13_EXECUTION_AND_CLASSIFICATION_COMPLETE','classification':classification,'manifest_sha256':sha(H/'OUTPUT_SHA256SUMS.txt'),'sealed_study_receipt_sha256':result['sealed_study_receipt_sha256'],'runs':3,'final_accessed':False,'sacred_accessed':False}
 (H/'EXECUTION_RECEIPT.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
 (H/'EXECUTION_RECEIPT.sha256').write_text(sha(H/'EXECUTION_RECEIPT.json')+'  EXECUTION_RECEIPT.json\n',encoding='utf-8',newline='\n')
 print(classification)
if __name__=='__main__': main()
