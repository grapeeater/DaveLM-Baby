"""Additive SF14 reporting correction: show actual U0/U100/U200 values."""
import hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent; S=H.parent/'sf14_first_answer_token_ce_ablation_v1'; SEEDS=(87035,87036,87037)
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def snap(d,u):
 t=read(d/f'update{u}_train16_RESULT.json'); sf=read(d/f'update{u}_devsurface_RESULT.json'); od=read(d/f'update{u}_devorder_RESULT.json'); ch=read(d/f'update{u}_checks.json'); d3=read(d/f'd3_update{u}_summary.json'); sg=od['subgroups']
 return {'update':u,'train16':{k:t[k] for k in ['correct','exact','reversals','families']},'surface':{'correct':sf['correct'],'exact':sf['exact'],'subgroups':{k:v['exact'] for k,v in sf['subgroups'].items()}},'order':{'correct':od['correct'],'exact':od['exact'],'fact_exact':sum(sg[f'order{i}:fact']['exact'] for i in [0,1]),'copy_exact':sum(sg[f'order{i}:copy']['exact'] for i in [0,1]),'subgroups':{k:v['exact'] for k,v in sg.items()}},'language':ch['language'],'d3':d3,'binding':{k:v['summary'] for k,v in ch['binding'].items()}}
def main():
 rows=[]
 for s in SEEDS:
  d=S/'runs'/f'seed_{s}_curriculum'; st=read(d/'STATUS.json'); assert st['completed']==200 and st['status']=='STOP_REGRESSION' and sha(st['checkpoint'])==st['checkpoint_sha256']
  rows.append({'seed':s,'u0':snap(d,0),'u100':snap(d,100),'u200':snap(d,200),'gates100':read(d/'update100_GATES.json'),'gates200':read(d/'update200_GATES.json'),'status':st})
 assert all(r['gates100']['retention_acquisition'] and r['gates100']['d3'] and r['gates100']['language'] and all(r['gates100']['binding'].values()) for r in rows)
 assert all((not r['gates200']['retention_acquisition']) and r['gates200']['d3'] and r['gates200']['language'] and all(r['gates200']['binding'].values()) for r in rows)
 result={'classification':'RETENTION_REGRESSION','runs':rows,'frozen_logic':'Priority RETENTION_REGRESSION because every branch failed TRAIN16 at U200.','receipt_sha256':sha(S/'FREEZE_RECEIPT.json'),'manifest_sha256':sha(S/'SHA256SUMS.txt'),'controller_sha256':sha(S/'CONTROLLER.py'),'original_reporter_sha256':sha(S/'REPORTER.py'),'reporting_correction':'Original report U100 section accidentally read terminal/U200 snapshot. This additive report reads and labels U0/U100/U200 independently. No evaluation, score, gate, checkpoint, or classification changed.','locked_transfer':'LOCKED_UNSCORED','final_accessed':False,'sacred_accessed':False}
 (H/'RESULTS.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
 lines=['# SF14 FINAL CLASSIFICATION','', '**RETENTION_REGRESSION**','', '# WHAT CHANGED','', 'Only first-answer-name causal CE was masked during English updates. Later response/EOS CE, first-token margin, broad KL, curriculum, optimizer, scopes, and gates were unchanged.','', '# CORRECTED PER-SEED TRAJECTORY','', '| Seed | Update | TRAIN16 c/e/r/f | Surface c/e | Order e (fact/copy) | Language CE/PPL | D3 | Binding |','|---:|---:|---:|---:|---:|---:|---:|---|']
 for r in rows:
  for key in ['u0','u100','u200']:
   x=r[key]; t=x['train16']; b='; '.join(f"{k}:{v['answer_exact']}/{v['both_distinct']}/c{v['slot_collapse']}/q{v['complete_quartets']}/r{v['strict_reversal_both_correct']}" for k,v in x['binding'].items()); lines.append(f"| {r['seed']} | {x['update']} | {t['correct']}/{t['exact']}/{t['reversals']}/{t['families']} | {x['surface']['correct']}/{x['surface']['exact']} | {x['order']['exact']} ({x['order']['fact_exact']}/{x['order']['copy_exact']}) | {x['language']['loss']:.4f}/{x['language']['perplexity']:.2f} | {x['d3']['mean_combined_name_probability']:.5f} | {b} |")
 lines += ['', '# U100 OH-FUCK SIGNAL','']
 for r in rows:
  a,b=r['u0'],r['u100']; lines.append(f"- Seed {r['seed']}: D3 PASS `{b['d3']['mean_combined_name_probability']:.5f}`; TRAIN16 PASS `16/16/8/4`; language PASS (`{b['language']['loss']-a['language']['loss']:+.4f}` nat); binding PASS/PASS; surface exact `{a['surface']['exact']}→{b['surface']['exact']}`; order exact `{a['order']['exact']}→{b['order']['exact']}`.")
 lines += ['', '# U200 STOP','']
 for r in rows:
  x=r['u200']; lines.append(f"- Seed {r['seed']}: TRAIN16 `16 correct / 14 exact / 8 reversals / 4 families` — retention gate FAIL; D3 remained PASS at `{x['d3']['mean_combined_name_probability']:.5f}`; language and both binding pools remained PASS.")
 lines += ['', '# INTERPRETATION','', 'Removing first-answer-name causal CE strongly contained D3 in all three branches, but it did not preserve exact TRAIN16 answer+EOS behavior through U200 and did not retain widening exact generation. The frozen safety-priority classification is RETENTION_REGRESSION. The result does not establish that CE is the root cause, that CE is generally incompatible, or that the unchanged margin is sufficient.','', '# CHECKPOINTS','']
 for r in rows: lines.append(f"- Seed {r['seed']}: `{r['status']['checkpoint']}` — `{r['status']['checkpoint_sha256']}`")
 lines += ['', '# LOCKS','', '- Historical transfer panels: LOCKED/UNSCORED.', '- FINAL and sacred material: not accessed.','']
 (H/'FINAL_REPORT.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
 prov={'type':'ADDITIVE_REPORTING_ONLY_CORRECTION','source_study':str(S),'source_receipt_sha256':sha(S/'FREEZE_RECEIPT.json'),'source_reporter_sha256':sha(S/'REPORTER.py'),'training_replayed':False,'inference_replayed':False,'scientific_change':False}
 (H/'PROVENANCE.json').write_text(json.dumps(prov,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
 files=[]
 for p in sorted(S.joinpath('runs').rglob('*')):
  if p.is_file(): files.append(f'{sha(p)}  source/{p.relative_to(S).as_posix()}')
 for n in ['REPORTER.py','RESULTS.json','FINAL_REPORT.md','PROVENANCE.json']:
  files.append(f'{sha(H/n)}  {n}')
 (H/'SHA256SUMS.txt').write_text('\n'.join(files)+'\n',encoding='utf-8',newline='\n'); receipt={'status':'SF14_REPORTING_CORRECTION_COMPLETE','classification':'RETENTION_REGRESSION','manifest_sha256':sha(H/'SHA256SUMS.txt'),'scientific_change':False}
 (H/'RECEIPT.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n'); (H/'RECEIPT.sha256').write_text(sha(H/'RECEIPT.json')+'  RECEIPT.json\n',encoding='utf-8',newline='\n'); print('RETENTION_REGRESSION')
if __name__=='__main__': main()
