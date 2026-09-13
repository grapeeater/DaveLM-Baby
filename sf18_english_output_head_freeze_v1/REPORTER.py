"""Frozen SF18 reporter with explicit U0/U100/U200 rows."""
import json
from pathlib import Path
import CONTROLLER as C
H=Path(__file__).resolve().parent
def snap(d,u):
 t=C.read(d/f'update{u}_train16_RESULT.json'); s=C.read(d/f'update{u}_devsurface_RESULT.json'); o=C.read(d/f'update{u}_devorder_RESULT.json'); c=C.read(d/f'update{u}_checks.json'); d3=C.read(d/f'd3_update{u}_summary.json'); sg=o['subgroups']; gp=d/f'update{u}_GATES.json'
 return {'update':u,'train16':{k:t[k] for k in ['correct','exact','reversals','families']},'surface':{'correct':s['correct'],'exact':s['exact']},'order':{'correct':o['correct'],'exact':o['exact'],'fact_exact':sum(sg[f'order{i}:fact']['exact'] for i in [0,1]),'copy_exact':sum(sg[f'order{i}:copy']['exact'] for i in [0,1])},'language':c['language'],'d3':d3,'binding':{k:v['summary'] for k,v in c['binding'].items()},'gates':C.read(gp) if gp.exists() else None}
def retention(g): return g['retention_acquisition'] and g['language'] and all(g['binding'].values())
def classify(rows):
 if len(rows)!=3 or any(r['status'] not in ['STOP_REGRESSION','ACQUISITION_SUCCESS','ACQUISITION_FAIL'] for r in rows): return 'MECHANICAL_INCOMPLETE'
 if any(not retention(r['gates']) for r in rows): return 'RETENTION_REGRESSION'
 if sum(r['gates']['endpoint_pass'] for r in rows)>=2: return 'OUTPUT_HEAD_CONSTRAINT_SUPPORTED'
 if sum((not r['gates']['d3']) and r['widening_movement'] for r in rows)>=2: return 'OUTPUT_HEAD_CONSTRAINT_INSUFFICIENT'
 if sum(r['completed']==200 and r['gates']['d3'] and retention(r['gates']) and not r['gates']['dev_surface'] and not r['gates']['dev_order'] for r in rows)>=2: return 'WIDENING_STALLED_WITH_HEAD_FROZEN'
 return 'MIXED_OR_UNRESOLVED'
def main():
 p=C.verify(); rows=[]
 for seed in p['seeds']:
  d=H/'runs'/f'seed_{seed}_curriculum'; st=C.read(d/'STATUS.json'); us=[0,100]+([200] if st['completed']==200 else []); ss=[snap(d,u) for u in us]; metrics=[json.loads(x) for x in (d/'TRAIN_METRICS.jsonl').read_text().splitlines()]; assert len(metrics)==st['completed']
  rows.append({'seed':seed,'status':st['status'],'completed':st['completed'],'checkpoint':st['checkpoint'],'checkpoint_sha256':st['checkpoint_sha256'],'gates':st['gates'],'snapshots':ss,'terminal':ss[-1],'widening_movement':ss[-1]['surface']['exact']>ss[0]['surface']['exact'] and ss[-1]['order']['exact']>ss[0]['order']['exact']})
 cls=classify(rows); result={'classification':cls,'runs':rows,'cross_seed':{'endpoint_passes':sum(r['gates']['endpoint_pass'] for r in rows),'d3_failures':sum(not r['gates']['d3'] for r in rows),'retention_failures':sum(not retention(r['gates']) for r in rows),'widening_movement':sum(r['widening_movement'] for r in rows)},'receipt_sha256':C.sha(H/'FREEZE_RECEIPT.json'),'manifest_sha256':C.sha(H/'SHA256SUMS.txt'),'controller_sha256':C.sha(H/'CONTROLLER.py'),'locked_transfer':'LOCKED_UNSCORED','final_accessed':False,'sacred_accessed':False}
 C.rt.atomic_json(result,H/'RESULTS.json')
 lines=['# SF18 FINAL CLASSIFICATION','',f'**{cls}**','','# SOLE SCIENTIFIC CHANGE','','Relative to sealed SF13 full-CE training, SF18 froze only the untied language-head weight and bias during English updates. Binding updates restored the unchanged full T13 scope. All objectives, weights, data, optimizer, schedule, gates, binding, and evaluation remained unchanged.','', '# PER-SEED TRAJECTORY','', '| Seed | U | TRAIN16 c/e/r/f | Surface c/e | Order c/e (fact/copy) | Language CE/PPL | D3 | Binding |','|---:|---:|---:|---:|---:|---:|---:|---|']
 for r in rows:
  for x in r['snapshots']:
   t=x['train16']; b='; '.join(f"{k}:{v['answer_exact']}/{v['both_distinct']}/c{v['slot_collapse']}/q{v['complete_quartets']}/r{v['strict_reversal_both_correct']}" for k,v in x['binding'].items()); lines.append(f"| {r['seed']} | {x['update']} | {t['correct']}/{t['exact']}/{t['reversals']}/{t['families']} | {x['surface']['correct']}/{x['surface']['exact']} | {x['order']['correct']}/{x['order']['exact']} ({x['order']['fact_exact']}/{x['order']['copy_exact']}) | {x['language']['loss']:.4f}/{x['language']['perplexity']:.2f} | {x['d3']['mean_combined_name_probability']:.5f} | {b} |")
  if r['completed']<200: lines.append(f"| {r['seed']} | 200 | — | — | — | — | — | NOT REACHED: frozen stop at U{r['completed']} |")
 lines+=['','# FROZEN GATE RESULT','',f"- Endpoint passes: {result['cross_seed']['endpoint_passes']}/3",f"- D3 failures: {result['cross_seed']['d3_failures']}/3",f"- Retention failures: {result['cross_seed']['retention_failures']}/3",f"- Widening movement: {result['cross_seed']['widening_movement']}/3",'','# INTERPRETATION','']
 if cls=='OUTPUT_HEAD_CONSTRAINT_SUPPORTED': lines.append('Freezing the untied language head during English updates was sufficient in at least two branches to satisfy the complete frozen endpoint. This claim is confined to SF18.')
 elif cls=='OUTPUT_HEAD_CONSTRAINT_INSUFFICIENT': lines.append('English-update head freezing did not contain replicated D3 leakage while retention and widening remained active in at least two branches.')
 elif cls=='WIDENING_STALLED_WITH_HEAD_FROZEN': lines.append('Head freezing contained D3 and retention but widening stalled below both frozen endpoint gates in at least two branches.')
 elif cls=='RETENTION_REGRESSION': lines.append('A frozen TRAIN16, language, or binding retention gate failed; that safety classification takes priority.')
 else: lines.append('The three valid branches did not yield a decisive preregistered pattern.')
 lines+=['','# CHECKPOINTS','']+[f"- {r['seed']}: `{r['checkpoint']}` — `{r['checkpoint_sha256']}`" for r in rows]+['','# LOCKS','','- Historical transfer panels: LOCKED/UNSCORED.','- FINAL and sacred material: not accessed.','']
 (H/'FINAL_REPORT.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
 out=[]
 for q in sorted((H/'runs').rglob('*')):
  if q.is_file(): out.append(f'{C.sha(q)}  {q.relative_to(H).as_posix()}')
 for n in ['RESULTS.json','FINAL_REPORT.md','RUN_LEDGER.json']: out.append(f'{C.sha(H/n)}  {n}')
 (H/'OUTPUT_SHA256SUMS.txt').write_text('\n'.join(out)+'\n',encoding='utf-8',newline='\n'); print(cls)
if __name__=='__main__': main()
