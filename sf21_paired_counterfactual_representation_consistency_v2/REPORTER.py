"""Frozen SF21 cross-seed reporter and 10.6M stopping-rule classifier."""
import json
from pathlib import Path
import CONTROLLER as C
H=Path(__file__).resolve().parent
def snap(d,u):
 t=C.read(d/f'update{u}_train16_RESULT.json'); s=C.read(d/f'update{u}_devsurface_RESULT.json'); o=C.read(d/f'update{u}_devorder_RESULT.json'); c=C.read(d/f'update{u}_checks.json'); d3=C.read(d/f'd3_update{u}_summary.json'); sg=o['subgroups']; gp=d/f'update{u}_GATES.json'
 return {'update':u,'train16':{k:t[k] for k in ['correct','exact','reversals','families']},'surface':{'correct':s['correct'],'exact':s['exact'],'subgroups':s['subgroups']},'order':{'correct':o['correct'],'exact':o['exact'],'fact_exact':sum(sg[f'order{i}:fact']['exact'] for i in [0,1]),'copy_exact':sum(sg[f'order{i}:copy']['exact'] for i in [0,1]),'subgroups':sg},'language':c['language'],'d3':d3,'binding':{k:v['summary'] for k,v in c['binding'].items()},'gates':C.read(gp) if gp.exists() else None}
def retention(g): return g['retention_acquisition'] and g['language'] and all(g['binding'].values())
def classify(rows):
 if len(rows)!=3 or any(r['status'] not in ['STOP_REGRESSION','ACQUISITION_SUCCESS','ACQUISITION_FAIL'] for r in rows): return 'MECHANICAL_INCOMPLETE'
 if any(not retention(r['gates']) for r in rows): return 'RETENTION_REGRESSION_10M_SERIES_CLOSED'
 if sum(r['gates']['endpoint_pass'] for r in rows)>=2: return 'SF21_COEXISTENCE_FRONTIER_SUCCESS'
 return 'SF21_COEXISTENCE_FRONTIER_FAIL_10M_SERIES_CLOSED'
def main():
 p=C.verify(); rows=[]
 for seed in p['seeds']:
  d=H/'runs'/f'seed_{seed}_curriculum'; st=C.read(d/'STATUS.json'); us=[0,100]+([200] if st['completed']==200 else []); ss=[snap(d,u) for u in us]
  metrics=[json.loads(x) for x in (d/'TRAIN_METRICS.jsonl').read_text().splitlines()]; assert len(metrics)==st['completed']
  cm=[x for x in metrics if x['kind']=='english']; telemetry={'mean_consistency_loss':sum(x['consistency_loss'] for x in cm)/len(cm),'mean_weighted_consistency':sum(x['weighted_consistency'] for x in cm)/len(cm),'mean_pair_cosine':sum(x['mean_pair_cosine'] for x in cm)/len(cm),'final_consistency_loss':cm[-1]['consistency_loss'],'final_pair_cosine':cm[-1]['mean_pair_cosine']}
  rows.append({'seed':seed,'status':st['status'],'completed':st['completed'],'checkpoint':st['checkpoint'],'checkpoint_sha256':st['checkpoint_sha256'],'gates':st['gates'],'snapshots':ss,'terminal':ss[-1],'consistency':telemetry})
 cls=classify(rows); result={'classification':cls,'runs':rows,'cross_seed':{'endpoint_passes':sum(r['gates']['endpoint_pass'] for r in rows),'retention_failures':sum(not retention(r['gates']) for r in rows),'u200_reached':sum(r['completed']==200 for r in rows)},'series_status':'SUCCESS_LINEAGE_PRESERVED' if cls=='SF21_COEXISTENCE_FRONTIER_SUCCESS' else ('MECHANICAL_INCOMPLETE' if cls=='MECHANICAL_INCOMPLETE' else '10.6M_FACTUAL_SUPERVISION_SERIES_CLOSED'),'locked_transfer':'LOCKED_UNSCORED','final_accessed':False,'sacred_accessed':False,'receipt_sha256':C.sha(H/'FREEZE_RECEIPT.json'),'manifest_sha256':C.sha(H/'SHA256SUMS.txt'),'controller_sha256':C.sha(H/'CONTROLLER.py')}
 C.rt.atomic_json(result,H/'RESULTS.json')
 lines=['# SF21 FINAL CLASSIFICATION','',f'**{cls}**','','# WHAT CHANGED','','SF21 added only `0.1 × mean(1 − cosine)` across the 16 frozen SF20 assignment-reversal pairs, using the final-normalized last context/query state that predicts the first answer token. All SF20 data, objectives, schedule, parents, optimizer, scopes, gates, and evaluators remained frozen.','', '# PER-SEED RESULTS','', '| Seed | Stop | TRAIN16 c/e/r/f | Surface c/e | Order e (fact/copy) | Language Δ | D3 U0→final | Binding | Mean cosine | Checkpoint SHA-256 |','|---:|---:|---:|---:|---:|---:|---:|---|---:|---|']
 for r in rows:
  a=r['snapshots'][0]; z=r['terminal']; t=z['train16']; b='; '.join(f"{k}:{v['answer_exact']}/{v['both_distinct']}/c{v['slot_collapse']}" for k,v in z['binding'].items()); lines.append(f"| {r['seed']} | {r['completed']} | {t['correct']}/{t['exact']}/{t['reversals']}/{t['families']} | {z['surface']['correct']}/{z['surface']['exact']} | {z['order']['exact']} ({z['order']['fact_exact']}/{z['order']['copy_exact']}) | {z['language']['loss']-a['language']['loss']:+.4f} | {a['d3']['mean_combined_name_probability']:.5f}→{z['d3']['mean_combined_name_probability']:.5f} | {b} | {r['consistency']['mean_pair_cosine']:.4f} | `{r['checkpoint_sha256']}` |")
 lines+=['','# FROZEN CROSS-SEED RESULT','',f"- Endpoint passes: {result['cross_seed']['endpoint_passes']}/3",f"- Retention failures: {result['cross_seed']['retention_failures']}/3",f"- Branches reaching U200: {result['cross_seed']['u200_reached']}/3",'','# INTERPRETATION','']
 if cls=='SF21_COEXISTENCE_FRONTIER_SUCCESS': lines.append('At least two branches met every frozen coexistence gate. The successful checkpoint lineage is preserved; this establishes only the tested controlled widening endpoint.')
 elif cls=='MECHANICAL_INCOMPLETE': lines.append('Execution or integrity prevented a valid scientific classification.')
 else: lines.append('SF21 did not move the coexistence frontier sufficiently. Under the prospectively declared stopping rule, the 10.6M factual-supervision treatment series is closed.')
 lines+=['','The consistency telemetry shows whether the auxiliary term engaged; it is not a behavioral gate and cannot establish relational generalization by itself. Historical transfer panels remained LOCKED/UNSCORED. FINAL and sacred material were not accessed.','']
 (H/'FINAL_REPORT.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
 out=[]
 for q in sorted((H/'runs').rglob('*')):
  if q.is_file(): out.append(f'{C.sha(q)}  {q.relative_to(H).as_posix()}')
 for n in ['RESULTS.json','FINAL_REPORT.md','RUN_LEDGER.json']: out.append(f'{C.sha(H/n)}  {n}')
 (H/'OUTPUT_SHA256SUMS.txt').write_text('\n'.join(out)+'\n',encoding='utf-8',newline='\n'); print(cls)
if __name__=='__main__': main()
