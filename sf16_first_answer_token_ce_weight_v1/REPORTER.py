"""Frozen SF16 aggregation and prospective classification."""
import hashlib,json
from pathlib import Path
import CONTROLLER as C
H=Path(__file__).resolve().parent
def snap(d,u):
 t=C.read(d/f'update{u}_train16_RESULT.json'); sf=C.read(d/f'update{u}_devsurface_RESULT.json'); od=C.read(d/f'update{u}_devorder_RESULT.json'); ch=C.read(d/f'update{u}_checks.json'); d3=C.read(d/f'd3_update{u}_summary.json'); sg=od['subgroups']
 return {'update':u,'train16':{k:t[k] for k in ['correct','exact','reversals','families']},'surface':{'correct':sf['correct'],'exact':sf['exact'],'subgroups':{k:v['exact'] for k,v in sf['subgroups'].items()}},'order':{'correct':od['correct'],'exact':od['exact'],'fact_exact':sum(sg[f'order{i}:fact']['exact'] for i in [0,1]),'copy_exact':sum(sg[f'order{i}:copy']['exact'] for i in [0,1]),'subgroups':{k:v['exact'] for k,v in sg.items()}},'language':ch['language'],'d3':d3,'binding':{k:v['summary'] for k,v in ch['binding'].items()}}
def retention(g): return g['retention_acquisition'] and g['language'] and all(g['binding'].values())
def classify(rows):
 if len(rows)!=3 or any(r['status'] not in ['STOP_REGRESSION','ACQUISITION_SUCCESS','ACQUISITION_FAIL'] for r in rows): return 'MECHANICAL_INCOMPLETE'
 if any(not retention(r['gates']) for r in rows): return 'RETENTION_REGRESSION'
 if sum(r['gates']['endpoint_pass'] for r in rows)>=2: return 'FIRST_TOKEN_CE_INTERPOLATION_SUPPORTED'
 d3_tradeoff=sum((not r['gates']['d3']) and r['widening_movement'] for r in rows)
 sf14_like=sum(r['gates']['d3'] and retention(r['gates']) and
                 r.get('terminal', {}).get('surface', {}).get('exact', 999) <= 2 and
                 r.get('terminal', {}).get('order', {}).get('exact', 999) <= 2 for r in rows)
 if d3_tradeoff>=2 or sf14_like>=2: return 'FIRST_TOKEN_CE_INTERPOLATION_INSUFFICIENT'
 return 'MIXED_OR_UNRESOLVED'
def main():
 p=C.verify(); rows=[]
 for seed in p['seeds']:
  d=H/'runs'/f'seed_{seed}_curriculum'; st=C.read(d/'STATUS.json'); u=st['completed']; a=snap(d,0); b=snap(d,u)
  rows.append({'seed':seed,'status':st['status'],'completed':u,'checkpoint':st['checkpoint'],'checkpoint_sha256':st['checkpoint_sha256'],'gates':st['gates'],'u0':a,'terminal':b,'u200':b if u==200 else 'NOT_REACHED_FROZEN_STOP','widening_movement':b['surface']['exact']>a['surface']['exact'] and b['order']['exact']>a['order']['exact']})
 cls=classify(rows); result={'classification':cls,'runs':rows,'cross_seed':{'endpoint_passes':sum(r['gates']['endpoint_pass'] for r in rows),'d3_failures':sum(not r['gates']['d3'] for r in rows),'retention_failures':sum(not retention(r['gates']) for r in rows),'widening_movement':sum(r['widening_movement'] for r in rows)},'receipt_sha256':C.sha(H/'FREEZE_RECEIPT.json'),'manifest_sha256':C.sha(H/'SHA256SUMS.txt'),'controller_sha256':C.sha(H/'CONTROLLER.py'),'locked_transfer':'LOCKED_UNSCORED','final_accessed':False,'sacred_accessed':False}
 C.rt.atomic_json(result,H/'RESULTS.json')
 lines=['# SF16 FINAL CLASSIFICATION','',f'**{cls}**','', '# WHAT CHANGED','', 'Only the first-answer-name causal CE weight changed from sealed SF15 weight 0.25 to 0.125 during English updates. Later response/EOS CE, first-token margin, broad KL, curriculum, binding, optimizer, scopes, and gates were unchanged.','', '# PER-SEED RESULTS','', '| Seed | Update | TRAIN16 c/e/r/f | Surface c/e | Order e (fact/copy) | Language CE/PPL | D3 | Binding |','|---:|---:|---:|---:|---:|---:|---:|---|']
 for r in rows:
  for x in [r['u0'],r['terminal']]:
   b='; '.join(f"{k}:{v['answer_exact']}/{v['both_distinct']}/c{v['slot_collapse']}/q{v['complete_quartets']}/r{v['strict_reversal_both_correct']}" for k,v in x['binding'].items()); t=x['train16']; lines.append(f"| {r['seed']} | {x['update']} | {t['correct']}/{t['exact']}/{t['reversals']}/{t['families']} | {x['surface']['correct']}/{x['surface']['exact']} | {x['order']['exact']} ({x['order']['fact_exact']}/{x['order']['copy_exact']}) | {x['language']['loss']:.4f}/{x['language']['perplexity']:.2f} | {x['d3']['mean_combined_name_probability']:.5f} | {b} |")
  if r['completed']<200: lines.append(f"| {r['seed']} | 200 | — | — | — | — | — | NOT REACHED: frozen stop at {r['completed']} |")
 lines += ['', '# U100 SIGNAL','']
 for r in rows:
  a,b=r['u0'],r['terminal']; lines.append(f"- Seed {r['seed']}: D3 `{b['d3']['mean_combined_name_probability']:.5f}`; TRAIN16 {'PASS' if r['gates']['retention_acquisition'] else 'FAIL'}; language {'PASS' if r['gates']['language'] else 'FAIL'} (`{b['language']['loss']-a['language']['loss']:+.4f}` nat); binding {'PASS' if all(r['gates']['binding'].values()) else 'FAIL'}; surface exact `{a['surface']['exact']}→{b['surface']['exact']}`; order exact `{a['order']['exact']}→{b['order']['exact']}`.")
 lines += ['', '# ENDPOINT GATES','',f"- Complete endpoint passes: {result['cross_seed']['endpoint_passes']}/3",f"- D3 failures: {result['cross_seed']['d3_failures']}/3",f"- Retention failures: {result['cross_seed']['retention_failures']}/3",f"- Widening movement: {result['cross_seed']['widening_movement']}/3",'', '# INTERPRETATION','']
 if cls=='FIRST_TOKEN_CE_INTERPOLATION_SUPPORTED': lines.append('A 0.125 first-answer-token CE weight was sufficient under SF16 to preserve D3 while recovering exact factual generation/retention and maintaining tested language, binding, and widening behavior. This does not establish optimality, sole causation, or a general scaling law.')
 elif cls=='FIRST_TOKEN_CE_INTERPOLATION_INSUFFICIENT': lines.append('The 0.125 first-answer-token CE treatment reproduced at least two branches of the preregistered D3-versus-exact-generation tradeoff. This identifies the observed tradeoff under SF16 without establishing CE as its sole cause. Per the frozen program stopping rule, SF16 is the final simple CE-weight interpolation; no automatic micro-dose sweep follows.')
 elif cls=='RETENTION_REGRESSION': lines.append('SF16 failed a frozen TRAIN16, language, or binding retention requirement; safety failure takes precedence.')
 else: lines.append('The valid SF16 branches did not produce a decisive preregistered pattern.')
 lines += ['', '# CHECKPOINTS','']+[f"- Seed {r['seed']}: `{r['checkpoint']}` — `{r['checkpoint_sha256']}`" for r in rows]+['', '# LOCKS','', '- Historical transfer panels: LOCKED/UNSCORED.', '- FINAL and sacred material: not accessed.','']
 (H/'FINAL_REPORT.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
 out=[]
 for q in sorted((H/'runs').rglob('*')):
  if q.is_file(): out.append(f'{C.sha(q)}  {q.relative_to(H).as_posix()}')
 for n in ['RESULTS.json','FINAL_REPORT.md','RUN_LEDGER.json']:
  out.append(f'{C.sha(H/n)}  {n}')
 (H/'OUTPUT_SHA256SUMS.txt').write_text('\n'.join(out)+'\n',encoding='utf-8',newline='\n'); print(cls)
if __name__=='__main__': main()
