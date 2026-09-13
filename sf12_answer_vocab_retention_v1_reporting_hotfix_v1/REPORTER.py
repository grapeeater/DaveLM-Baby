"""Frozen SF12 reporter and cross-seed classifier."""
import hashlib,json,math
from pathlib import Path
import sys
SOURCE=Path(r'C:\\DaveLM-CADAVER\\sf12_answer_vocab_retention_v1')
sys.path.insert(0,str(SOURCE))
import CONTROLLER as C

OUT=Path(__file__).resolve().parent
H=SOURCE
read=C.read

def retention_regression(g):
    return (not g['retention_acquisition']) or (not g['language']) or (not all(g['binding'].values()))

def partial_progress(run):
    if run['completed']!=200 or not run['gates']['continue']: return False
    return (run['gates']['development_gain'] and run['dev_surface']['exact']>=12 and run['dev_order']['exact']>=8)

def classify(rows):
    if len(rows)!=3 or any(r['status'] not in ['STOP_REGRESSION','ACQUISITION_SUCCESS','ACQUISITION_FAIL'] for r in rows):
        return 'INCONCLUSIVE_EARLY_STOP'
    if any(retention_regression(r['gates']) for r in rows): return 'RETENTION_REGRESSION'
    if sum(r['gates']['endpoint_pass'] for r in rows)>=2: return 'ANSWER_VOCAB_RETENTION_SUPPORTED_FULL_ENDPOINT'
    if sum(not r['gates']['d3'] for r in rows)>=2: return 'ANSWER_VOCAB_RETENTION_WEAKENED'
    if sum(partial_progress(r) for r in rows)>=2: return 'ANSWER_VOCAB_RETENTION_SUPPORTED_PARTIAL_WIDENING'
    if sum(r['completed']==200 and r['gates']['continue'] for r in rows)>=2 and sum(partial_progress(r) for r in rows)<2:
        return 'WIDENING_STALLED_UNDER_RETENTION'
    return 'MIXED_OR_UNRESOLVED'

def summarize(seed):
    out=H/'runs'/f'seed_{seed}_curriculum'; st=read(out/'STATUS.json'); u=st['completed']
    ret=read(out/f'update{u}_train16_RESULT.json'); ds=read(out/f'update{u}_devsurface_RESULT.json'); do=read(out/f'update{u}_devorder_RESULT.json')
    ch=read(out/f'update{u}_checks.json'); d3=read(out/f'd3_update{u}_summary.json'); tel=read(out/f'update{u}_NAME_RETENTION_TELEMETRY.json')
    sg=do['subgroups']; fact=sum(sg[f'order{o}:fact']['exact'] for o in [0,1]); copy=sum(sg[f'order{o}:copy']['exact'] for o in [0,1])
    metrics=[json.loads(x) for x in (out/'TRAIN_METRICS.jsonl').read_text().splitlines()]
    name_train=[m['name_retention'] for m in metrics if m['kind']=='english']
    train_tel={k:sum(x[k] for x in name_train)/len(name_train) for k in ['mean_S_student','mean_S_teacher','mean_excess_before_relu','active_fraction','mean_active_excess','R_name','R_name_loss_contribution']}
    return {'seed':seed,'status':st['status'],'completed':u,'checkpoint':st['checkpoint'],'checkpoint_sha256':st['checkpoint_sha256'],'gates':st['gates'],
      'train16':{k:ret[k] for k in ['correct','exact','reversals','families']},'dev_surface':{'correct':ds['correct'],'exact':ds['exact'],'subgroups':ds['subgroups']},
      'dev_order':{'correct':do['correct'],'exact':do['exact'],'fact_exact':fact,'copy_exact':copy,'subgroups':sg},'language':ch['language'],'d3':d3,
      'binding':{k:v['summary'] for k,v in ch['binding'].items()},'R_name_endpoint_probe':tel,'R_name_training_mean':train_tel}

def render(result):
    lines=['# SF12 FINAL CLASSIFICATION','',f"**{result['classification']}**",'', '# WHAT CHANGED','',
      'The sole scientific change from SF11 was adding the frozen teacher-anchored answer-vocabulary excess term `R_name` to English updates. Curriculum, schedules, parents, optimizer, full KL, margin loss, parameter scope, binding rehearsal, evaluation, and gates remained fixed.','',
      '# SEALED PROVENANCE','',f"- Receipt: `{result['receipt_sha256']}`",f"- Manifest: `{result['manifest_sha256']}`",f"- Controller: `{result['controller_sha256']}`",'- Historical/FINAL/sacred panels: locked and unscored','',
      '# PER-SEED RESULTS','', '| Seed | Stop | TRAIN16 c/e/r/f | Surface c/e | Order e (fact/copy) | Language CE | D3 | Binding | R_name probe | Status |','|---:|---:|---:|---:|---:|---:|---:|---|---:|---|']
    for r in result['runs']:
      b='; '.join(f"{k}:{v['answer_exact']}/{v['both_distinct']}/c{v['slot_collapse']}" for k,v in r['binding'].items())
      t=r['train16']; lines.append(f"| {r['seed']} | {r['completed']} | {t['correct']}/{t['exact']}/{t['reversals']}/{t['families']} | {r['dev_surface']['correct']}/{r['dev_surface']['exact']} | {r['dev_order']['exact']} ({r['dev_order']['fact_exact']}/{r['dev_order']['copy_exact']}) | {r['language']['loss']:.4f} | {r['d3']['mean_combined_name_probability']:.5f} | {b} | {r['R_name_endpoint_probe']['R_name']:.6f} | {r['status']} |")
    x=result['cross_seed']; lines += ['', '# CROSS-SEED RESULT','',f"- Complete endpoint successes: {x['complete']}/3",f"- Partial retention/widening successes: {x['partial']}/3",f"- D3 failures: {x['d3_fail']}/3",f"- Retention regressions: {x['retention_regression']}/3",'', '# DID THE NEW TERM ACTUALLY ENGAGE','']
    for r in result['runs']:
      t=r['R_name_training_mean']; lines.append(f"- Seed {r['seed']}: mean R_name `{t['R_name']:.6f}`, active fraction `{t['active_fraction']:.3f}`, mean student/teacher mass `{t['mean_S_student']:.6f}` / `{t['mean_S_teacher']:.6f}`.")
    lines += ['', '# DAVE-CODED INTERPRETATION','',result['dave_coded'],'', '# SCIENTIFIC INTERPRETATION','',result['scientific_interpretation'],'', '# ARTIFACTS','',f"- Study: `{H}`",f"- Machine-readable results: `{OUT/'RESULTS.json'}`",f"- Output hashes: `{OUT/'OUTPUT_SHA256SUMS.txt'}`",'', '# EXACTLY ONE NEXT ACTION','',result['next_action'],'']
    return '\n'.join(lines)

def main():
    p=C.verify(); rows=[summarize(s) for s in p['seeds']]; classification=classify(rows)
    complete=sum(r['gates']['endpoint_pass'] for r in rows); partial=sum(partial_progress(r) for r in rows); d3f=sum(not r['gates']['d3'] for r in rows); rr=sum(retention_regression(r['gates']) for r in rows)
    if classification=='ANSWER_VOCAB_RETENTION_SUPPORTED_FULL_ENDPOINT':
      dave='The patch kept the four trained names under the D3 ceiling in at least two Babies while they completed the wider wording/order lesson and retained TRAIN16, language, and binding.'
      sci='Under this frozen curriculum and recipe, dedicated teacher-anchored retention of the trained answer vocabulary was sufficient to contain the replicated D3 regression while preserving endpoint widening. This does not solve corpus coverage or establish the exact internal mechanism.'
      nxt='Prospectively design an independent replication/transfer confirmation without opening any locked panel in this study.'
    elif classification=='ANSWER_VOCAB_RETENTION_SUPPORTED_PARTIAL_WIDENING':
      dave='The patch contained D3 in at least two Babies and wider behavior remained alive, but the full widening endpoint was not completed.'
      sci='The treatment supports answer-vocabulary retention as a practical D3 containment measure in this setting, while leaving acquisition incomplete. It does not establish corpus coverage or a general mechanism.'
      nxt='Review the frozen widening residuals to preregister one bounded acquisition follow-up while keeping this retention term fixed.'
    elif classification=='ANSWER_VOCAB_RETENTION_WEAKENED':
      dave='The dedicated name-retention term did not stop the replicated D3 breach in at least two Babies.'
      sci='This weakens the proposition that answer-vocabulary under-attention on the existing KL positions is sufficient to explain or fix the D3 regression. Corpus coverage remains untested.'
      nxt='Prospectively review broader retention-position coverage as the next bounded hypothesis without changing this result.'
    elif classification=='WIDENING_STALLED_UNDER_RETENTION':
      dave='The patch kept D3 safe, but the wider wording/order behavior did not make the preregistered material progress in at least two Babies.'
      sci='Dedicated answer-vocabulary retention contained the measured pollution under this recipe, but the tested pressure also coincided with insufficient widening. This does not isolate why widening stalled.'
      nxt='Review the frozen treatment telemetry and widening failures before choosing one acquisition-side change.'
    elif classification=='RETENTION_REGRESSION':
      dave='At least one Baby lost an original protected capability; the retention safety discipline takes priority over widening.'
      sci='SF12 failed a preregistered non-D3 retention gate. No broader efficacy claim is justified.'
      nxt='Audit the specific frozen retention failure before authorizing another treatment.'
    else:
      dave='The three Babies did not produce a decisive preregistered cross-seed pattern.'
      sci='The tested answer-vocabulary retention hypothesis remains unresolved under this bounded study.'
      nxt='Review the cross-seed divergence before selecting any new treatment variable.'
    result={'classification':classification,'runs':rows,'cross_seed':{'complete':complete,'partial':partial,'d3_fail':d3f,'retention_regression':rr},
      'receipt_sha256':C.sha(H/'FREEZE_RECEIPT.json'),'manifest_sha256':C.sha(H/'SHA256SUMS.txt'),'controller_sha256':C.sha(H/'CONTROLLER.py'),
      'dave_coded':dave,'scientific_interpretation':sci,'next_action':nxt,'locked_transfer':'LOCKED_UNSCORED','final_accessed':False,'sacred_accessed':False}
    C.rt.atomic_json(result,OUT/'RESULTS.json'); (OUT/'FINAL_REPORT.md').write_text(render(result),encoding='utf-8',newline='\n')
    payload=[]
    for path in sorted((H/'runs').rglob('*')):
      if path.is_file(): payload.append(f"{C.sha(path)}  {path.relative_to(H).as_posix()}")
    for n in ['RESULTS.json','FINAL_REPORT.md']:
      payload.append(f"{C.sha(OUT/n)}  {n}")
    payload.append(f"{C.sha(H/'RUN_LEDGER.json')}  source/RUN_LEDGER.json")
    (OUT/'OUTPUT_SHA256SUMS.txt').write_text('\n'.join(payload)+'\n',encoding='utf-8',newline='\n')
    print(classification)

if __name__=='__main__': main()
