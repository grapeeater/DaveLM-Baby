"""Prespecified classification and legally gated transfer, after both endpoints exist."""
import json, math, os
from pathlib import Path
import torch
import CONTROLLER as C
E,ROOT=C.E,C.ROOT

def main():
    p,b,base,_,_,_,rows,_=C.verify()
    C.check_branch_receipt()
    result={'study':p['study'],'arms':{},'comparisons':{},'transfer':{},'historical_status_unchanged':True}
    for arm in ('control','treatment'):
        out=ROOT/arm
        assert not (out/'HARD_STOP.json').exists()
        status=C.read(out/'STATUS.json'); assert status['status']=='ARM_ENDPOINT_COMPLETE' and status['completed']==200
        cp=out/'checkpoint_200.pt'; assert C.sha(cp)==status['checkpoint_sha256']
        a=C.read(out/'update200_acquisition_RESULT.json')
        checks=C.read(out/'update200_checks.json'); d3=C.read(out/'d3_update200_summary.json')
        gates=C.read(out/'update200_gates.json'); gates['acquisition']=E.acquire(a)
        trajectories={str(u):C.read(out/f'update{u}_acquisition_RESULT.json') for u in (100,125,150,175,200)}
        raw=[json.loads(x) for x in (out/'update200_acquisition_RAW.jsonl').read_text(encoding='utf-8').splitlines()]
        byname={}
        for name in ('Alex','Owen','Mia','Nora'):
            rs=[r for r in raw if r['candidates'][r['correct_index']].strip().rstrip('.')==name]
            byname[name]={'n':len(rs),'correct':sum(r['correct'] for r in rs),'exact':sum(r['exact'] for r in rs),
                          'mean_margin':sum(r['margin'] for r in rs)/len(rs),'min_margin':min(r['margin'] for r in rs)}
        result['arms'][arm]={'checkpoint':str(cp),'sha256':C.sha(cp),'completed_updates':200,
            'continuation_updates':100,'acquisition':a,'gates':gates,'by_correct_name':byname,
            'language':checks['language'],'D3':d3,
            'binding':{k:v['summary'] for k,v in checks['binding'].items()},'trajectory':trajectories,
            'eligible':all(gates.values())}
    ar=[result['arms'][x]['acquisition'] for x in ('control','treatment')]
    result['comparisons']['treatment_minus_control']={k:ar[1][k]-ar[0][k] for k in ('correct','exact','reversals','families','mean_margin','min_margin')}
    raw={arm:{r['id']:r for r in [json.loads(x) for x in (ROOT/arm/'update200_acquisition_RAW.jsonl').read_text(encoding='utf-8').splitlines()]} for arm in ('control','treatment')}
    pairs={}; families={}
    differences=[]
    for key,c in raw['control'].items():
        t=raw['treatment'][key]
        differences.append({'id':key,'control_margin':c['margin'],'treatment_margin':t['margin'],
                            'difference':t['margin']-c['margin'],'control_correct':c['correct'],
                            'treatment_correct':t['correct'],'control_exact':c['exact'],'treatment_exact':t['exact']})
        pairs.setdefault(c['pair_id'],[]).append(key); families.setdefault(c['family_id'],[]).append(key)
    result['paired_items']=differences
    result['paired_reversals']=[{'pair_id':k,**{arm:all(raw[arm][i]['correct'] for i in ids) for arm in raw}} for k,ids in pairs.items()]
    result['paired_families']=[{'family_id':k,**{arm:{'complete':all(raw[arm][i]['correct'] for i in ids),
        'mean_margin':sum(raw[arm][i]['margin'] for i in ids)/len(ids)} for arm in raw}} for k,ids in families.items()]
    C.rt.atomic_json(result,ROOT/'ENDPOINT_RESULTS_BEFORE_TRANSFER.json')
    # Only qualified arms may access panel content; a failed arm never receives transfer inference.
    for arm in ('control','treatment'):
        if not result['arms'][arm]['eligible']:
            result['transfer'][arm]={'status':'LOCKED_UNSCORED'}; continue
        device=torch.device('cuda')
        m=C.rt.load_model(ROOT/arm/'checkpoint_200.pt',device,b)
        tok=E.Tokenizer.from_file(base['tokenizer'])
        result['transfer'][arm]={'status':'LEGALLY_OPENED_AFTER_ALL_ENDPOINT_GATES'}
        for label in ('HELDOUT','ALTERNATE','COPY','COMPETING'):
            result['transfer'][arm][label]=E.panel(m,b,ROOT/arm,label.lower(),C.read(b/(label+'.json')),tok,device)
        del m; torch.cuda.empty_cache()
    t,c=result['arms']['treatment'],result['arms']['control']
    if t['eligible'] and not c['eligible']: classification='SF4_TREATMENT_ACQUISITION_PASS_CONTROL_FAIL'
    elif t['eligible'] and c['eligible']: classification='SF4_BOTH_ACQUISITION_PASS'
    elif c['eligible']: classification='SF4_CONTROL_PASS_TREATMENT_ACQUISITION_FAIL'
    else: classification='SF4_MATCHED_STUDY_COMPLETE_ACQUISITION_FAIL'
    result['classification']=classification
    result['common_equivalence']=C.read(ROOT/'COMMON_STATE_EQUIVALENCE.json')
    result['receipt_hashes']={'protocol_receipt':C.sha(ROOT/'FREEZE_RECEIPT.json'),
        'protocol_manifest':C.sha(ROOT/'PRETRAIN_SHA256SUMS.txt'),'branch_receipt':C.sha(ROOT/'BRANCH_RECEIPT.json'),
        'common_state':C.sha(ROOT/'COMMON_STATE.pt')}
    result['scope']='TRAIN acquisition plus only individually gate-authorized transfer; no FINAL/sacred'
    C.rt.atomic_json(result,ROOT/'RESULTS.json')
    with (ROOT/'RESULTS.jsonl').open('x',encoding='utf-8',newline='\n') as f:
        for arm,rec in result['arms'].items(): f.write(json.dumps({'arm':arm,**rec})+'\n')
        for row in differences: f.write(json.dumps({'type':'paired_item',**row})+'\n')
        f.flush(); os.fsync(f.fileno())
    lines=['# SF4 common-state late-English LR study','',classification,'',
        'This is one prospective matched study with one manipulated variable: late English learning-rate sequence.',
        'Control stayed at 5e-5; treatment decreased linearly from 5e-5 to zero over 90 English updates after100.',
        'Binding LR, parent KL, factual loss, data, scope, seed and optimizer state policy were identical.',
        'Historical SF2_ACQUISITION_FAIL and SF3_HARD_STOP_UPDATE100_REPLAY_MISMATCH remain unchanged.','',
        '## Validity','',
        'Two fresh100-update common prefixes matched exactly in model, optimizer, RNG, controller state, all100 training metric records and baseline/end-of-prefix outputs. Both real branch loaders restored exactly the same model, optimizer and RNG. Branch hashes were sealed before either continuation.',
        'The control is prospective under the current environment; no historical tensor equivalence was required.',
        'The first post-branch update has identical LR in both arms and is an additional matched numerical control.',
        '', '## Endpoint results','',
        '| Arm | Correct | Exact+EOS | Reversals | Families | CE | PPL | D3 mass | Acquisition |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for arm,v in result['arms'].items():
        q=v['acquisition']; l=v['language']
        lines.append(f"| {arm} | {q['correct']}/16 | {q['exact']}/16 | {q['reversals']}/8 | {q['families']}/4 | {l['loss']:.8f} | {l['perplexity']:.5f} | {v['D3']['mean_combined_name_probability']:.8f} | {'PASS' if v['gates']['acquisition'] else 'FAIL'} |")
    lines+=['','All four acquisition counts were required exactly at update200. Language CE allowed at most+0.25 nat versus common update0. Each binding pool required answer>=76/80, BD>=76/80, collapse0. D3 required mean four-name mass<=0.01. No best-checkpoint selection.','',
             '## Trajectory and retention','']
    for arm,v in result['arms'].items():
        lines += [f'### {arm}','',json.dumps({'gates':v['gates'],'binding':v['binding'],'by_correct_name':v['by_correct_name']},indent=2),'',
                  '| Update | Correct | Exact | Reversals | Families | Minimum margin |','|---|---:|---:|---:|---:|---:|']
        for u,q in v['trajectory'].items(): lines.append(f"| {u} | {q['correct']} | {q['exact']} | {q['reversals']} | {q['families']} | {q['min_margin']:.8f} |")
        lines+=['',f"Transfer: {result['transfer'][arm]['status']}",'']
    lines+=['## Paired effect and limitations','',json.dumps(result['comparisons'],indent=2),'',
        'The paired raw item, reversal and family results are preserved in RESULTS.json. Four repeatedly trained families and one stochastic seed do not support population-level uncertainty or broad generalization claims.',
        'The intervention changes both late step-size profile and total English step size, including AdamW decay scaling and relative effective binding pressure; this study cannot attribute an effect specifically to smoothing/settling versus less total adaptation.',
        'A failing schedule does not falsify every LR-decay scheme. A success establishes only the frozen training-family acquisition gate and any separately reported authorized transfer.','',
        '## Artifacts','',json.dumps(result['receipt_hashes'],indent=2),'']
    for arm,v in result['arms'].items(): lines += [f"{arm}: {v['checkpoint']}",v['sha256'],'']
    lines+=['## Interpretation and exact next action','',
        'If treatment alone passes, this schedule improves this fixed common-state endpoint; request review before any replication. If both pass, no binary gate advantage is established; compare frozen margin/retention evidence. If treatment fails, the specified decay did not achieve the target; do not rescue or tune it.',
        'Recommended next action: review this completed paired result and its full frozen per-item trajectory before authorizing another experiment. No second treatment is started.','',
        'Dave-coded: the common state is shared and the comparison is real; call the endpoint exactly by the gates, regardless of how close a failed arm comes.']
    (ROOT/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    C.verify(); C.check_branch_receipt()
    hashes=[]
    for path in sorted(ROOT.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path!=ROOT/'SHA256SUMS.txt':
            hashes.append(C.sha(path)+'  '+path.relative_to(ROOT).as_posix())
    (ROOT/'SHA256SUMS.txt').write_text('\n'.join(hashes)+'\n')
    print(json.dumps({'classification':classification,'arms':{a:{'acquisition':v['acquisition'],'gates':v['gates'],'sha256':v['sha256']} for a,v in result['arms'].items()},'manifest':C.sha(ROOT/'SHA256SUMS.txt')},indent=2),flush=True)

if __name__=='__main__': main()
