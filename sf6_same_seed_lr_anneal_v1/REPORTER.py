"""Frozen descriptive paired analysis. No inference or treatment selection."""
import json, math
from pathlib import Path
import CONTROLLER as C

def mean(v): return sum(v)/len(v)

def main():
    p=C.verify(); results={}; pairs=[]
    for r in p['runs']:
        key=f"seed_{r['seed']}_{r['arm']}"; d=C.H/'runs'/key
        status=C.read(d/'STATUS.json'); u=status['completed']
        assert C.sha(status['checkpoint'])==status['checkpoint_sha256']
        trajectory={}
        for t in [0,100,200]:
            if not (d/f'update{t}_acquisition_RESULT.json').exists(): continue
            trajectory[str(t)]={'acquisition':C.read(d/f'update{t}_acquisition_RESULT.json'),
                'checks':C.read(d/f'update{t}_checks.json'),'d3':C.read(d/f'd3_update{t}_summary.json')}
        results[key]={'seed':r['seed'],'arm':r['arm'],'status':status,'trajectory':trajectory}
    for seed in p['seeds']:
        c=results[f'seed_{seed}_control']; t=results[f'seed_{seed}_anneal']
        cp=c['status']['gates']['endpoint_pass']; tp=t['status']['gates']['endpoint_pass']
        q={'seed':seed,'control_pass':cp,'anneal_pass':tp,
           'outcome':'both' if cp and tp else 'anneal_only' if tp else 'control_only' if cp else 'neither','paired_margins':{}}
        margins={}
        for u in [100,200]:
            paths=[C.H/'runs'/f'seed_{seed}_{arm}'/f'update{u}_acquisition_RAW.jsonl' for arm in ['control','anneal']]
            if not all(x.exists() for x in paths): continue
            rr=[{v['id']:v for v in [json.loads(l) for l in x.read_text(encoding='utf-8').splitlines()]} for x in paths]
            assert rr[0].keys()==rr[1].keys()
            dif={k:rr[1][k]['margin']-rr[0][k]['margin'] for k in rr[0]}
            margins[u]=dif
            q['paired_margins'][str(u)]={'per_item_anneal_minus_control':dif,'mean':mean(list(dif.values())),
                 'mean_absolute':mean([abs(v) for v in dif.values()]),'max_absolute':max(abs(v) for v in dif.values()),
                 'residual_item_delta':dif['TRAIN:g0:carried:wooden boat:a0']}
        if 100 in margins and 200 in margins:
            dd={k:margins[200][k]-margins[100][k] for k in margins[200]}
            q['prefix_adjusted_delta200_minus_delta100']={'per_item':dd,'mean':mean(list(dd.values()))}
        pairs.append(q)
    wins=sum(x['outcome']=='anneal_only' for x in pairs); losses=sum(x['outcome']=='control_only' for x in pairs)
    classification='SUGGESTIVE_ANNEAL_BENEFIT' if wins>=2 and losses==0 else 'CONTROL_FAVORED_OR_MIXED' if losses else 'INCONCLUSIVE'
    if any(x['status']['completed']<200 for x in results.values()): classification='INCONCLUSIVE_EARLY_STOP'
    out={'classification':classification,'anneal_only':wins,'control_only':losses,'pairs':pairs,'runs':results,
         'transfer':'LOCKED_UNSCORED','final_sacred_accessed':False,'no_followup_experiment':True}
    C.rt.atomic_json(out,C.H/'RESULTS.json')
    lines=['# SF6 — same-seed independent late-English-LR annealing', '',f'**{classification}**. Three preregistered pairs; six independent Pilot1 starts. No trajectory equality requirement.','',
      'The only training-variable difference is the late English AdamW LR. Binding LR, data, ordering, KL, scopes and evaluation are unchanged. Historical SF1/SF2/SF3/SF4/SF5 classifications remain unchanged.','',
      '| Seed | Arm | Update | Correct /16 | Exact /16 | Reversals /8 | Families /4 | CE | PPL | D3 | All gates |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in p['runs']:
        x=results[f"seed_{r['seed']}_{r['arm']}"]; u=x['status']['completed']; z=x['trajectory'][str(u)]; a=z['acquisition']; l=z['checks']['language']
        lines.append(f"| {r['seed']} | {r['arm']} | {u} | {a['correct']} | {a['exact']} | {a['reversals']} | {a['families']} | {l['loss']:.6f} | {l['perplexity']:.3f} | {z['d3']['mean_combined_name_probability']:.8f} | {x['status']['status']} |")
    lines+=['','## Paired evidence','',f'Anneal-only full-gate successes: **{wins}/3**. Control-only: **{losses}/3**. These are descriptive seed-level observations, not a population claim.','']
    for q in pairs:
        lines.append(f"- Seed {q['seed']}: {q['outcome']}. Paired margin summaries: `{json.dumps(q['paired_margins'])}`")
    lines+=['','The update100 differences measure numerical variation before the intervention. They do not bound possible later variation. Historical ~0.00697-nat observed variation was context only. No bitwise mismatch invalidated a run, no tolerance was tuned, and no seed was excluded or replaced. N=3 cannot provide strong statistical certainty.','',
      '## Every frozen gate and retention pool','']
    for key,x in results.items():
        lines += [f'### {key}', '',f"Status: **{x['status']['status']}**. Gates: `{json.dumps(x['status']['gates'])}`",'']
        for u,z in x['trajectory'].items():
            lines.append(f"Update {u}: acquisition `{json.dumps(z['acquisition'])}`; language `{json.dumps(z['checks']['language'])}`; D3 `{json.dumps(z['d3'])}`.")
            for pool,v in z['checks']['binding'].items():
                lines.append(f"- {pool}: `{json.dumps(v['summary'])}`, gate `{v['gate']}`.")
        lines += ['',f"Checkpoint: `{x['status']['checkpoint']}`",f"SHA-256: `{x['status']['checkpoint_sha256']}`",'']
    lines+=['## Interpretation and limits','',
      'An ACQUISITION_SUCCESS is exactly the frozen training-family endpoint plus all retention gates. It does not establish transfer, broad English, or a settling/overshoot mechanism. Annealing also changes LR-scaled decay and effective late update size. Failure receives no rescue. All transfer/copy/competing-name/FINAL/sacred panels remained locked and unscored.','',
      'See TREATMENT_DECISION.md for the prospective critique and rejected alternatives; PROTOCOL.json for frozen rules; PREFLIGHT.json and per-run BASELINE_REPRODUCTION.json for integrity controls. Raw scores, generations, D3 positions, binding rows and training metrics are retained per run.','',
      'Dave-coded: We finally took the annealing shot against same-seed controls. The table says whether it clears the actual gate; a close miss stays a miss. No hidden exam was opened.','',
      'Next action: review this completed study and its paired outcomes before any transfer evaluation or further treatment. No second experiment was started.','',
      f"Receipt SHA-256: `{C.sha(C.H/'FREEZE_RECEIPT.json')}`",f"Frozen manifest SHA-256: `{C.sha(C.H/'SHA256SUMS.txt')}`"]
    (C.H/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    # Hash complete outputs without a circular manifest; exclude rolling binary state (superseded by final checkpoint).
    files=[C.H/'RESULTS.json',C.H/'FINAL_REPORT.md']
    files += sorted(x for x in (C.H/'runs').rglob('*') if x.is_file() and x.name!='restart.pt')
    files += sorted(x for x in (C.H/'logs').glob('*.log') if x.is_file())
    receipt=''.join(f'{C.sha(x)}  {x.relative_to(C.H).as_posix()}\n' for x in files)
    (C.H/'OUTPUT_SHA256SUMS.txt').write_text(receipt,encoding='utf-8',newline='\n')
    for line in receipt.splitlines():
        h,n=line.split('  ',1); assert C.sha(C.H/n)==h
    print('STUDY_COMPLETE',classification,'OUTPUT_MANIFEST',C.sha(C.H/'OUTPUT_SHA256SUMS.txt'),flush=True)

if __name__=='__main__': main()
