"""Post-stop artifact-only accounting. CPU deserialization, no model execution/updates."""
import json,math,os
from pathlib import Path
import torch
import CONTROLLER as C
R=C.ROOT

def main():
    C.verify()
    assert not (R/'BRANCH_RECEIPT.json').exists()
    a,b=[torch.load(R/n/'restart.pt',map_location='cpu',weights_only=False) for n in ('common_a','common_b')]
    rows=[]; sumsq=0.; count=0; maximum=0.
    for n,x in a['model'].items():
        y=b['model'][n]; d=y.double()-x.double()
        row={'tensor':n,'count':x.numel(),'exact':torch.equal(x,y),'max_abs':float(d.abs().max()),'l2':float(d.norm())}
        rows.append(row); sumsq+=float((d*d).sum()); count+=x.numel(); maximum=max(maximum,row['max_abs'])
    optrows=[]
    for k,sa in a['optimizer']['state'].items():
        for field,x in sa.items():
            y=b['optimizer']['state'][k][field]
            if torch.is_tensor(x):
                optrows.append({'parameter_id':k,'field':field,'exact':torch.equal(x,y),'max_abs':float((y.double()-x.double()).abs().max())})
    diffs=[]
    for u in range(1,101):
        x,y=[C.read(R/n/f'metric_{u:03}.json') for n in ('common_a','common_b')]
        d={k:y[k]-x[k] for k in ('ce','kl','loss','grad_norm') if k in x and x[k]!=y[k]}
        if d: diffs.append({'update':u,'differences_B_minus_A':d})
    details={'model':{'tensors':len(rows),'mismatched':sum(not x['exact'] for x in rows),'max_abs':maximum,'RMS':math.sqrt(sumsq/count)},
             'optimizer':{'fields':len(optrows),'mismatched':sum(not x['exact'] for x in optrows),'max_abs':max(x['max_abs'] for x in optrows)},
             'model_rows':rows,'optimizer_rows':optrows,'metric_differences':diffs,
             'note':'First logged difference is not proof of first differing weights or operator. No kernel mechanism diagnosed.'}
    C.rt.atomic_json(details,R/'MISMATCH_ACCOUNTING.json')
    result={'classification':'SF4_HARD_STOP_COMMON_STATE_REPRODUCTION_MISMATCH','control_updates':0,'treatment_updates':0,
            'common_updates':{'common_a':100,'common_b':100},'common_equivalence':C.read(R/'COMMON_STATE_EQUIVALENCE.json'),
            'mismatch':{k:v for k,v in details.items() if k in ('model','optimizer')},
            'earliest_logged_difference':diffs[0],'differing_metric_records':len(diffs),'common_results':{},
            'branches':'NOT_CREATED','acquisition_endpoint':'NOT_EVALUATED','transfer':'LOCKED_UNSCORED',
            'FINAL':'SEALED_UNACCESSED','sacred':'SEALED_UNACCESSED',
            'protocol_receipt_sha256':C.sha(R/'FREEZE_RECEIPT.json'),'protocol_manifest_sha256':C.sha(R/'PRETRAIN_SHA256SUMS.txt'),
            'historical_classifications':{'SF2':'SF2_ACQUISITION_FAIL','SF3':'SF3_HARD_STOP_UPDATE100_REPLAY_MISMATCH'}}
    for n in ('common_a','common_b'):
        q=C.read(R/n/'update100_checks.json')
        result['common_results'][n]={'acquisition':C.read(R/n/'update100_acquisition_RESULT.json'),
            'language':q['language'],'binding':{k:v['summary'] for k,v in q['binding'].items()},
            'D3':C.read(R/n/'d3_update100_summary.json'),'gates':C.read(R/n/'update100_gates.json'),
            'checkpoint':str(R/n/'checkpoint_100.pt'),'checkpoint_sha256':C.sha(R/n/'checkpoint_100.pt'),
            'restart_sha256':C.sha(R/n/'restart.pt')}
    C.rt.atomic_json(result,R/'RESULTS.json')
    with (R/'RESULTS.jsonl').open('x',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(result)+'\n'); f.flush(); os.fsync(f.fileno())
    lines=['# SF4 final report: hard stop before branching','',
        '**SF4_HARD_STOP_COMMON_STATE_REPRODUCTION_MISMATCH**','',
        'The prospective common-state test failed. No control continuation, treatment continuation, or annealed update ran. No branch receipt was issued. This is an infrastructure-validity failure, not a treatment acquisition failure.','',
        '## Decision and prospective design','',
        'Selected one late English learning-rate schedule variable: constant5e-5 control versus linearly decaying5e-5-to-zero over90 English updates after100. Both would keep binding LR5e-5, exact SF2 factual CE+parentKL, frozen data/batches, parameter scope and restored optimizer/RNG state.',
        'Baby evidence favored this bounded test: the lone historical SF2 error was a -0.030-nat near-tie after an Owen-side shift; head swapping showed upstream-led movement, so a head constraint was less direct. Margin losses add objective assumptions; replay and KL changes target retention that SF2 already preserved; extra constant CE adds budget. Architecture/corpus changes were unsupported. Comparative models were background only.',
        'Two checkpoints do not establish monotonic overshoot. The LR hypothesis was prospective and tentative. It changes integrated step size and LR-scaled decay as well as schedule shape, so even a completed success would not uniquely establish a settling mechanism.',
        'The design predesignated common_a, required two fresh100-update replicas to match exactly, then required identical real branch loads and identical first same-LR continuation update101. Protocol, sources, LR records and gates were sealed before either common prefix ran.','',
        '## Preflight and common-state evidence','',
        'Static/dependency checks passed:65 pinned input/source/runtime hashes, all200 frozen schedule rows, masking, KL-pool exact-array disjointness, inherited semantic/overlap validation, runtime and mock restart/negative-state test. No locked panel was parsed or scored.',
        'Common_a and common_b each passed update0 reproduction and common100 safety gates. Their Python/CPU/GPU RNG states, completed update, English index, scope and provenance matched exactly. All update0 raw acquisition, D3 and binding/language outputs were byte-identical.',
        f"At update100, {details['model']['mismatched']}/{len(rows)} model tensors differed; max absolute delta={maximum:.12g}, RMS={details['model']['RMS']:.12g}. Optimizer differing fields={details['optimizer']['mismatched']}/{details['optimizer']['fields']}.",
        f"The first **logged** difference was update{diffs[0]['update']}: {diffs[0]['differences_B_minus_A']}. {len(diffs)}/100 metric records differed. This does not locate the first changed parameter or low-level operator.",
        'The earlier forensic A/B equality remains historical evidence. This fresh pair did not meet that guarantee despite matching recorded versions and deterministic-algorithm mode. The specific cause is unresolved; no driver/kernel or training-mechanism claim is warranted.','',
        '## Results (common-prefix validation only)','',
        '| Replica | Correct | Exact+EOS | Reversals | Families | DEV CE | PPL | D3 name mass |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for n,v in result['common_results'].items():
        q=v['acquisition']; l=v['language']
        lines.append(f"| {n} | {q['correct']}/16 | {q['exact']}/16 | {q['reversals']}/8 | {q['families']}/4 | {l['loss']:.10f} | {l['perplexity']:.6f} | {v['D3']['mean_combined_name_probability']:.10f} |")
    lines+=['','Both replicas, on each nonsacred pool separately:80/80 answer,80/80 BOTH_DISTINCT, zero collapse,20/20 quartets,40/40 strict reversals; queried-row and downstream correctness conditional on BD both1.0.','',
        '## Frozen gates and stopping','',
        '- Acquisition at200:16/16 correct AND16/16 exact AND8/8 reversals AND4/4 families. **NOT EVALUATED**.',
        '- Language:CE<=own common0CE+0.25. Both common100 prefixes **PASS**.',
        '- Binding:each pool answer>=76,BD>=76,collapse0. Both common100 prefixes **PASS**.',
        '- D3:mean four-name mass<=0.01. Both common100 prefixes **PASS**; this prospective safety gate does not rewrite SF2.',
        '- Common exact model/optimizer/RNG/controller equality: **FAIL**. Mandatory hard stop before branching.',
        '- Control/treatment endpoint classification: **NOT RUN**. No effect estimate.',
        '- All HELDOUT/ALTERNATE/COPY/COMPETING panels: **LOCKED/UNSCORED**. FINAL/sacred untouched.','',
        '## Narrow interpretation and recommended next action','',
        'This study establishes that a fresh fully state-checked branch prerequisite did not pass on the recorded current stack. Similar aggregate scores cannot substitute for the frozen numerical guarantee. The LR hypothesis remains untested. SF2 and SF3 classifications are unchanged.',
        'Exact recommended next action: prospectively instrument the unchanged common-prefix calculation around the first logged divergence (updates11-12), recording forward outputs, gradients and post-Adam tensors to identify the earliest non-reproducible operation. Review that bounded reproducibility diagnostic before authorizing it; do not tune LR or relax equality from these outcomes. No such replay or second experiment was run here.','',
        '## Artifact identities','',f"Protocol receipt: {result['protocol_receipt_sha256']}",f"Protocol manifest: {result['protocol_manifest_sha256']}",'']
    for n,v in result['common_results'].items():
        lines.extend([f"{n} checkpoint: {v['checkpoint']}",v['checkpoint_sha256'],f"Full restart SHA256: {v['restart_sha256']}",''])
    lines+=['Detailed state and optimizer deltas: MISMATCH_ACCOUNTING.json. Raw metrics and predictions remain in common_a/common_b. Sources, protocol and inputs are listed in PRETRAIN_SHA256SUMS.txt; final outputs are listed in SHA256SUMS.txt.','',
        'Dave-coded: the two starting runs look the same on the small scorecard, but they are not the same numerical state. We obeyed the stop gate. Neither annealing nor its control continuation has been tested.']
    (R/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    C.rt.atomic_json({'status':result['classification'],'treatment_updates':0,'control_updates':0,'transfer_opened':False},R/'STATUS.json')
    C.verify()
    payload=[p for p in sorted(R.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p != R/'SHA256SUMS.txt']
    (R/'SHA256SUMS.txt').write_text('\n'.join(C.sha(p)+'  '+p.relative_to(R).as_posix() for p in payload)+'\n',encoding='utf-8',newline='\n')
    for path in payload+[R/'SHA256SUMS.txt']: os.chmod(path,0o444)
    print(json.dumps({'classification':result['classification'],'model_delta':details['model'],'optimizer_delta':details['optimizer'],
        'final_report_sha256':C.sha(R/'FINAL_REPORT.md'),'manifest_sha256':C.sha(R/'SHA256SUMS.txt'),
        'checkpoints':{k:v['checkpoint_sha256'] for k,v in result['common_results'].items()}},indent=2))

if __name__=='__main__': main()
