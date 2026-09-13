"""Exact common-state test and irreversible pre-outcome branch receipt."""
import json, os
from pathlib import Path
import torch
import CONTROLLER as C
ROOT=C.ROOT

def main():
    C.verify()
    assert not (ROOT/'BRANCH_RECEIPT.json').exists()
    states=[]
    for arm in ('common_a','common_b'):
        assert not (ROOT/arm/'HARD_STOP.json').exists()
        assert C.read(ROOT/arm/'STATUS.json')['status']=='COMMON_COMPLETE'
        s=torch.load(ROOT/arm/'restart.pt',map_location='cpu',weights_only=False)
        assert s['completed']==100 and s['english_index']==90 and s['scope']=='binding'
        assert s['provenance']==C.provenance()
        states.append(s)
    a,b=states
    fields=['model','optimizer','rng','scope','completed','english_index','provenance']
    checks={k:C.equality(a[k],b[k]) for k in fields}
    metric_equal=all(C.read(ROOT/'common_a'/f'metric_{i:03}.json')==C.read(ROOT/'common_b'/f'metric_{i:03}.json') for i in range(1,101))
    for u in (0,100):
        for n in (f'update{u}_acquisition_RAW.jsonl',f'update{u}_checks.json',f'd3_update{u}_RAW.jsonl'):
            checks[n]=C.sha(ROOT/'common_a'/n)==C.sha(ROOT/'common_b'/n)
    checks['all100_metric_records_exact']=metric_equal
    C.rt.atomic_json({'checks':checks,'all_exact':all(checks.values()),
                     'model_tensors':len(a['model']),'optimizer_parameters':len(a['optimizer']['state']),
                     'source_sha256':{ar:C.sha(ROOT/ar/'restart.pt') for ar in ('common_a','common_b')}},ROOT/'COMMON_STATE_EQUIVALENCE.json')
    assert all(checks.values()), 'HARD_STOP_COMMON_REPRODUCTION_FAILED'
    shared={k:v for k,v in a.items() if k!='arm'}
    C.rt.atomic_torch_save(shared,ROOT/'COMMON_STATE.pt')
    for arm in ('control','treatment'):
        C.rt.atomic_torch_save({**shared,'arm':arm},ROOT/f'{arm}_initial.pt')
        chk=torch.load(ROOT/f'{arm}_initial.pt',map_location='cpu',weights_only=False)
        assert C.equality(shared,{k:v for k,v in chk.items() if k!='arm'})
    names=['COMMON_STATE.pt','control_initial.pt','treatment_initial.pt','COMMON_STATE_EQUIVALENCE.json']
    r={'status':'COMMON_STATE_EXACT_AND_BRANCHES_FROZEN','provenance':C.provenance(),
       'completed':100,'next_update':101,'state_digest':C.state_digest(shared),
       'files':{n:C.sha(ROOT/n) for n in names},'treatment_updates_executed':0,'control_updates_executed':0}
    C.rt.atomic_json(r,ROOT/'BRANCH_RECEIPT.json')
    (ROOT/'BRANCH_RECEIPT.sha256').write_text(C.sha(ROOT/'BRANCH_RECEIPT.json')+'  BRANCH_RECEIPT.json\n')
    for n in names+['BRANCH_RECEIPT.json','BRANCH_RECEIPT.sha256']:
        os.chmod(ROOT/n,0o444)
    C.check_branch_receipt()
    print(json.dumps(r,indent=2),flush=True)

if __name__=='__main__': main()
