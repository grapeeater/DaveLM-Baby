"""Verify frozen S1 data and score a tiny parent batch; no optimizer steps."""
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.baby_v010 import selection_s1 as s

def main():
    s.verify()
    dev=json.loads((s.OUT/'DIAGNOSTIC.json').read_text())
    bybody={}
    for row in dev:bybody.setdefault(row['body_id'],[]).append(row)
    for group in bybody.values():
        assert len(group)==group[0]['pair_count']
        for a,b in zip(group,group[1:]):
            assert [i for i,(x,y) in enumerate(zip(a['input'],b['input'])) if x!=y]==[a['query_position']]
            assert a['target'][0]!=b['target'][0]
    schedules={seed:json.loads((s.OUT/f'SCHEDULE_{seed}.json').read_text()) for seed in (110001,110002)}
    for schedule in schedules.values():
        for batch in schedule:
            if batch['task']=='language':continue
            items=batch['items']
            assert len(items)==16 and sum(r['kind']=='keyed' for r in items)==12
            for a,b in zip(items[:12:2],items[1:12:2]):
                assert [i for i,(x,y) in enumerate(zip(a['input'],b['input'])) if x!=y]==[a['query_position']]
    model,_=s.model_load();model.eval()
    items=next(x['items'] for x in schedules[110001] if x['task']=='structured')[:2]
    x,y,mask,first=s.pack(items,torch.device('cuda'))
    for i,j,t in first:assert y[i,j].item()==t and x[i,j].item()==items[i]['input'][-1] and mask[i,j]
    began=time.monotonic()
    with torch.no_grad():
        z=model(x)
        ce=torch.nn.functional.cross_entropy(z[mask],y[mask])
        assert torch.isfinite(ce)
        # Causal padding check: a row scored alone equals its padded batch state.
        n=len(items[0]['input'])-1
        solo=model(x[:1,:n+1])[0,-1]
        maxdiff=(solo-z[0,n]).abs().max().item()
        assert maxdiff<1e-3
    report={'status':'PASS','optimizer_updates':0,'parent_sha256':s.digest(s.PARENT),'manifest_sha256':s.digest(s.OUT/'MANIFEST.json'),'counterfactual_bodies':len(bybody),'schedule_seeds':[110001,110002],'loss_finite':True,'padding_max_logit_difference':maxdiff,'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'smoke_seconds':time.monotonic()-began,'protected_material_opened':False}
    s.write(s.OUT/'PREFLIGHT.json',report)
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
