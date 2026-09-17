"""Isolated, preregistered S1 comparison; historical inputs are read-only."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import random
import shutil
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/selection_s1'
PARENT = ROOT / 'runs/structured_v2r4_seed106001_from6000_terminal/checkpoint_16000.pt'
PARENT_SHA = '94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827'
PANEL = ROOT / 'data/generated/foundation_v2/panels.json'
PROTOCOL = ROOT / 'design/V010_SELECTION_REPAIR_S1.md'

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def write(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')

def queries(item):
    from .isolation_transforms import parse_records, locate_query
    pairs=parse_records(item)
    _,q=locate_query(item)
    assert q is not None
    out=[]
    for index,(key,value) in enumerate(pairs):
        row=copy.deepcopy(item)
        row['input'][q]=key
        row.update(query_key=key,query_index=index,target_span=value[:],target=value+item['target'][-2:])
        assert sum(row['input'][i:i+1+len(value)]==[key]+value for i in range(len(row['input'])))==1
        row['candidate_heads']=[v[0] for _,v in pairs]
        row['query_position']=q
        out.append(row)
    return out

def generate():
    from .data import build_banks, read_u16, LANG_TRAIN
    from .data_v2 import make_item
    from .isolation_transforms import parse_records
    from .v2r4_provenance import FROZEN_PANELS_SHA256,require_frozen_file
    require_frozen_file(PANEL,FROZEN_PANELS_SHA256,'panels')
    assert digest(PARENT)==PARENT_SHA
    if (OUT/'MANIFEST.json').exists(): raise RuntimeError('already frozen')
    banks=build_banks(read_u16(LANG_TRAIN))
    frozen=json.loads(PANEL.read_text())
    denied_inputs=set()
    denied_spans=set()
    for group in frozen.values():
        for row in group:
            denied_inputs.add(tuple(row['input']))
            denied_spans.add(tuple(row['target_span']))
            if row['kind']=='keyed':
                denied_spans.update(tuple(v) for _,v in parse_records(row))
    rng=random.Random(110200)
    dev=[]
    for k in (2,3,4):
        count=0
        while count<48:
            row=make_item(rng,banks,kind='keyed',difficulty='full')
            pairs=parse_records(row)
            if row['pair_count']!=k or len({v[0] for _,v in pairs})!=k: continue
            if any(tuple(v) in denied_spans for _,v in pairs): continue
            group=queries(row)
            if any(tuple(x['input']) in denied_inputs for x in group): continue
            body_id=f'K{k}_{count}'
            for x in group: x['body_id']=body_id
            dev.extend(group)
            denied_spans.update(tuple(v) for _,v in pairs)
            denied_inputs.update(tuple(x['input']) for x in group)
            count+=1
    write(OUT/'DIAGNOSTIC.json',dev)
    audits={}
    for seed in (110001,110002):
        rng=random.Random(110100+seed-110001)
        schedule=[]
        rejected=0
        for step in range(800):
            if rng.random()<.20:
                schedule.append({'task':'language','rng_seed':rng.randrange(2**31)})
                continue
            items=[]
            while len(items)<16:
                kind='keyed' if len(items)<12 else 'induction'
                item=make_item(rng,banks,kind=kind,difficulty='full',low_prior=rng.random()<.20)
                if kind=='keyed':
                    pairs=parse_records(item)
                    if len({v[0] for _,v in pairs})!=len(pairs):
                        rejected+=1;continue
                    all_queries=queries(item)
                    if any(tuple(v) in denied_spans for _,v in pairs) or any(tuple(x['input']) in denied_inputs for x in all_queries):
                        rejected+=1;continue
                    # Reserve two batch slots; if only one remains, use induction.
                    if len(items)>14: continue
                    items.extend(rng.sample(all_queries,2))
                else:
                    if tuple(item['input']) in denied_inputs or tuple(item['target_span']) in denied_spans:
                        rejected+=1;continue
                    items.append(item)
            schedule.append({'task':'structured','items':items})
        write(OUT/f'SCHEDULE_{seed}.json',schedule)
        rows=[x for s in schedule if s['task']=='structured' for x in s['items']]
        assert not any(tuple(x['input']) in denied_inputs or tuple(x['target_span']) in denied_spans for x in rows)
        audits[str(seed)]={'updates':len(schedule),'language_updates':sum(s['task']=='language' for s in schedule),'rows':len(rows),'rejected':rejected,'no_exact_input_or_candidate_span_leakage':True,'keyed_rows':sum(x['kind']=='keyed' for x in rows)}
    files=[PROTOCOL,Path(__file__),PANEL,OUT/'DIAGNOSTIC.json',OUT/'SCHEDULE_110001.json',OUT/'SCHEDULE_110002.json']
    files += [ROOT/'src/baby_v010'/name for name in ('data.py','data_v2.py','model.py','config.py','evaluate.py','train_v2r4.py','isolation_transforms.py')]
    write(OUT/'MANIFEST.json',{'parent_sha256':PARENT_SHA,'files':{str(p.relative_to(ROOT)):digest(p) for p in files},'audits':audits,'diagnostic_rows':len(dev),'diagnostic_bodies':144,'protected_material_opened':False})
    print(json.dumps(audits),flush=True)

def verify():
    m=json.loads((OUT/'MANIFEST.json').read_text())
    for p,h in m['files'].items():
        if digest(ROOT/p)!=h: raise RuntimeError('hash mismatch '+p)
    if digest(PARENT)!=PARENT_SHA: raise RuntimeError('parent mismatch')
    return m

PACK_HOOK = None

def pack(items,device):
    import torch
    n=max(len(x['input'])+len(x['target'])-1 for x in items)
    x=torch.zeros((len(items),n),dtype=torch.long,device=device)
    y=x.clone();mask=torch.zeros_like(x,dtype=torch.bool)
    first=[]
    for i,row in enumerate(items):
        seq=row['input']+row['target']; l=len(seq)-1;s=len(row['input'])-1
        x[i,:l]=torch.tensor(seq[:-1],device=device);y[i,:l]=torch.tensor(seq[1:],device=device)
        mask[i,s:l]=True
        if row['kind']=='keyed': first.append((i,s,row['target'][0]))
    if PACK_HOOK is not None:
        PACK_HOOK(items, x)
    return x,y,mask,first

def model_load():
    import torch
    from .config import BabyVNextConfig
    from .model import BabyVNextLM
    cp=torch.load(PARENT,map_location='cpu',weights_only=False)
    assert cp['update']==16000 and not cp.get('protected_material_opened')
    config=BabyVNextConfig.from_dict(cp['config'])
    model=BabyVNextLM(config).cuda()
    model.load_state_dict(cp['model_state_dict'])
    return model,config

def diagnostic(model,items):
    import torch
    from .evaluate import score_items
    result=[];model.eval()
    with torch.no_grad():
        for s in range(0,len(items),16):
            batch=items[s:s+16]
            scores=score_items(model,batch,torch.device('cuda'))
            x,_,_,_=pack(batch,torch.device('cuda'))
            logits=model(x)
            for i,(item,score) in enumerate(zip(batch,scores)):
                z=logits[i,len(item['input'])-1]
                heads=item['candidate_heads']; values=z[heads].tolist()
                score.update(body_id=item['body_id'],K=item['pair_count'],query_index=item['query_index'],candidate_logits=values,inventory_correct=heads[max(range(len(heads)),key=lambda j:values[j])]==item['target'][0],value_exact=score['emitted'][:len(item['target_span'])]==item['target_span'],rest_lock=all(v==1 for v in score['all_ranks'][1:len(item['target_span'])]))
                result.append(score)
    bodies={}
    for r in result: bodies.setdefault(r['body_id'],[]).append(r)
    acc=[];effects=[];all_success=[]
    for rows in bodies.values():
        acc.append(sum(r['target_rank']==1 for r in rows)/len(rows))
        all_success.append(all(r['target_rank']==1 for r in rows))
        for a in rows:
            for b in rows:
                if a is b:continue
                i,j=a['query_index'],b['query_index']
                effects.append((a['candidate_logits'][i]-a['candidate_logits'][j])-(b['candidate_logits'][i]-b['candidate_logits'][j]))
    summary={'body_accuracy':statistics.mean(acc),'body_all_correct':statistics.mean(all_success),'query_logit_effect':statistics.mean(effects),'mean_margin':statistics.mean(r['first_margin'] for r in result),'rest_lock':statistics.mean(r['rest_lock'] for r in result),'per_K':{str(k):statistics.mean(r['target_rank']==1 for r in result if r['K']==k) for k in (2,3,4)},'body_accuracies':acc}
    return {'summary':summary,'rows':result}

def measure(model,dev,full=True):
    import torch
    from .evaluate import evaluate_panels,language_ce
    from .data import read_u16
    from .train_v2r4 import DEV_STREAM
    d=diagnostic(model,dev)
    stream=torch.tensor(read_u16(DEV_STREAM),dtype=torch.long)
    d['language_dev_ce']=language_ce(model,stream,list(range(0,32*256,256)),torch.device('cuda'),limit=32)
    if full:
        panels=json.loads(PANEL.read_text())
        panels={k:v for k,v in panels.items() if k not in ('all_intact','novel','induction')}
        d['frozen']=evaluate_panels(model,panels,torch.device('cuda'))
        isolation=json.loads((ROOT/'runs/v2r4_isolation_panels/ISOLATION_PANELS.json').read_text())
        d['isolation']=evaluate_panels(model,isolation,torch.device('cuda'))
    return d

def run(arm,seed,until,resume):
    import torch
    import torch.nn.functional as F
    from .train_v2r4 import set_seed,capability_optimizer,language_batch
    from .data import LANG_TRAIN,read_u16
    verify()
    dest=OUT/f'{arm}_{seed}'
    if dest.exists() and not resume:raise RuntimeError('refuse overwrite '+str(dest))
    dest.mkdir(exist_ok=resume)
    set_seed(seed)
    model,config=model_load()
    optimizer,_,_=capability_optimizer(model)
    stream=torch.tensor(read_u16(LANG_TRAIN),dtype=torch.long)
    dev=json.loads((OUT/'DIAGNOSTIC.json').read_text())
    schedule=json.loads((OUT/f'SCHEDULE_{seed}.json').read_text())
    start=time.monotonic()
    start_step=0
    if resume:
        cp=torch.load(dest/'checkpoint_16400.pt',map_location='cpu',weights_only=False)
        assert cp['manifest_sha256']==digest(OUT/'MANIFEST.json')
        model.load_state_dict(cp['model_state_dict']);optimizer.load_state_dict(cp['optimizer_state_dict'])
        baseline=json.loads((dest/'eval_0000.json').read_text());start_step=400
        torch.set_rng_state(cp['torch_rng_state']);torch.cuda.set_rng_state_all(cp['cuda_rng_state'])
    else:
        baseline=measure(model,dev)
        write(dest/'eval_0000.json',baseline)
    print(json.dumps({'arm':arm,'step':0,'summary':baseline['summary'],'ce':baseline['language_dev_ce']}),flush=True)
    initial_ce=baseline['language_dev_ce']
    # Reset dropout RNG after evaluation; identical starting stochastic state.
    if not resume:
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    reason='terminal'
    for step in range(start_step+1,until+1):
        spec=schedule[step-1]
        if shutil.disk_usage(ROOT).free<10*2**30:reason='hard_stop_disk';break
        if time.monotonic()-start>7200:reason='hard_stop_runtime';break
        model.train();optimizer.zero_grad(set_to_none=True)
        if spec['task']=='language':
            x,y=language_batch(stream,random.Random(spec['rng_seed']),16,256,torch.device('cuda'))
            z=model(x);loss=F.cross_entropy(z.flatten(0,1),y.flatten())
        else:
            x,y,mask,first=pack(spec['items'],torch.device('cuda'));z=model(x)
            loss=F.cross_entropy(z[mask],y[mask])
            if arm=='treatment' and first:
                ii,ss,tt=zip(*first)
                loss=loss+F.cross_entropy(z[list(ii),list(ss)],torch.tensor(tt,device='cuda'))
        if not torch.isfinite(loss):reason='hard_stop_nonfinite_loss';break
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
        if not torch.isfinite(norm):reason='hard_stop_nonfinite_gradient';break
        optimizer.step()
        with (dest/'events.jsonl').open('a') as f:f.write(json.dumps({'step':step,'task':spec['task'],'loss':loss.item(),'gradient_norm':norm.item()})+'\n')
        if step%25==0:print(json.dumps({'arm':arm,'step':step,'loss':loss.item(),'elapsed_s':time.monotonic()-start}),flush=True)
        if step%200==0:
            cp={'config':config.to_dict(),'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'parent_checkpoint_sha256':PARENT_SHA,'update':16000+step,'seed':seed,'arm':arm,'protocol':'V010_SELECTION_REPAIR_S1','protected_material_opened':False,'torch_rng_state':torch.get_rng_state(),'cuda_rng_state':torch.cuda.get_rng_state_all(),'manifest_sha256':digest(OUT/'MANIFEST.json')}
            torch.save(cp,dest/f'checkpoint_{16000+step}.pt')
            report=measure(model,dev,full=(step in (400,800)))
            write(dest/f'eval_{step:04d}.json',report)
            print(json.dumps({'arm':arm,'step':step,'summary':report['summary'],'ce':report['language_dev_ce']}),flush=True)
            if report['language_dev_ce']>initial_ce+.20:reason='hard_stop_language';break
            if step==400:
                other=OUT/f'control_{seed}/eval_0400.json'
                if arm=='treatment' and other.exists():
                    control=json.loads(other.read_text())
                    a=report['summary'];b=baseline['summary']
                    if a['body_accuracy']-b['body_accuracy']<.05 and control['summary']['body_accuracy']-b['body_accuracy']<.05 and a['mean_margin']-b['mean_margin']<.25:
                        reason='futility';break
    write(dest/f'RECEIPT_{step:04d}.json',{'reason':reason if until==800 or reason!='terminal' else 'matched_futility_boundary','last_step':step,'parent_sha256':digest(PARENT),'manifest_sha256':digest(OUT/'MANIFEST.json'),'elapsed_s':time.monotonic()-start,'artifacts':{p.name:digest(p) for p in dest.iterdir() if p.is_file()}})

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['generate','run']);p.add_argument('--arm',choices=['control','treatment']);p.add_argument('--seed',type=int,default=110001);p.add_argument('--until',type=int,choices=[400,800],default=400);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.action=='generate':generate()
    else:run(a.arm,a.seed,a.until,a.resume)

if __name__=='__main__':main()
