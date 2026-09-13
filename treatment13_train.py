"""Authoritative T13 learned mapping-row localization trainer."""
from __future__ import annotations
import argparse, hashlib, json, random, sys, time
from pathlib import Path
from typing import Any, Dict
import torch
import torch.nn.functional as F

from treatment13_config import *
from treatment13_model import Treatment13Model

MAX_STEPS = 1000
LEARNING_RATE = 3.0e-4
WEIGHT_DECAY = 0.05
GRADIENT_CLIP_NORM = 2.0

SCHEDULE_PATH = OUT_ROOT / "treatment13_frozen_schedule.json"
TRAINING_METRICS_PATH = OUT_ROOT / "treatment13_training_metrics.jsonl"
TRAINING_RESULT_PATH = OUT_ROOT / "treatment13_training_result.json"
CHECKPOINT_ROOT = OUT_ROOT / "checkpoints" / "learned_mapping_row_localization" / "seed_8380"
FINAL_CHECKPOINT_PATH = CHECKPOINT_ROOT / "latest.pt"

def require(c, m):
    if not c: raise RuntimeError(m)
def sha256_file(p: Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()
def read_json(p): return json.loads(p.read_text(encoding='utf-8'))
def extract_state(c: Any):
    for k in ('model_state_dict','model_state','model','state_dict'):
        if isinstance(c,dict) and isinstance(c.get(k),dict): return c[k]
    raise RuntimeError('unsupported checkpoint')

def build_schedule(pool):
    ids=[q['quartet_id'] for q in pool['quartets']]
    require(len(ids)==200 and len(set(ids))==200,'quartet count')
    rng=random.Random(1_100_000 + SEED)
    steps=[]
    for epoch in range(MAX_STEPS // 25):
        order=list(ids); rng.shuffle(order)
        for j in range(0,len(order),8):
            chunk=order[j:j+8]; require(len(chunk)==8,'schedule chunk')
            steps.append({'step':len(steps)+1,'epoch':epoch+1,'quartet_ids':chunk})
    require(len(steps)==MAX_STEPS,'schedule length')
    return {'artifact_type':'treatment13_frozen_schedule','experiment':NAME,'seed':SEED,
            'steps':MAX_STEPS,'batch_size':BATCH_SIZE,'quartets_per_step':8,
            'documents_per_step':BATCH_SIZE,'presentations_per_quartet':40,
            'schedule_document_events':MAX_STEPS*BATCH_SIZE,
            'steps_data':steps}

def build_batch(step, byq, device):
    docs=[]
    for qid in step['quartet_ids']: docs.extend(byq[qid]['docs'])
    require(len(docs)==BATCH_SIZE,'batch docs')
    x=torch.tensor([d['full_document_token_ids'] for d in docs],dtype=torch.long,device=device)
    q=torch.tensor([d['qdp'] for d in docs],dtype=torch.long,device=device)
    a=torch.tensor([d['answer_causal_position'] for d in docs],dtype=torch.long,device=device)
    y=torch.tensor([d['target_value_token'] for d in docs],dtype=torch.long,device=device)
    d=torch.tensor([d['distractor_value_token'] for d in docs],dtype=torch.long,device=device)
    return docs,x,q,a,y,d

def train_step(model,opt,batch):
    docs,x,q,a,y,d=batch; opt.zero_grad(set_to_none=True)
    logits,extra=model(x,q,a)
    rows=torch.arange(x.shape[0],device=x.device)
    al=logits[rows,a,:]
    loss=F.cross_entropy(al,y); require(torch.isfinite(loss).item(),'nonfinite loss')
    loss.backward(); gn=torch.nn.utils.clip_grad_norm_(model.parameters(),GRADIENT_CLIP_NORM)
    require(torch.isfinite(gn).item(),'nonfinite grad norm'); opt.step()
    with torch.no_grad():
        marg=al[rows,y]-al[rows,d]
        rw=extra['row_weights']; loc=extra['localization_attention']
    return {'answer_ce_loss':float(loss.item()),'answer_accuracy':float((al.argmax(-1)==y).float().mean().item()),
            'target_gt_distractor_rate':float((marg>0).float().mean().item()),
            'mean_target_distractor_margin':float(marg.mean().item()),
            'localizer_mean_max_attention':float(loc.max(dim=1).values.mean().item()),
            'retrieval_argmax_rate':float((rw.argmax(-1)==torch.tensor([0 if int(z['query_slot'])==0 else 1 for z in docs],device=x.device)).float().mean().item()),
            'supervised_answer_decisions':int(y.numel())}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--device',default='cuda'); ap.add_argument('--smoke-test',action='store_true'); args=ap.parse_args()
    require(torch.cuda.is_available() if args.device=='cuda' else True,'GPU unavailable')
    device=torch.device(args.device); random.seed(SEED); torch.manual_seed(SEED)
    pre=read_json(PREFLIGHT_RESULT_PATH); require(pre['status']=='TREATMENT13_LOCALIZATION_PREFLIGHT_PASS','preflight')
    require(sha256_file(TRAIN_POOL_PATH)==pre['provenance']['train_pool_sha256'],'train hash')
    pool=read_json(TRAIN_POOL_PATH); byq={q['quartet_id']:q for q in pool['quartets']}
    schedule=build_schedule(pool); SCHEDULE_PATH.write_text(json.dumps(schedule,indent=1),encoding='utf-8')
    model=Treatment13Model().to(device); model.train()
    ck=torch.load(START_CHECKPOINT_PATH,map_location=device); base=extract_state(ck)
    model.base_model.load_state_dict(base)
    require(model.base_model_parameter_count()==10594944,'base parameter count')
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0.05)
    if args.smoke_test:
        model.eval(); batch=build_batch(schedule['steps_data'][0],byq,device)
        with torch.no_grad(): logits,ex=model(batch[1],batch[2],batch[3])
        print({'status':'TREATMENT13_GPU_SMOKE_PASS','device_name':torch.cuda.get_device_name(0),'logits':list(logits.shape),'loc':list(ex['localization_attention'].shape),'optimizer_steps':0})
        return 0
    start=time.time(); metrics=[]
    for i,s in enumerate(schedule['steps_data']):
        rec=train_step(model,opt,build_batch(s,byq,device)); rec.update(step=i+1,elapsed_seconds=time.time()-start); metrics.append(rec)
        if (i+1)%50==0: print(rec,flush=True)
    TRAINING_METRICS_PATH.write_text(''.join(json.dumps(x)+'\n' for x in metrics),encoding='utf-8')
    CHECKPOINT_ROOT.mkdir(parents=True,exist_ok=True)
    torch.save({'model_state_dict':model.state_dict(),'step':MAX_STEPS,'seed':SEED,'objective':'answer_only_causal_cross_entropy_with_learned_mapping_row_localization'},FINAL_CHECKPOINT_PATH)
    result={'status':'TREATMENT13_TRAINED','steps':MAX_STEPS,'batch_size':BATCH_SIZE,'seed':SEED,'schedule_sha256':sha256_file(SCHEDULE_PATH),'final_checkpoint_sha256':sha256_file(FINAL_CHECKPOINT_PATH),'final_step':metrics[-1]}
    TRAINING_RESULT_PATH.write_text(json.dumps(result,indent=1),encoding='utf-8'); print(result)
if __name__=='__main__':
    try: main()
    except RuntimeError as e: print(f'TRAIN ABORT: {e}',file=sys.stderr); raise
