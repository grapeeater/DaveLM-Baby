"""Frozen T13 retention exam; no gradients or parameter updates."""
from __future__ import annotations
import argparse, hashlib, json, statistics, sys
from collections import defaultdict
from pathlib import Path
import torch
from treatment13_config import *
from treatment13_model import Treatment13Model

FINAL_CHECKPOINT_PATH = OUT_ROOT / 'checkpoints' / 'learned_mapping_row_localization' / 'seed_8380' / 'latest.pt'
FINAL_AUDIT_RESULT_PATH = OUT_ROOT / 'treatment13_final_retention_audit.json'

def sha256_file(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def state(c):
    for k in ('model_state_dict','model_state','model','state_dict'):
        if isinstance(c,dict) and isinstance(c.get(k),dict): return c[k]
    raise RuntimeError('unsupported checkpoint')
def row_positions(d):
    q=int(d['query_slot']); a=int(d['query_key_clause_pos']); b=int(d['target_clause_pos']); c=int(d['other_key_pos']); e=int(d['distractor_value_pos'])
    return (a,b,c,e) if q==0 else (c,e,a,b)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--device',default='cuda'); ap.add_argument('--batch-size',type=int,default=32); ap.add_argument('--pool',choices=['retention','train'],default='retention'); args=ap.parse_args()
    if args.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('GPU unavailable')
    dev=torch.device(args.device); pool_path = RETENTION_POOL_PATH if args.pool=='retention' else TRAIN_POOL_PATH; pool=json.loads(pool_path.read_text(encoding='utf-8')); docs=[d for q in pool['quartets'] for d in q['docs']]
    model=Treatment13Model().to(dev); model.load_state_dict(state(torch.load(FINAL_CHECKPOINT_PATH,map_location=dev))); model.eval()
    rows=[]
    with torch.no_grad():
      for st in range(0,len(docs),args.batch_size):
        ch=docs[st:st+args.batch_size]; x=torch.tensor([d['full_document_token_ids'] for d in ch],dtype=torch.long,device=dev); q=torch.tensor([d['qdp'] for d in ch],dtype=torch.long,device=dev); a=torch.tensor([d['answer_causal_position'] for d in ch],dtype=torch.long,device=dev); logits,ex=model(x,q,a); idx=torch.arange(len(ch),device=dev)
        for i,d in enumerate(ch):
          v=logits[i,a[i]].float(); probs=torch.softmax(v,dim=-1); target=int(d['target_value_token']); dist=int(d['distractor_value_token']); rp=row_positions(d); att=ex['localization_attention'][i]; pos0=1+int(att[:,0].argmax()); pos1=1+int(att[:,1].argmax()); true={rp[0],rp[2]}; top_hit=int(pos0 in true or pos1 in true); true_mass=float(att[[rp[0]-1,rp[2]-1],:].sum().item())
          rows.append({'doc_id':d['doc_id'],'quartet_id':d['quartet_id'],'query_slot':int(d['query_slot']),'orientation':int(d['orientation']),'member':d['member'],'layout_combo':d['layout_combo'],'target':target,'distractor':dist,'predicted':int(v.argmax().item()),'answer_correct':int(v.argmax().item())==target,'target_prob':float(probs[target].item()),'distractor_prob':float(probs[dist].item()),'candidate_mass':float((probs[target]+probs[dist]).item()),'margin':float((v[target]-v[dist]).item()),'loc_top_positions':[pos0,pos1],'true_source_positions':[rp[0],rp[2]],'localization_source_top_hit':top_hit,'localization_true_source_mass':true_mass,'valid_candidates':int(ex['valid_candidate_counts'][i].item())})
    def summ(rs):
      return {'n':len(rs),'answer_exact':sum(r['answer_correct'] for r in rs),'answer_accuracy':sum(r['answer_correct'] for r in rs)/len(rs),'target_gt_distractor_rate':sum(r['margin']>0 for r in rs)/len(rs),'mean_margin':statistics.mean(r['margin'] for r in rs),'median_margin':statistics.median(r['margin'] for r in rs),'mean_target_prob':statistics.mean(r['target_prob'] for r in rs),'mean_candidate_mass':statistics.mean(r['candidate_mass'] for r in rs),'localization_top_source_hit_rate':statistics.mean(r['localization_source_top_hit'] for r in rs),'mean_localization_mass_on_true_sources':statistics.mean(r['localization_true_source_mass'] for r in rs)}
    byq=defaultdict(list); bys=defaultdict(list); byo=defaultdict(list); byl=defaultdict(list)
    for r in rows: byq[r['quartet_id']].append(r); bys[r['query_slot']].append(r); byo[r['orientation']].append(r); byl[r['layout_combo']].append(r)
    quart=sum(len(g)==4 and all(r['answer_correct'] for r in g) for g in byq.values())
    reversal_pairs=changed=followed=0
    for g in byq.values():
      bm={r['member']:r for r in g}
      for key in ('k0','k1'):
        a1=bm.get('o1_'+key); a2=bm.get('o2_'+key)
        if a1 is None or a2 is None: continue
        reversal_pairs+=1; changed += int(a1['predicted'] != a2['predicted']); followed += int(a1['answer_correct'] and a2['answer_correct'])
    result={'status':'TREATMENT13_FINAL_RETENTION_AUDIT' if args.pool=='retention' else 'TREATMENT13_TRAINING_AUDIT','pool':args.pool,'checkpoint_sha256':sha256_file(FINAL_CHECKPOINT_PATH),'pool_sha256':sha256_file(pool_path),'documents':len(rows),'quartets':len(byq),'overall':summ(rows),'complete_quartet':{'success_count':quart,'success_rate':quart/len(byq)},'per_query_slot':{str(k):summ(v) for k,v in sorted(bys.items())},'per_orientation':{str(k):summ(v) for k,v in sorted(byo.items())},'per_layout_combo':{k:summ(v) for k,v in sorted(byl.items())},'reversal_following':{'pairs':reversal_pairs,'prediction_changed':changed,'prediction_changed_rate':changed/reversal_pairs if reversal_pairs else None,'both_correct':followed,'both_correct_rate':followed/reversal_pairs if reversal_pairs else None},'per_document_rows':rows,'device':str(dev),'device_name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu','no_grad':True,'optimizer_steps':0}
    FINAL_AUDIT_RESULT_PATH.write_text(json.dumps(result,indent=1),encoding='utf-8'); print(json.dumps({k:v for k,v in result.items() if k!='per_document_rows'},indent=1))
if __name__=='__main__':
  main()
