import json,hashlib,statistics,sys
from pathlib import Path
from collections import defaultdict
import torch
ROOT=Path(r'C:\DaveLM-CADAVER'); SRC=ROOT/'treatment13_learned_mapping_row_localization_seed8380'; OUT=ROOT/'treatment13_bounded_shared_antisymmetric_seed8380'; sys.path.insert(0,str(OUT)); sys.path.insert(0,str(ROOT)); sys.path.insert(0,r'C:\DaveLM-v0.9')
from bounded_model import make_bounded_model
CK=OUT/'checkpoints/seed_8380/latest.pt'; RET=SRC/'treatment13_retention_quartet_pool.json'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def state(c): return c['model_state_dict']
def rowpos(d):
 q=int(d['query_slot']); a=int(d['query_key_clause_pos']); c=int(d['other_key_pos']); return (a,c) if q==0 else (c,a)
def cat(p0,p1,t0,t1):
 T={t0,t1}
 if p0!=p1 and {p0,p1}==T:return 'BOTH_DISTINCT'
 if p0==p1:return 'SLOT_COLLAPSE'
 if sum(p in T for p in (p0,p1))==1:return 'EXACTLY_ONE'
 return 'NEITHER'
def main():
 dev=torch.device('cuda'); pool=json.loads(RET.read_text()); docs=[d for q in pool['quartets'] for d in q['docs']]; m=make_bounded_model(state(torch.load(CK,map_location=dev,weights_only=False)),1.5919504165649414,dev); m.eval(); rows=[]
 with torch.no_grad():
  for st in range(0,len(docs),32):
   ch=docs[st:st+32]; x=torch.tensor([d['full_document_token_ids'] for d in ch],device=dev); qp=torch.tensor([d['qdp'] for d in ch],device=dev); ap=torch.tensor([d['answer_causal_position'] for d in ch],device=dev); logits,ex=m(x,qp,ap)
   for i,d in enumerate(ch):
    v=logits[i,ap[i]].float(); pr=torch.softmax(v,-1); t=int(d['target_value_token']); di=int(d['distractor_value_token']); t0,t1=rowpos(d); att=ex['localization_attention'][i]; p0=int(att[:,0].argmax())+1; p1=int(att[:,1].argmax())+1; c=cat(p0,p1,t0,t1); rows.append({'doc_id':d['doc_id'],'quartet_id':d['quartet_id'],'member':d['member'],'query_slot':int(d['query_slot']),'orientation':int(d['orientation']),'layout_combo':d['layout_combo'],'target':t,'distractor':di,'predicted':int(v.argmax()),'answer_correct':bool(v.argmax()==t),'target_probability':float(pr[t]),'distractor_probability':float(pr[di]),'margin':float(v[t]-v[di]),'pos0':p0,'pos1':p1,'true0':t0,'true1':t1,'localization_category':c,'true_source_attention_mass':float(att[[t0-1,t1-1],:].sum()),'row_weights':[float(z) for z in ex['row_weights'][i]]})
 by=defaultdict(list); bq=defaultdict(list); bl=defaultdict(list); bs=defaultdict(list); bo=defaultdict(list)
 for r in rows: by[r['localization_category']].append(r); bq[r['quartet_id']].append(r); bl[r['layout_combo']].append(r); bs[r['query_slot']].append(r); bo[r['orientation']].append(r)
 def summ(v): return {'n':len(v),'answer_exact':sum(r['answer_correct'] for r in v),'answer_accuracy':sum(r['answer_correct'] for r in v)/len(v),'both_distinct_rate':sum(r['localization_category']=='BOTH_DISTINCT' for r in v)/len(v),'mean_margin':statistics.mean(r['margin'] for r in v)}
 quartpat=defaultdict(int); qstable=0
 for v in bq.values():
  cs=[r['localization_category'] for r in v]; quartpat['4/4 BOTH_DISTINCT' if all(c=='BOTH_DISTINCT' for c in cs) else '4/4 EXACTLY_ONE' if all(c=='EXACTLY_ONE' for c in cs) else 'mixed']+=1; qstable+=len({(r['pos0'],r['pos1']) for r in v})==1
 rev=chg=both=0
 for v in bq.values():
  bm={r['member']:r for r in v}
  for k in ('k0','k1'):
   a,b=bm.get('o1_'+k),bm.get('o2_'+k)
   if a and b: rev+=1; chg+=a['predicted']!=b['predicted']; both+=a['answer_correct'] and b['answer_correct']
 out={'status':'BOUNDED_SHARED_ANTISYMMETRIC_FINAL_RETENTION_AUDIT','documents':len(rows),'quartets':len(bq),'overall':summ(rows),'by_localization_category':{k:summ(v) for k,v in by.items()},'localization_counts':{k:len(v) for k,v in by.items()},'slot0_true_source_hit_rate':sum(r['pos0'] in (r['true0'],r['true1']) for r in rows)/len(rows),'slot1_true_source_hit_rate':sum(r['pos1'] in (r['true0'],r['true1']) for r in rows)/len(rows),'mean_true_source_attention_mass':statistics.mean(r['true_source_attention_mass'] for r in rows),'per_query_slot':{str(k):summ(v) for k,v in bs.items()},'per_orientation':{str(k):summ(v) for k,v in bo.items()},'per_layout':{k:summ(v) for k,v in bl.items()},'complete_quartets':sum(all(r['answer_correct'] for r in v) for v in bq.values()),'quartet_patterns':dict(quartpat),'quartet_coordinate_identical':qstable,'reversal':{'pairs':rev,'prediction_changed':chg,'both_correct':both,'both_correct_rate':both/rev},'checkpoint_sha256':sha(CK),'retention_pool_sha256':sha(RET),'no_grad':True,'optimizer_steps':0,'per_document_rows':rows}; (OUT/'FINAL_FROZEN_RETENTION_RESULTS.json').write_text(json.dumps(out,indent=1)); print(json.dumps({k:v for k,v in out.items() if k!='per_document_rows'},indent=1))
if __name__=='__main__': main()
