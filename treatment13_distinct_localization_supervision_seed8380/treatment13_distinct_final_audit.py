from __future__ import annotations
import json,hashlib,statistics
from collections import defaultdict
from pathlib import Path
import torch
ROOT=Path(r'C:\DaveLM-CADAVER'); OUT=ROOT/'treatment13_distinct_localization_supervision_seed8380'; SRC=ROOT/'treatment13_learned_mapping_row_localization_seed8380'; RET=SRC/'treatment13_retention_quartet_pool.json'; CK=OUT/'checkpoints'/'distinct_localization_supervision'/'seed_8380'/'latest.pt'; OUTJSON=OUT/'FINAL_FROZEN_RETENTION_RESULTS.json'
import sys; sys.path.insert(0,str(ROOT)); sys.path.insert(0,r'C:\DaveLM-v0.9')
from treatment13_model import Treatment13Model
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def state(c): return c['model_state_dict']
def rowpos(d):
 q=int(d['query_slot']); a=int(d['query_key_clause_pos']); c=int(d['other_key_pos']); return (a,c) if q==0 else (c,a)
def classify(p0,p1,t0,t1):
 T={t0,t1}
 if p0!=p1 and {p0,p1}==T:return 'BOTH_DISTINCT'
 if p0==p1:return 'SLOT_COLLAPSE'
 if sum(p in T for p in (p0,p1))==1:return 'EXACTLY_ONE'
 return 'NEITHER'
def main():
 dev=torch.device('cuda'); pool=json.loads(RET.read_text()); docs=[d for q in pool['quartets'] for d in q['docs']]; m=Treatment13Model().to(dev); m.load_state_dict(state(torch.load(CK,map_location=dev))); m.eval(); rows=[]
 with torch.no_grad():
  for st in range(0,len(docs),32):
   ch=docs[st:st+32]; x=torch.tensor([d['full_document_token_ids'] for d in ch],device=dev); qp=torch.tensor([d['qdp'] for d in ch],device=dev); ap=torch.tensor([d['answer_causal_position'] for d in ch],device=dev); logits,ex=m(x,qp,ap); ix=torch.arange(len(ch),device=dev)
   for i,d in enumerate(ch):
    v=logits[i,ap[i]].float(); pr=torch.softmax(v,-1); t=int(d['target_value_token']); di=int(d['distractor_value_token']); t0,t1=rowpos(d); att=ex['localization_attention'][i]; p0=int(att[:,0].argmax())+1; p1=int(att[:,1].argmax())+1; cat=classify(p0,p1,t0,t1); w=ex['row_weights'][i]; selected=int(w.argmax().item()); slot_map=None; selected_correct=None; selected_query=None
    if cat=='BOTH_DISTINCT':
      slot_map={0:t0 if p0==t0 else t1,1:t0 if p1==t0 else t1}; selected_correct=slot_map[selected] in {t0,t1}; query_src=t0 if int(d['query_slot'])==0 else t1; selected_query=slot_map[selected]==query_src
    rows.append({'doc_id':d['doc_id'],'quartet_id':d['quartet_id'],'member':d['member'],'query_slot':int(d['query_slot']),'orientation':int(d['orientation']),'layout_combo':d['layout_combo'],'target':t,'distractor':di,'predicted':int(v.argmax()),'answer_correct':bool(v.argmax()==t),'target_probability':float(pr[t]),'distractor_probability':float(pr[di]),'candidate_mass':float(pr[t]+pr[di]),'margin':float(v[t]-v[di]),'pos0':p0,'pos1':p1,'true0':t0,'true1':t1,'localization_category':cat,'true_source_attention_mass':float(att[[t0-1,t1-1],:].sum()),'row_weights':[float(z) for z in w],'selected_slot':selected,'selected_row_correct':selected_correct,'selected_query_row_correct':selected_query})
 def summ(v):
  return {'n':len(v),'answer_exact':sum(r['answer_correct'] for r in v),'answer_accuracy':sum(r['answer_correct'] for r in v)/len(v),'target_gt_distractor_rate':sum(r['margin']>0 for r in v)/len(v),'mean_margin':statistics.mean(r['margin'] for r in v),'median_margin':statistics.median(r['margin'] for r in v),'mean_candidate_mass':statistics.mean(r['candidate_mass'] for r in v),'mean_target_probability':statistics.mean(r['target_probability'] for r in v),'both_distinct_rate':sum(r['localization_category']=='BOTH_DISTINCT' for r in v)/len(v)}
 bycat=defaultdict(list); byq=defaultdict(list); bys=defaultdict(list); byo=defaultdict(list); byl=defaultdict(list); byquart=defaultdict(list)
 for r in rows: bycat[r['localization_category']].append(r); byq[r['query_slot']].append(r); bys[r['query_slot']].append(r); byo[r['orientation']].append(r); byl[r['layout_combo']].append(r); byquart[r['quartet_id']].append(r)
 quartet_patterns=defaultdict(int); coord_same=0; coord_non=[]
 for k,v in byquart.items():
  cs=[r['localization_category'] for r in v]; quartet_patterns['4/4 BOTH_DISTINCT' if all(c=='BOTH_DISTINCT' for c in cs) else '4/4 EXACTLY_ONE' if all(c=='EXACTLY_ONE' for c in cs) else 'mixed']+=1; coords={ (r['pos0'],r['pos1']) for r in v}; coord_same+=len(coords)==1
 rev=0; changed=0; bothcorrect=0
 for v in byquart.values():
  bm={r['member']:r for r in v}
  for k in ('k0','k1'):
   a,b=bm.get('o1_'+k),bm.get('o2_'+k)
   if a and b: rev+=1; changed+=a['predicted']!=b['predicted']; bothcorrect+=a['answer_correct'] and b['answer_correct']
 result={'status':'FINAL_FROZEN_RETENTION_RESULTS','documents':len(rows),'quartets':len(byquart),'overall':summ(rows),'by_localization_category':{k:summ(v) for k,v in bycat.items()},'localization_counts':{k:len(v) for k,v in bycat.items()},'slot0_true_source_hit_rate':sum(r['pos0'] in (r['true0'],r['true1']) for r in rows)/len(rows),'slot1_true_source_hit_rate':sum(r['pos1'] in (r['true0'],r['true1']) for r in rows)/len(rows),'mean_true_source_attention_mass':statistics.mean(r['true_source_attention_mass'] for r in rows),'downstream_selected_row_correct_both_distinct':sum(r['selected_row_correct'] for r in rows if r['selected_row_correct'] is not None)/sum(r['selected_row_correct'] is not None for r in rows),'downstream_selected_query_row_correct_both_distinct':sum(r['selected_query_row_correct'] for r in rows if r['selected_query_row_correct'] is not None)/sum(r['selected_query_row_correct'] is not None for r in rows),'category_x_selected_row':{k:{'n':len([r for r in v if r['selected_row_correct'] is not None]),'correct':sum(bool(r['selected_row_correct']) for r in v if r['selected_row_correct'] is not None),'answer_correct':sum(r['answer_correct'] for r in v)} for k,v in bycat.items()},'category_x_answer':{k:{'n':len(v),'correct':sum(r['answer_correct'] for r in v),'accuracy':sum(r['answer_correct'] for r in v)/len(v)} for k,v in bycat.items()},'per_query_slot':{str(k):summ(v) for k,v in sorted(bys.items())},'per_orientation':{str(k):summ(v) for k,v in sorted(byo.items())},'per_layout':{k:summ(v) for k,v in sorted(byl.items())},'complete_quartets':sum(all(r['answer_correct'] for r in v) for v in byquart.values()),'quartet_patterns':dict(quartet_patterns),'quartet_coordinate_identical':coord_same,'quartet_coordinate_nonidentical':len(byquart)-coord_same,'reversal':{'pairs':rev,'prediction_changed':changed,'both_correct':bothcorrect,'both_correct_rate':bothcorrect/rev},'checkpoint_sha256':sha(CK),'retention_pool_sha256':sha(RET),'no_grad':True,'optimizer_steps':0,'per_document_rows':rows}
 OUTJSON.write_text(json.dumps(result,indent=1)); print(json.dumps({k:v for k,v in result.items() if k!='per_document_rows'},indent=1))
if __name__=='__main__': main()
