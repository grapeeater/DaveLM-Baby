import json,hashlib
from pathlib import Path
from collections import Counter,defaultdict
ROOT=Path(r'C:\DaveLM-CADAVER')
CH=ROOT/'treatment13_distinct_localization_supervision_seed8380'/'FINAL_FROZEN_RETENTION_RESULTS.json'
BD=ROOT/'treatment13_bounded_shared_antisymmetric_seed8380'/'FINAL_FROZEN_RETENTION_RESULTS.json'
OUT=ROOT/'forensics'/'PHYSICAL_SOURCE_ORDER_ASYMMETRY_CENSUS.json'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p): return json.loads(p.read_text(encoding='utf-8'))
def physical_label(pos,t0,t1): return 'earlier' if pos==min(t0,t1) else 'later' if pos==max(t0,t1) else 'other'
def query_src(r): return r['true0'] if int(r['query_slot'])==0 else r['true1']
def main():
 ch=load(CH); bd=load(BD); cr={r['doc_id']:r for r in ch['per_document_rows']}; br={r['doc_id']:r for r in bd['per_document_rows']}; assert set(cr)==set(br) and len(br)==320
 # Champion EO found/missed order
 eo=[]
 for r in cr.values():
  if r['localization_category']!='EXACTLY_ONE': continue
  t0,t1=r['true0'],r['true1']; found=[p for p in (r['pos0'],r['pos1']) if p in (t0,t1)][0]; missed=t1 if found==t0 else t0
  eo.append({'doc_id':r['doc_id'],'quartet_id':r['quartet_id'],'layout':r['layout_combo'],'query_slot':r['query_slot'],'orientation':r['orientation'],'true0':t0,'true1':t1,'found':found,'missed':missed,'found_order':physical_label(found,t0,t1),'missed_order':physical_label(missed,t0,t1),'found_true_source':'true0' if found==t0 else 'true1','missed_true_source':'true0' if missed==t0 else 'true1','found_query_relevant':found==query_src(r),'missed_query_relevant':missed==query_src(r)})
 # Champion BD physical ownership
 bd_own=[]
 for r in cr.values():
  if r['localization_category']!='BOTH_DISTINCT': continue
  t0,t1=r['true0'],r['true1']; earlier=min(t0,t1); slot0='earlier' if r['pos0']==earlier else 'later'; slot1='earlier' if r['pos1']==earlier else 'later'; bd_own.append({'doc_id':r['doc_id'],'quartet_id':r['quartet_id'],'layout':r['layout_combo'],'query_slot':r['query_slot'],'orientation':r['orientation'],'slot0_order':slot0,'slot1_order':slot1,'earlier_slot':0 if slot0=='earlier' else 1,'later_slot':1 if slot0=='earlier' else 0})
 # Bounded collapse details
 collapse=[]
 for r in br.values():
  if r['localization_category']!='SLOT_COLLAPSE': continue
  t0,t1=r['true0'],r['true1']; c=r['pos0']; ch=cr[r['doc_id']]; owner=[k for k,p in enumerate((ch['pos0'],ch['pos1'])) if p==c]
  collapse.append({'doc_id':r['doc_id'],'quartet_id':r['quartet_id'],'layout':r['layout_combo'],'member':r['member'],'query_slot':r['query_slot'],'orientation':r['orientation'],'true0':t0,'true1':t1,'source_order':physical_label(c,t0,t1),'collapsed_true_source':'true0' if c==t0 else 'true1' if c==t1 else 'other','query_relevant':c==query_src(r),'champion_category':ch['localization_category'],'champion_pos0':ch['pos0'],'champion_pos1':ch['pos1'],'bounded_pos0':r['pos0'],'bounded_pos1':r['pos1'],'champion_slots_at_c':owner,'source_separation':abs(t1-t0)})
 # quartet stability
 qgroups=defaultdict(list)
 for r in br.values(): qgroups[r['quartet_id']].append(r)
 bounded_qstable={qid:{'n':len(v),'all_collapse':all(r['localization_category']=='SLOT_COLLAPSE' for r in v),'collapse_positions':sorted(set(r['pos0'] for r in v)),'collapse_categories':[r['localization_category'] for r in v]} for qid,v in qgroups.items()}
 # distributions
 def count(rows,key): return dict(Counter(str(r[key]) for r in rows))
 out={'status':'PHYSICAL_SOURCE_ORDER_ASYMMETRY_DISCRIMINATOR','sources':{str(CH):sha(CH),str(BD):sha(BD)},'document_count':320,'champion_eo':{'n':len(eo),'found_order':dict(Counter(r['found_order'] for r in eo)),'missed_order':dict(Counter(r['missed_order'] for r in eo)),'found_source':dict(Counter(r['found_true_source'] for r in eo)),'missed_source':dict(Counter(r['missed_true_source'] for r in eo)),'found_query_relevant':dict(Counter(r['found_query_relevant'] for r in eo)),'missed_query_relevant':dict(Counter(r['missed_query_relevant'] for r in eo)),'by_query_slot':count(eo,'query_slot'),'by_orientation':count(eo,'orientation'),'by_layout':count(eo,'layout'),'rows':eo},'champion_bd_ownership':{'n':len(bd_own),'earlier_slot0_later_slot1':sum(r['earlier_slot']==0 for r in bd_own),'earlier_slot1_later_slot0':sum(r['earlier_slot']==1 for r in bd_own),'rows':bd_own},'bounded_collapse':{'n':len(collapse),'source_order':dict(Counter(r['source_order'] for r in collapse)),'collapsed_true_source':dict(Counter(r['collapsed_true_source'] for r in collapse)),'query_relevant':dict(Counter(r['query_relevant'] for r in collapse)),'query_slot':dict(Counter(r['query_slot'] for r in collapse)),'orientation':dict(Counter(r['orientation'] for r in collapse)),'layout':dict(Counter(r['layout'] for r in collapse)),'separation':dict(Counter(r['source_separation'] for r in collapse)),'champion_category':dict(Counter(r['champion_category'] for r in collapse)),'champion_owner_slots_at_c':dict(Counter(str(tuple(r['champion_slots_at_c'])) for r in collapse)),'rows':collapse},'bounded_collapse_quartets':{qid:v for qid,v in bounded_qstable.items() if v['all_collapse']},'confidence_metrics_saved':False,'retention_rerun':False,'model_loaded':False}
 OUT.write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps({k:v for k,v in out.items() if k not in ('champion_eo','champion_bd_ownership','bounded_collapse','bounded_collapse_quartets')},indent=2)); print('PHYSICAL_SOURCE_ORDER_ASYMMETRY_DISCRIMINATOR_COMPLETE')
if __name__=='__main__': main()
