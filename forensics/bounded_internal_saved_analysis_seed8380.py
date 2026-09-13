import json, statistics
from collections import Counter
from pathlib import Path
import torch
ROOT=Path(r'C:\DaveLM-CADAVER')
DIR=ROOT/'forensics'/'bounded_internal_mechanism_diagnostic_seed8380'
RES=json.loads((DIR/'RESULTS.json').read_text())
T=torch.load(DIR/'INTERNAL_MEASUREMENTS.pt',map_location='cpu',weights_only=True)
cr={r['doc_id']:r for r in RES['all_rows']['champion']}; br={r['doc_id']:r for r in RES['all_rows']['bounded']}
cl=RES['collapse_decomposition']; bd_l=T['champion_loc_logits']; ba_h=T['bounded_answer_hidden']; brv=T['bounded_retrieved']
def stat(v):
 v=[float(x) for x in v]
 if not v:return {'n':0}
 q=statistics.quantiles(v,n=4,method='inclusive') if len(v)>1 else [v[0]]*3
 return {'n':len(v),'mean':statistics.mean(v),'median':statistics.median(v),'std':statistics.pstdev(v),'q25':q[0],'q75':q[2],'min':min(v),'max':max(v)}
def main():
 # Champion score margins on the 24 former BD cases that became collapse.
 m0=[]; m1=[]; owner=[]
 for r in cl:
  if r['champion_category']!='BOTH_DISTINCT':continue
  i=br[r['doc_id']]['index']; x,y=r['x'],r['y']; m0.append(float(bd_l[i,x-1,0]-bd_l[i,y-1,0])); m1.append(float(bd_l[i,x-1,1]-bd_l[i,y-1,1])); owner.append('slot0' if r['champion_pos0']==x else 'slot1')
 # Recompute physical/ownership cross tab.
 tab=Counter(('later' if r['x_order']=='later' else 'earlier', 'slot0' if r['champion_pos0']==r['x'] else 'slot1') for r in cl)
 # Saved retrieval/answer hidden summaries for the seven BD answer errors.
 errors=[]
 for r in RES['seven_bd_answer_errors']:
  i=r['index']; qsrc=r['true0'] if r['query_slot']==0 else r['true1']; selected=0 if r['row_weights'][0]>=r['row_weights'][1] else 1; pos=r['pos0'] if selected==0 else r['pos1']; qok=pos==qsrc
  errors.append({'doc_id':r['doc_id'],'quartet_id':r['quartet_id'],'layout':r['layout'],'query_row_correct':qok,'target_logit':r['target_logit'],'distractor_logit':r['distractor_logit'],'margin':r['target_logit']-r['distractor_logit'],'predicted':r['predicted'],'target':r['target'],'distractor':r['distractor'],'target_probability':r['target_probability'],'distractor_probability':r['distractor_probability'],'row_weights':r['row_weights'],'answer_hidden_norm':float(T['bounded_answer_hidden'][i].norm()),'retrieved_norm':float(T['bounded_retrieved'][i].norm())})
 out={'status':'BOUNDED_INTERNAL_SAVED_ANALYSIS','former_bd_to_collapse_champion_margins':{'n':len(m0),'collapse_target_minus_other_slot0':stat(m0),'collapse_target_minus_other_slot1':stat(m1),'slot0_prefers_target':sum(x>0 for x in m0),'slot1_prefers_target':sum(x>0 for x in m1),'both_scorers_prefer_target':sum(a>0 and b>0 for a,b in zip(m0,m1))},'physical_order_x_champion_owner_crosstab':{f'{a}|{b}':n for (a,b),n in tab.items()},'eo_group_summary':{},'seven_answer_errors':errors}
 for k,v in RES['eo_groups'].items():
  out['eo_group_summary'][k]={'n':len(v),'found_score':stat([x['champion_found_score'] for x in v]),'missed_score_wandering':stat([x['champion_missed_score'] for x in v]),'missed_score_opposite':stat([x['champion_missed_opposite_score'] for x in v]),'best_nonsource_wandering':stat([x['champion_best_nonsource_score'] for x in v])}
 (DIR/'ANALYSIS.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__':main()
