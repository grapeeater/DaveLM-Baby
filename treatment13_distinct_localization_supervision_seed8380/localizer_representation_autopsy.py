import json,hashlib,statistics,sys
from pathlib import Path
import torch
ROOT=Path(r'C:\DaveLM-CADAVER'); OUT=ROOT/'treatment13_distinct_localization_supervision_seed8380'; SRC=ROOT/'treatment13_learned_mapping_row_localization_seed8380'; CK=OUT/'checkpoints/distinct_localization_supervision/seed_8380/latest.pt'; RET=SRC/'treatment13_retention_quartet_pool.json'; DEST=OUT/'LOCALIZER_REPRESENTATION_AUTOPSY.json'
sys.path[:0]=[str(ROOT),r'C:\DaveLM-v0.9']; from treatment13_model import Treatment13Model
def rowpos(d): return (int(d['query_key_clause_pos']),int(d['other_key_pos'])) if int(d['query_slot'])==0 else (int(d['other_key_pos']),int(d['query_key_clause_pos']))
def cat(p0,p1,s0,s1):
 T={s0,s1}
 if p0!=p1 and {p0,p1}==T:return 'BOTH_DISTINCT'
 if p0==p1:return 'SLOT_COLLAPSE'
 if sum(p in T for p in (p0,p1))==1:return 'EXACTLY_ONE'
 return 'NEITHER'
def main():
 pool=json.loads(RET.read_text()); docs=[d for q in pool['quartets'] for d in q['docs']]; m=Treatment13Model().to('cuda'); c=torch.load(CK,map_location='cuda',weights_only=False); m.load_state_dict(c['model_state_dict']); m.eval(); W=m.localizer.scorer.weight.detach().cpu(); bias=m.localizer.scorer.bias.detach().cpu(); rows=[]
 with torch.no_grad():
  for st in range(0,len(docs),32):
   ch=docs[st:st+32]; x=torch.tensor([d['full_document_token_ids'] for d in ch],device='cuda'); q=torch.tensor([d['qdp'] for d in ch],device='cuda'); a=torch.tensor([d['answer_causal_position'] for d in ch],device='cuda'); logits,e=m(x,q,a); h=m._captured.float().cpu(); att=e['localization_attention'].float().cpu();
   for i,d in enumerate(ch):
    s0,s1=rowpos(d); T=x.shape[1]; n=T-1-4; valid=[p for p in range(1,1+n) if p<int(q[i]) and p+4<T]; sc=h[i,1:1+n,:]@W.T+bias; p0=int(att[i,:,0].argmax())+1;p1=int(att[i,:,1].argmax())+1; ccat=cat(p0,p1,s0,s1); Tset={s0,s1}; miss=[]
    for slot,p in enumerate((p0,p1)):
     if p not in Tset:
      ms=s1 if s0 in (p0,p1) else s0; miss.append({'slot':slot,'missed_source':ms,'winner_nonsource':p,'missed_logit':float(sc[ms-1,slot]),'winner_logit':float(sc[p-1,slot]),'gap_source_minus_nonsource':float(sc[ms-1,slot]-sc[p-1,slot]),'missed_probability':float(att[i,ms-1,slot]),'winner_probability':float(att[i,p-1,slot]),'missed_norm':float(h[i,ms].norm()),'winner_norm':float(h[i,p].norm()),'missed_scores_both_slots':[float(sc[ms-1,0]),float(sc[ms-1,1])],'winner_scores_both_slots':[float(sc[p-1,0]),float(sc[p-1,1])],'W_slot0_dot_missed':float(h[i,ms]@W[0]),'W_slot1_dot_missed':float(h[i,ms]@W[1]),'W_slot0_dot_winner':float(h[i,p]@W[0]),'W_slot1_dot_winner':float(h[i,p]@W[1])})
    v=logits[i,a[i]].argmax().item(); rows.append({'doc_id':d['doc_id'],'quartet_id':d['quartet_id'],'layout_combo':d['layout_combo'],'query_slot':int(d['query_slot']),'orientation':int(d['orientation']),'category':ccat,'answer_correct':bool(v==int(d['target_value_token'])),'pos0':p0,'pos1':p1,'true0':s0,'true1':s1,'slot_scores_at_true0':[float(sc[s0-1,0]),float(sc[s0-1,1])],'slot_scores_at_true1':[float(sc[s1-1,0]),float(sc[s1-1,1])],'true_norms':[float(h[i,s0].norm()),float(h[i,s1].norm())],'misses':miss,'source_score_weight':W.tolist(),'source_score_bias':bias.tolist()})
 # summarize missing cases and same-layout controls
 fail=[r for r in rows if r['category']=='EXACTLY_ONE']; succ=[r for r in rows if r['category']=='BOTH_DISTINCT']
 def mean(rs,key): return statistics.mean([key(r) for r in rs]) if rs else None
 miss=[x for r in fail for x in r['misses']]
 controls=[]
 for r in fail:
  ss=[z for z in succ if z['layout_combo']==r['layout_combo']]; controls.extend(ss)
 def flat_norm(rs,idx): return [r['true_norms'][idx] for r in rs]
 result={'status':'CHAMPION_LOCALIZER_REPRESENTATION_AUTOPSY_COMPLETE','checkpoint_sha256':hashlib.sha256(CK.read_bytes()).hexdigest(),'retention_pool_sha256':hashlib.sha256(RET.read_bytes()).hexdigest(),'reproduction':{'BOTH_DISTINCT':len(succ),'EXACTLY_ONE':len(fail),'SLOT_COLLAPSE':sum(r['category']=='SLOT_COLLAPSE' for r in rows),'answer_exact':sum(r['answer_correct'] for r in rows)},'scoring_pipeline':{'input':'final post-LayerNorm hidden state at eligible position j','equation':'score[j,k]=h_j dot W_localizer[k] + bias[k]','parameters':'LearnedLocalizer.scorer Linear(320,2,bias=True)','slot_parameterization':'shared input representation with two independent output columns; no normalization beyond base final LayerNorm'},'summary':{'exactly_one_n':len(fail),'wandering_n':len(miss),'wandering_gap_mean':mean(miss,lambda x:x['gap_source_minus_nonsource']),'wandering_gap_median':statistics.median(x['gap_source_minus_nonsource'] for x in miss),'missed_logit_mean':mean(miss,lambda x:x['missed_logit']),'winner_logit_mean':mean(miss,lambda x:x['winner_logit']),'missed_prob_mean':mean(miss,lambda x:x['missed_probability']),'winner_prob_mean':mean(miss,lambda x:x['winner_probability']),'missed_norm_mean':mean(miss,lambda x:x['missed_norm']),'winner_norm_mean':mean(miss,lambda x:x['winner_norm']),'cross_slot_missed_score_mean':mean(miss,lambda x:max(x['missed_scores_both_slots'])),'cross_slot_winner_score_mean':mean(miss,lambda x:max(x['winner_scores_both_slots'])),'same_layout_control_docs':len(controls),'same_layout_control_true_norm_mean':mean(controls,lambda r:statistics.mean(r['true_norms']))},'localizer_weight':W.tolist(),'localizer_bias':bias.tolist(),'per_document':rows,'wandering_cases':miss,'model_eval':True,'no_grad':True,'optimizer_created':False,'optimizer_steps':0}
 DEST.write_text(json.dumps(result,indent=1)); print(json.dumps({k:v for k,v in result.items() if k not in ('per_document','wandering_cases','localizer_weight')},indent=1)); print('CHAMPION_LOCALIZER_REPRESENTATION_AUTOPSY_COMPLETE')
if __name__=='__main__': main()
