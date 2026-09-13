import json, math, statistics, hashlib, sys
from pathlib import Path
import torch
ROOT=Path(r'C:\DaveLM-CADAVER'); OUT=ROOT/'treatment13_distinct_localization_supervision_seed8380'; SRC=ROOT/'treatment13_learned_mapping_row_localization_seed8380'
sys.path[:0]=[str(ROOT),r'C:\DaveLM-v0.9']; from treatment13_model import Treatment13Model
CK=OUT/'checkpoints/distinct_localization_supervision/seed_8380/latest.pt'; RET=SRC/'treatment13_retention_quartet_pool.json'; DEST=OUT/'LOCALIZATION_DISTRIBUTION_DIAGNOSTIC.json'
def rowpos(d): return (int(d['query_key_clause_pos']),int(d['other_key_pos'])) if int(d['query_slot'])==0 else (int(d['other_key_pos']),int(d['query_key_clause_pos']))
def cat(p0,p1,t0,t1):
 T={t0,t1}
 if p0!=p1 and {p0,p1}==T:return 'BOTH_DISTINCT'
 if p0==p1:return 'SLOT_COLLAPSE'
 if sum(x in T for x in (p0,p1))==1:return 'EXACTLY_ONE'
 return 'NEITHER'
def main():
 pool=json.loads(RET.read_text()); docs=[d for q in pool['quartets'] for d in q['docs']]; m=Treatment13Model().to('cuda'); c=torch.load(CK,map_location='cuda',weights_only=False); m.load_state_dict(c['model_state_dict']); m.eval(); rows=[]
 with torch.no_grad():
  for st in range(0,len(docs),32):
   ch=docs[st:st+32]; x=torch.tensor([d['full_document_token_ids'] for d in ch],device='cuda'); q=torch.tensor([d['qdp'] for d in ch],device='cuda'); a=torch.tensor([d['answer_causal_position'] for d in ch],device='cuda'); logits,e=m(x,q,a); ix=torch.arange(len(ch),device='cuda')
   att=e['localization_attention'].float().cpu()
   for i,d in enumerate(ch):
    A0=att[i,:,0]; A1=att[i,:,1]; s0,s1=rowpos(d); qpos=int(d['qdp']); Tlen=x.shape[1]; n=Tlen-1-4; valid=[p for p in range(1,1+n) if p<qpos and p+4<Tlen]; non=[p for p in valid if p not in (s0,s1)]; ii=[p-1 for p in non]; p0=int(A0.argmax())+1;p1=int(A1.argmax())+1; found=[]; missed=[]
    for slot,p in enumerate((p0,p1)):
     if p in (s0,s1): found.append((slot,p))
     else: missed.append((slot,p))
    true_mass=float(A0[s0-1]+A1[s0-1]+A0[s1-1]+A1[s1-1]); on=float(sum(min(float(A0[j-1]),float(A1[j-1])) for j in non)); dn=float(sum(float(A0[j-1])*float(A1[j-1]) for j in non)); os=float(A0[s0-1]*A1[s0-1]+A0[s1-1]*A1[s1-1]); overlap=float(torch.minimum(A0,A1).sum()); dot=float(torch.dot(A0,A1)); js=float(0.5*((A0*((A0.clamp_min(1e-12))/((A0+A1).clamp_min(1e-12)/2)).log()).sum()+(A1*((A1.clamp_min(1e-12))/((A0+A1).clamp_min(1e-12)/2)).log()).sum()))
    ent=[]; tops=[]; missinfo=[]
    for slot,A in enumerate((A0,A1)):
     probs=A[[p-1 for p in valid]]; ent.append(float(-(probs.clamp_min(1e-12)*probs.clamp_min(1e-12).log()).sum())); vals,inds=torch.topk(probs,min(2,len(probs))); tops.append({'p1':float(vals[0]),'p2':float(vals[1]),'margin':float(vals[0]-vals[1]),'pos1':valid[int(inds[0])],'pos2':valid[int(inds[1])]})
    for slot,wp in missed:
     ms=s1 if found and found[0][1]==s0 else s0; missinfo.append({'slot':slot,'wrong_position':wp,'missed_source':ms,'missed_source_probability':float((A0 if slot==0 else A1)[ms-1]),'max_non_source_probability':float(max((A0 if slot==0 else A1)[j-1].item() for j in non)),'source_vs_best_non_source_margin':float((A0 if slot==0 else A1)[ms-1].item()-max((A0 if slot==0 else A1)[j-1].item() for j in non)),'entropy':ent[slot],'top':tops[slot]})
    v=logits[i,a[i]].argmax().item(); rows.append({'doc_id':d['doc_id'],'layout_combo':d['layout_combo'],'localization_category':cat(p0,p1,s0,s1),'answer_correct':bool(v==int(d['target_value_token'])),'pos0':p0,'pos1':p1,'true0':s0,'true1':s1,'O_non':on,'D_non':dn,'O_src':os,'overlap_total':overlap,'dot':dot,'js_divergence':js,'slot_entropies':ent,'top_slots':tops,'miss_info':missinfo})
 cats={k:[r for r in rows if r['localization_category']==k] for k in ('BOTH_DISTINCT','EXACTLY_ONE','SLOT_COLLAPSE','NEITHER')}
 def summ(rs):
  if not rs: return {'n':0,'answer_correct':0,'O_non':None,'D_non':None,'O_src':None,'overlap_total':None,'dot':None,'js_divergence':None,'entropy_slot0':None,'entropy_slot1':None}
  def q(k):
   z=[r[k] for r in rs]; return {'mean':statistics.mean(z),'median':statistics.median(z),'min':min(z),'max':max(z)} if z else None
  return {'n':len(rs),'answer_correct':sum(r['answer_correct'] for r in rs),'O_non':q('O_non'),'D_non':q('D_non'),'O_src':q('O_src'),'overlap_total':q('overlap_total'),'dot':q('dot'),'js_divergence':q('js_divergence'),'entropy_slot0':{'mean':statistics.mean(r['slot_entropies'][0] for r in rs),'median':statistics.median(r['slot_entropies'][0] for r in rs)},'entropy_slot1':{'mean':statistics.mean(r['slot_entropies'][1] for r in rs),'median':statistics.median(r['slot_entropies'][1] for r in rs)}}
 result={'status':'CHAMPION_LOCALIZATION_DISTRIBUTION_DIAGNOSTIC_COMPLETE','checkpoint_sha256':hashlib.sha256(CK.read_bytes()).hexdigest(),'retention_pool_sha256':hashlib.sha256(RET.read_bytes()).hexdigest(),'documents':len(rows),'reproduction':{'BOTH_DISTINCT':sum(r['localization_category']=='BOTH_DISTINCT' for r in rows),'EXACTLY_ONE':sum(r['localization_category']=='EXACTLY_ONE' for r in rows),'SLOT_COLLAPSE':sum(r['localization_category']=='SLOT_COLLAPSE' for r in rows),'answer_exact':sum(r['answer_correct'] for r in rows)},'summaries':{k:summ(v) for k,v in cats.items()},'by_layout':{lay:summ([r for r in rows if r['layout_combo']==lay]) for lay in sorted(set(r['layout_combo'] for r in rows))},'exactly_one_missinfo':[x for r in cats['EXACTLY_ONE'] for x in r['miss_info']],'per_document':rows,'model_eval':True,'no_grad':True,'optimizer_created':False,'optimizer_steps':0}
 DEST.write_text(json.dumps(result,indent=1)); print(json.dumps({k:v for k,v in result.items() if k not in ('per_document','exactly_one_missinfo')},indent=1)); print('CHAMPION_LOCALIZATION_DISTRIBUTION_DIAGNOSTIC_COMPLETE')
if __name__=='__main__': main()
