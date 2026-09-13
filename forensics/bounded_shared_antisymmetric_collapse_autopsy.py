import json, hashlib, statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(r'C:\DaveLM-CADAVER')
CH=ROOT/'treatment13_distinct_localization_supervision_seed8380'/'FINAL_FROZEN_RETENTION_RESULTS.json'
BD=ROOT/'treatment13_bounded_shared_antisymmetric_seed8380'/'FINAL_FROZEN_RETENTION_RESULTS.json'
OUT=ROOT/'forensics'/'BOUNDED_SHARED_ANTISYMMETRIC_COLLAPSE_AUTOPSY.json'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p): return json.loads(p.read_text(encoding='utf-8'))
def cat(r): return r['localization_category']
def selected_info(r):
    # In BOTH_DISTINCT, each localization slot maps unambiguously to one saved true source.
    t0,t1=r['true0'],r['true1']; p0,p1=r['pos0'],r['pos1']
    selected=0 if r['row_weights'][0] >= r['row_weights'][1] else 1
    selpos=p0 if selected==0 else p1
    selected_true = t0 if selpos==t0 else (t1 if selpos==t1 else None)
    qsrc=t0 if r['query_slot']==0 else t1
    return selected,selected_true,selected_true is not None,selected_true==qsrc

def main():
    ch=load(CH); bd=load(BD); cr={r['doc_id']:r for r in ch['per_document_rows']}; br={r['doc_id']:r for r in bd['per_document_rows']}
    assert set(cr)==set(br) and len(br)==320
    transitions=Counter(); trans_answers=defaultdict(Counter); quartet_trans=defaultdict(list)
    for did in sorted(br):
        a,b=cr[did],br[did]; key=f'{cat(a)} -> {cat(b)}'; transitions[key]+=1; trans_answers[key][f"{int(a['answer_correct'])}->{int(b['answer_correct'])}"]+=1; quartet_trans[b['quartet_id']].append((a,b))
    def qids(pred): return sorted({b['quartet_id'] for a,b in zip(cr.values(),br.values()) if pred(a,b)})
    collapse=[(cr[d],br[d]) for d in br if cat(br[d])=='SLOT_COLLAPSE']
    repaired=[(cr[d],br[d]) for d in br if cat(cr[d])=='EXACTLY_ONE' and cat(br[d])=='BOTH_DISTINCT']
    damaged=[(cr[d],br[d]) for d in br if cat(cr[d])=='BOTH_DISTINCT' and cat(br[d]) in ('EXACTLY_ONE','SLOT_COLLAPSE')]
    def distr(rows, field): return Counter(str(r[field]) for r in rows)
    crows=[b for a,b in collapse]
    collapse_info=[]
    for a,b in collapse:
        pos=b['pos0']; t0,t1=b['true0'],b['true1']; qsrc=t0 if int(b['query_slot'])==0 else t1; collapse_info.append({'doc_id':b['doc_id'],'quartet_id':b['quartet_id'],'layout':b['layout_combo'],'member':b['member'],'query_slot':b['query_slot'],'orientation':b['orientation'],'true0':t0,'true1':t1,'champion_pos0':a['pos0'],'champion_pos1':a['pos1'],'bounded_pos0':b['pos0'],'bounded_pos1':b['pos1'],'collapsed_true_source':'true0' if pos==t0 else ('true1' if pos==t1 else 'other'),'query_relevant':pos==qsrc,'earlier_source':'true0' if t0<t1 else 'true1','answer_correct':b['answer_correct'],'row_weights':b.get('row_weights')})
    cat_counts=Counter(cat(r) for r in br.values())
    bd_rows=[r for r in br.values() if cat(r)=='BOTH_DISTINCT']; bd_errors=[r for r in bd_rows if not r['answer_correct']]
    bd_error_info=[]
    for r in bd_errors:
        selected,selected_true,selok,qok=selected_info(r); bd_error_info.append({'doc_id':r['doc_id'],'quartet_id':r['quartet_id'],'layout':r['layout_combo'],'query_slot':r['query_slot'],'orientation':r['orientation'],'pos0':r['pos0'],'pos1':r['pos1'],'true0':r['true0'],'true1':r['true1'],'row_weights':r['row_weights'],'selected_slot':selected,'selected_true_source':selected_true,'selected_row_correct':selok,'query_row_correct':qok,'answer_correct':r['answer_correct'],'predicted':r['predicted'],'target':r['target'],'distractor':r['distractor']})
    # Quartet transition categories.
    quartet_classes=Counter(); quartet_details={}
    for qid, pairs in quartet_trans.items():
        old=[cat(a) for a,b in pairs]; new=[cat(b) for a,b in pairs]
        if all(x=='BOTH_DISTINCT' for x in old) and all(x=='BOTH_DISTINCT' for x in new): cls='stable_success'
        elif any(x=='EXACTLY_ONE' for x in old) and all(x=='BOTH_DISTINCT' for x in new): cls='repaired'
        elif all(x=='BOTH_DISTINCT' for x in old) and any(x in ('EXACTLY_ONE','SLOT_COLLAPSE') for x in new): cls='damaged'
        elif all(x=='EXACTLY_ONE' for x in old) and all(x!='BOTH_DISTINCT' for x in new): cls='persistent_failure'
        else: cls='mixed_transition'
        quartet_classes[cls]+=1; quartet_details[qid]={'classification':cls,'old_categories':old,'new_categories':new}
    # Destination counts for non-source positions among all non-BD rows.
    wrong_dest=Counter()
    for r in br.values():
        if cat(r)!='BOTH_DISTINCT':
            for p in (r['pos0'],r['pos1']):
                if p not in (r['true0'],r['true1']): wrong_dest[str(p)]+=1
    out={'status':'BOUNDED_SHARED_ANTISYMMETRIC_COLLAPSE_AUTOPSY','sources':{str(CH):sha(CH),str(BD):sha(BD)},'document_count':320,'transitions':dict(transitions),'transition_answer_correctness':{k:dict(v) for k,v in trans_answers.items()},'new_collapse_quartets':sorted({b['quartet_id'] for a,b in collapse}),'repaired_quartets':sorted({b['quartet_id'] for a,b in repaired}),'damaged_quartets':sorted({b['quartet_id'] for a,b in damaged}),'collapse_counts':{'total':len(collapse),'onto_true0':sum(x['collapsed_true_source']=='true0' for x in collapse_info),'onto_true1':sum(x['collapsed_true_source']=='true1' for x in collapse_info),'onto_other':sum(x['collapsed_true_source']=='other' for x in collapse_info),'query_relevant':sum(x['query_relevant'] for x in collapse_info),'non_query_relevant':sum(not x['query_relevant'] for x in collapse_info),'earlier':sum(x['earlier_source']=='true0' for x in collapse_info),'later':sum(x['earlier_source']=='true1' for x in collapse_info)},'collapse_by_layout':dict(distr(crows,'layout_combo')),'collapse_by_query_slot':dict(distr(crows,'query_slot')),'collapse_by_orientation':dict(distr(crows,'orientation')),'collapse_info':collapse_info,'wrong_non_source_destinations':dict(wrong_dest),'old_exactly_one_to_new':Counter(cat(br[d]) for d in br if cat(cr[d])=='EXACTLY_ONE'),'bd_answer_errors':bd_error_info,'quartet_classes':dict(quartet_classes),'quartet_details':quartet_details,'saved_scorer_decomposition_available':False,'rho_saturation_metrics_available':False,'retention_rerun':False,'model_loaded':False,'optimizer_created':False}
    OUT.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k not in ('collapse_info','bd_answer_errors','quartet_details')},indent=2))
    print('COLLAPSE_AUTOPSY_COMPLETE')
if __name__=='__main__': main()
