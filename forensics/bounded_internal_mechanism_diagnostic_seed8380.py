import json, hashlib, math, statistics
from collections import Counter, defaultdict
from pathlib import Path
import torch

ROOT=Path(r'C:\DaveLM-CADAVER')
OUT=ROOT/'forensics'/'bounded_internal_mechanism_diagnostic_seed8380'
OUT.mkdir(parents=True,exist_ok=True)
CH=ROOT/'treatment13_distinct_localization_supervision_seed8380'/'checkpoints'/'distinct_localization_supervision'/'seed_8380'/'latest.pt'
BO=ROOT/'treatment13_bounded_shared_antisymmetric_seed8380'/'checkpoints'/'seed_8380'/'latest.pt'
RET=ROOT/'treatment13_learned_mapping_row_localization_seed8380'/'treatment13_retention_quartet_pool.json'
POOL_SHA='29dc566e90df7c9a592014d1d0cf60dd5da654d570096c96d0c7287bcd5f9072'
CH_SHA='0d5bf015a303ada49e6f3ab0ff3d574ea57ddcc18505cab29c2715267841e4b4'
BO_SHA='a4ac52d1f24ad253d0a9a475891bf326e4b429f23d872b981968d797521601f2'
RHO=1.5919504166
import sys
sys.path.insert(0,str(ROOT)); sys.path.insert(0,r'C:\DaveLM-v0.9'); sys.path.insert(0,str(ROOT/'treatment13_bounded_shared_antisymmetric_seed8380'))
from treatment13_model import Treatment13Model
from bounded_model import make_bounded_model

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p): return json.loads(p.read_text(encoding='utf-8'))
def state(c): return c['model_state_dict']
def rowpos(d):
    q=int(d['query_slot']); a=int(d['query_key_clause_pos']); c=int(d['other_key_pos'])
    return (a,c) if q==0 else (c,a)
def cat(p0,p1,t0,t1):
    T={t0,t1}
    if p0!=p1 and {p0,p1}==T:return 'BOTH_DISTINCT'
    if p0==p1:return 'SLOT_COLLAPSE'
    if sum(p in T for p in (p0,p1))==1:return 'EXACTLY_ONE'
    return 'NEITHER'
def stats(vals):
    vals=[float(x) for x in vals]
    if not vals:return {'n':0}
    q=statistics.quantiles(vals,n=4,method='inclusive') if len(vals)>1 else [vals[0]]*3
    return {'n':len(vals),'mean':statistics.mean(vals),'median':statistics.median(vals),'std':statistics.pstdev(vals),'q25':q[0],'q75':q[2],'min':min(vals),'max':max(vals)}
def ranks(scores,valid):
    ix=[p for p in range(scores.shape[0]) if valid[p]]
    order=sorted(ix,key=lambda p: float(scores[p]),reverse=True)
    return {p:i+1 for i,p in enumerate(order)}
def run_model(kind,ck_path,docs,device):
    ck=torch.load(ck_path,map_location=device,weights_only=False); sd=state(ck)
    if kind=='champion':
        m=Treatment13Model().to(device); m.load_state_dict(sd,strict=True); params={'W':m.localizer.scorer.weight.detach().cpu().float(),'b':m.localizer.scorer.bias.detach().cpu().float()}
    else:
        m=make_bounded_model(sd,RHO,device); params={k:getattr(m.localizer,k).detach().cpu().float() for k in ('u','v','b_s','b_r','rho')}
    for p in m.parameters(): p.requires_grad_(False)
    m.eval(); n=193-1-4; hs=[]; norms=[]; lgs=[]; ats=[]; rms=[]; ss=[]; aa=[]; tt=[]; rr=[]; valid_all=[]; ans_h=[]; retrieved=[]; rows=[]
    with torch.inference_mode():
        for st in range(0,len(docs),32):
            ch=docs[st:st+32]; x=torch.tensor([d['full_document_token_ids'] for d in ch],dtype=torch.long,device=device); qp=torch.tensor([d['qdp'] for d in ch],dtype=torch.long,device=device); ap=torch.tensor([d['answer_causal_position'] for d in ch],dtype=torch.long,device=device)
            out,ex=m(x,qp,ap); hidden=m._captured.float(); H=hidden[:,1:1+n,:]; loc=ex['localization_scores'].float(); att=ex['localization_attention'].float(); valmask=torch.zeros((len(ch),n),dtype=torch.bool,device=device)
            for i,d in enumerate(ch): valmask[i,:int(qp[i].item()-1)]=True
            hs.append(H.cpu()); norms.append(H.norm(dim=-1).cpu()); lgs.append(loc.cpu()); ats.append(att.cpu()); valid_all.append(valmask.cpu()); ans_h.append(hidden[torch.arange(len(ch),device=device),ap].cpu()); retrieved.append(ex['retrieved'].float().cpu())
            if kind=='bounded':
                S=H.matmul(params['u'].to(device))+params['b_s'].to(device); A=H.matmul(params['v'].to(device))+params['b_r'].to(device); TT=torch.tanh(A); R=params['rho'].item()*TT
                ss.append(S.cpu()); aa.append(A.cpu()); tt.append(TT.cpu()); rr.append(R.cpu())
            for i,d in enumerate(ch):
                t0,t1=rowpos(d); p0=int(att[i,:,0].argmax().item())+1; p1=int(att[i,:,1].argmax().item())+1; v=out[i,ap[i]].float(); pr=torch.softmax(v,-1); target=int(d['target_value_token']); distractor=int(d['distractor_value_token']); valid=valmask[i].tolist(); rk0=ranks(loc[i,:,0].cpu(),valid); rk1=ranks(loc[i,:,1].cpu(),valid)
                rec={'index':st+i,'doc_id':d['doc_id'],'quartet_id':d['quartet_id'],'member':d['member'],'layout':d['layout_combo'],'query_slot':int(d['query_slot']),'orientation':int(d['orientation']),'qpos':int(d['qdp']),'anspos':int(d['answer_causal_position']),'true0':t0,'true1':t1,'earlier':min(t0,t1),'later':max(t0,t1),'query_source':t0 if int(d['query_slot'])==0 else t1,'target':target,'distractor':distractor,'predicted':int(v.argmax()),'answer_correct':bool(v.argmax()==target),'target_logit':float(v[target]),'distractor_logit':float(v[distractor]),'target_probability':float(pr[target]),'distractor_probability':float(pr[distractor]),'pos0':p0,'pos1':p1,'localization_category':cat(p0,p1,t0,t1),'slot0_rank_true0':rk0.get(t0),'slot0_rank_true1':rk0.get(t1),'slot1_rank_true0':rk1.get(t0),'slot1_rank_true1':rk1.get(t1),'row_weights':[float(z) for z in ex['row_weights'][i].cpu()]}
                rows.append(rec)
    data={'kind':kind,'rows':rows,'hidden':torch.cat(hs),'norms':torch.cat(norms),'loc_logits':torch.cat(lgs),'attention':torch.cat(ats),'valid':torch.cat(valid_all),'answer_hidden':torch.cat(ans_h),'retrieved':torch.cat(retrieved),'params':params}
    if kind=='bounded': data.update({'S':torch.cat(ss),'a':torch.cat(aa),'tanh_a':torch.cat(tt),'R':torch.cat(rr)})
    return data

def main():
    assert sha(CH).lower()==CH_SHA and sha(BO).lower()==BO_SHA and sha(RET).lower()==POOL_SHA
    docs=[d for q in load(RET)['quartets'] for d in q['docs']]; assert len(docs)==320
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); assert dev.type=='cuda'
    ch=run_model('champion',CH,docs,dev); bo=run_model('bounded',BO,docs,dev)
    cr={r['doc_id']:r for r in ch['rows']}; br={r['doc_id']:r for r in bo['rows']}; assert set(cr)==set(br)
    cc=Counter(); ca=defaultdict(Counter)
    for did in br:
        k=cr[did]['localization_category']+' -> '+br[did]['localization_category']; cc[k]+=1; ca[k][f"{int(cr[did]['answer_correct'])}->{int(br[did]['answer_correct'])}"]+=1
    # Physical bounded collapse and decomposition.
    collapse=[]; all_mech=[]
    for r in bo['rows']:
        if r['localization_category']!='SLOT_COLLAPSE': continue
        i=r['index']; t0,t1=r['true0'],r['true1']; x=r['pos0']; y=t1 if x==t0 else t0; xi=x-1; yi=y-1; Sx=float(bo['S'][i,xi]); Sy=float(bo['S'][i,yi]); Rx=float(bo['R'][i,xi]); Ry=float(bo['R'][i,yi]); dS=Sx-Sy; dR=Rx-Ry; l0x=float(bo['loc_logits'][i,xi,0]); l0y=float(bo['loc_logits'][i,yi,0]); l1x=float(bo['loc_logits'][i,xi,1]); l1y=float(bo['loc_logits'][i,yi,1]); collapse.append({'doc_id':r['doc_id'],'quartet_id':r['quartet_id'],'layout':r['layout'],'x':x,'y':y,'x_order':'earlier' if x<y else 'later','query_relevant':x==r['query_source'],'champion_category':cr[r['doc_id']]['localization_category'],'champion_pos0':cr[r['doc_id']]['pos0'],'champion_pos1':cr[r['doc_id']]['pos1'],'S_x':Sx,'S_y':Sy,'Delta_S':dS,'a_x':float(bo['a'][i,xi]),'a_y':float(bo['a'][i,yi]),'tanh_x':float(bo['tanh_a'][i,xi]),'tanh_y':float(bo['tanh_a'][i,yi]),'R_x':Rx,'R_y':Ry,'Delta_R':dR,'abs_Rx_over_rho':abs(Rx)/RHO,'abs_Ry_over_rho':abs(Ry)/RHO,'Delta_slot0':l0x-l0y,'Delta_slot1':l1x-l1y,'att0_x':float(bo['attention'][i,xi,0]),'att0_y':float(bo['attention'][i,yi,0]),'att1_x':float(bo['attention'][i,xi,1]),'att1_y':float(bo['attention'][i,yi,1]),'rank0_x':r['slot0_rank_true0'] if x==t0 else r['slot0_rank_true1'],'rank0_y':r['slot0_rank_true1'] if x==t0 else r['slot0_rank_true0'],'rank1_x':r['slot1_rank_true0'] if x==t0 else r['slot1_rank_true1'],'rank1_y':r['slot1_rank_true1'] if x==t0 else r['slot1_rank_true0'],'both_slots_prefer_x':l0x>l0y and l1x>l1y,'abs_DeltaS_gt_abs_DeltaR':abs(dS)>abs(dR),'DeltaS_same_sign_x_pref':(dS>0)==(x>y),'same_sign_direct':(l0x-l0y)*(l1x-l1y)>0})
    # True-source per-document mechanism summaries.
    category_mech=defaultdict(lambda:defaultdict(list))
    for r in bo['rows']:
        i=r['index']; t0,t1=r['true0'],r['true1']; vals={'max_abs_R_over_rho':max(abs(float(bo['R'][i,t0-1]))/RHO,abs(float(bo['R'][i,t1-1]))/RHO),'abs_Delta_R':abs(float(bo['R'][i,t0-1]-bo['R'][i,t1-1])),'abs_Delta_S':abs(float(bo['S'][i,t0-1]-bo['S'][i,t1-1])),'ratio_abs_DeltaS_DeltaR':abs(float(bo['S'][i,t0-1]-bo['S'][i,t1-1]))/(abs(float(bo['R'][i,t0-1]-bo['R'][i,t1-1]))+1e-9),'true0_abs_R_over_rho':abs(float(bo['R'][i,t0-1]))/RHO,'true1_abs_R_over_rho':abs(float(bo['R'][i,t1-1]))/RHO}
        for k,v in vals.items(): category_mech[r['localization_category']][k].append(v)
    mech_summary={c:{k:stats(v) for k,v in d.items()} for c,d in category_mech.items()}
    sat_summary={c:{'max_R_near_0.90':sum(v>=.9 for v in d['max_abs_R_over_rho']),'max_R_strong_0.99':sum(v>=.99 for v in d['max_abs_R_over_rho']),'n':len(d['max_abs_R_over_rho'])} for c,d in category_mech.items()}
    # Champion EO vs bounded group summaries.
    eo_groups=defaultdict(list)
    for did in cr:
        if cr[did]['localization_category']=='EXACTLY_ONE':
            c=cr[did]; b=br[did]; i=b['index']; found=c['pos0'] if c['pos0'] in (c['true0'],c['true1']) else c['pos1']; missed=c['true1'] if found==c['true0'] else c['true0']; wi=0 if c['pos0'] not in (c['true0'],c['true1']) else 1; fi=missed-1; fpos=found-1
            ns=[p for p in range(1,188) if p+1 < c['qpos'] and p+1 not in (c['true0'],c['true1'])]; win=max(ns,key=lambda p:float(ch['loc_logits'][i,p,wi])) if ns else None
            eo_groups['EO_to_'+b['localization_category']].append({'doc_id':did,'quartet_id':c['quartet_id'],'layout':c['layout'],'found_order':'earlier' if found<missed else 'later','wandering_slot':wi,'champion_found_score':float(ch['loc_logits'][i,fpos,wi]),'champion_missed_score':float(ch['loc_logits'][i,fi,wi]),'champion_missed_opposite_score':float(ch['loc_logits'][i,fi,1-wi]),'champion_best_nonsource_score':float(ch['loc_logits'][i,win,wi]) if win is not None else None,'bounded_category':b['localization_category']})
    # Representation shifts by transition and source positions.
    rep_summary={}
    for did in cr:
        c=cr[did]; b=br[did]; i=b['index']; cs=[]; ls=[]; nd=[]
        for p in (c['true0'],c['true1']):
            hc=ch['hidden'][i,p-1]; hb=bo['hidden'][i,p-1]; cs.append(float(torch.nn.functional.cosine_similarity(hc.view(1,-1),hb.view(1,-1)).item())); ls.append(float((hc-hb).norm().item())); nd.append(abs(float(ch['norms'][i,p-1]-bo['norms'][i,p-1])))
        key=c['localization_category']+' -> '+b['localization_category']; rep_summary.setdefault(key,{'cos':[],'l2':[],'normdiff':[]}); rep_summary[key]['cos'].append(statistics.mean(cs)); rep_summary[key]['l2'].append(statistics.mean(ls)); rep_summary[key]['normdiff'].append(statistics.mean(nd))
    rep_summary={k:{m:stats(v) for m,v in d.items()} for k,d in rep_summary.items()}
    # Offline scorer-only cross-evaluation over saved hidden states.
    def offline(W,bias,Hdata):
        out=[]
        W=W.float(); bias=bias.float();
        for r in Hdata['rows']:
            i=r['index']; sc=Hdata['hidden'][i].matmul(W.T)+bias; valid=Hdata['valid'][i]; sc[~valid]=-float('inf'); a=torch.softmax(sc,dim=0); p0=int(a[:,0].argmax())+1; p1=int(a[:,1].argmax())+1; out.append(cat(p0,p1,r['true0'],r['true1']))
        return Counter(out)
    hybrid_ch_on_bo=offline(ch['params']['W'],ch['params']['b'],bo); hybrid_bo_on_ch=offline(torch.stack((bo['params']['u']+bo['params']['rho']*0,bo['params']['u']-bo['params']['rho']*0)),torch.stack((bo['params']['b_s'],bo['params']['b_s'])),ch) if False else None
    # Correct bounded scorer offline on champion hidden states.
    H=ch['hidden']; S=H.matmul(bo['params']['u'])+bo['params']['b_s']; A=H.matmul(bo['params']['v'])+bo['params']['b_r']; L=torch.stack((S+RHO*torch.tanh(A),S-RHO*torch.tanh(A)),dim=-1); hybrid_bo_on_ch=Counter()
    for r in ch['rows']:
        i=r['index']; sc=L[i].clone(); sc[~ch['valid'][i]]=float('-inf'); a=torch.softmax(sc,dim=0); hybrid_bo_on_ch[cat(int(a[:,0].argmax())+1,int(a[:,1].argmax())+1,r['true0'],r['true1'])]+=1
    result={'status':'BOUNDED_INTERNAL_MECHANISM_DIAGNOSTIC_COMPLETE','provenance':{'champion_checkpoint':sha(CH),'bounded_checkpoint':sha(BO),'retention_pool':sha(RET),'expected_rho':RHO,'device':str(dev),'model_loaded':True,'optimizer_created':False,'no_grad':True,'retention_rerun_for_behavior':False},'reproduction':{'champion':dict(Counter(r['localization_category'] for r in ch['rows'])),'bounded':dict(Counter(r['localization_category'] for r in bo['rows'])),'champion_answer_exact':sum(r['answer_correct'] for r in ch['rows']),'bounded_answer_exact':sum(r['answer_correct'] for r in bo['rows']),'bounded_collapse_order':dict(Counter(x['x_order'] for x in collapse)),'matched_transitions':dict(cc),'transition_answer_correctness':{k:dict(v) for k,v in ca.items()}},'collapse_decomposition':collapse,'mechanism_by_category':mech_summary,'saturation_counts':sat_summary,'eo_groups':{k:v for k,v in eo_groups.items()},'representation_shift':rep_summary,'offline_champion_scorer_on_bounded_hidden':dict(hybrid_ch_on_bo),'offline_bounded_scorer_on_champion_hidden':dict(hybrid_bo_on_ch),'seven_bd_answer_errors':[r for r in bo['rows'] if r['localization_category']=='BOTH_DISTINCT' and not r['answer_correct']],'all_rows':{'champion':ch['rows'],'bounded':bo['rows']}}
    torch.save({'champion_hidden':ch['hidden'],'bounded_hidden':bo['hidden'],'champion_loc_logits':ch['loc_logits'],'bounded_loc_logits':bo['loc_logits'],'champion_attention':ch['attention'],'bounded_attention':bo['attention'],'bounded_S':bo['S'],'bounded_a':bo['a'],'bounded_tanh_a':bo['tanh_a'],'bounded_R':bo['R'],'champion_norms':ch['norms'],'bounded_norms':bo['norms'],'champion_answer_hidden':ch['answer_hidden'],'bounded_answer_hidden':bo['answer_hidden'],'champion_retrieved':ch['retrieved'],'bounded_retrieved':bo['retrieved'],'metadata':result['all_rows']},OUT/'INTERNAL_MEASUREMENTS.pt')
    (OUT/'RESULTS.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'status':result['status'],'reproduction':result['reproduction'],'saturation_counts':result['saturation_counts'],'offline_champion_scorer_on_bounded_hidden':result['offline_champion_scorer_on_bounded_hidden'],'offline_bounded_scorer_on_champion_hidden':result['offline_bounded_scorer_on_champion_hidden']},indent=2))
if __name__=='__main__': main()
