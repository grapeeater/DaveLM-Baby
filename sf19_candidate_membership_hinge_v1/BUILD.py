"""Build SF19 from sealed SF14 by adding one bounded candidate-membership hinge."""
import difflib,hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parent;SF14=ROOT/'sf14_first_answer_token_ce_ablation_v1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def wj(p,o):p.write_text(json.dumps(o,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
def main():
 assert not any(p.name!='BUILD.py' for p in H.iterdir());r=json.loads((SF14/'FREEZE_RECEIPT.json').read_text());assert sha(SF14/'FREEZE_RECEIPT.json')==(SF14/'FREEZE_RECEIPT.sha256').read_text().split()[0] and sha(SF14/'SHA256SUMS.txt')==r['manifest_sha256']
 for line in(SF14/'SHA256SUMS.txt').read_text().splitlines():hh,n=line.split('  ',1);assert sha(SF14/n)==hh,n
 fs=['D3_SELECTION.json','DEV_ORDER.json','DEV_SURFACE.json','EXTERNAL_INPUTS.json','KL_POOL_MANIFEST.json','KL_POOL.json','SCHEDULE.json','SF13_KL_SCHEDULE.json','SF2_ENGINE.py','SF2_PROTOCOL.json','TRAIN.json','TRAIN16_RETENTION.json']
 for n in fs:shutil.copyfile(SF14/n,H/n)
 for d in['data','sources']:shutil.copytree(SF14/d,H/d)
 old=(SF14/'CONTROLLER.py').read_text(encoding='utf-8');new=old.replace('"""SF13: broad-coverage KL retention at fixed retention compute (sampling geometry change).\n\nONE scientific variable vs SF11/SF12: retention position coverage / sampling geometry\n(contiguous sequential 160-entry slice -> frozen row-distributed 160-entry schedule).\nAll loss terms, curriculum, optimizer, scope, gates and persistence are unchanged from SF11."""','"""SF19: sealed SF14 zero-first-token-CE recipe plus a bounded candidate-membership hinge.\n\nThe sole scientific package adds a relative first-position objective requiring one valid candidate to outrank all noncandidate vocabulary tokens. Later CE, pairwise margin, broad KL, data, optimizer, scope, schedule and gates remain SF14."""',1)
 helper=r'''

MEMBERSHIP_M = 1.0
LAMBDA_MEMBERSHIP = 0.25

def candidate_membership_hinge(logits, ids_in_batch, idx):
    """Require the best valid candidate to outrank every noncandidate token by M."""
    losses=[]; gaps=[]
    for k,rid in enumerate(ids_in_batch):
        r=idx[rid]; pos=len(r['prompt_token_ids']); row=logits[k,pos]
        ci=r['correct_index']; correct=r['candidate_token_ids'][ci][0]; wrong=r['candidate_token_ids'][1-ci][0]
        assert correct in NAME_TOKEN_IDS and wrong in NAME_TOKEN_IDS and correct!=wrong
        best_candidate=torch.maximum(row[correct],row[wrong])
        allowed=torch.ones(row.shape[0],dtype=torch.bool,device=row.device);allowed[correct]=False;allowed[wrong]=False
        best_noncandidate=row.masked_fill(~allowed,float('-inf')).max()
        gap=best_candidate-best_noncandidate;gaps.append(gap);losses.append(torch.clamp(MEMBERSHIP_M-gap,min=0.0))
    stacked=torch.stack(losses); gapstack=torch.stack(gaps)
    return stacked.mean(),{'membership_active_fraction':float((stacked.detach()>0).float().mean()),'membership_mean_gap':float(gapstack.detach().mean()),'membership_min_gap':float(gapstack.detach().min())}
'''
 marker='\ndef verify(sealed=True):\n';assert marker in new;new=new.replace(marker,helper+marker,1).replace('SF14_PROSPECTIVE_PREFLIGHT_PASS','SF19_PROSPECTIVE_PREFLIGHT_PASS').replace('SF14_PRE_PARENT_LOAD_PASS','SF19_PRE_PARENT_LOAD_PASS')
 new=new.replace("'first_answer_token_ce_weight': 0.0, 'later_response_ce_weight': 1.0}","'first_answer_token_ce_weight': 0.0, 'later_response_ce_weight': 1.0,\n                  'membership_lambda': LAMBDA_MEMBERSHIP, 'membership_M': MEMBERSHIP_M}",1)
 target="        ce = kl_val = margin_loss = None"
 assert new.count(target)==1;new=new.replace(target,"        ce = kl_val = margin_loss = membership_loss = None\n        membership_meta = None",1)
 oldloss="""            margin_loss = margin_hinge_term(logits, u['ids'], idx)
            loss = loss + LAMBDA_MARGIN * margin_loss"""
 newloss="""            margin_loss = margin_hinge_term(logits, u['ids'], idx)
            membership_loss, membership_meta = candidate_membership_hinge(logits, u['ids'], idx)
            loss = loss + LAMBDA_MARGIN * margin_loss + LAMBDA_MEMBERSHIP * membership_loss"""
 assert new.count(oldloss)==1;new=new.replace(oldloss,newloss,1)
 metric="        if margin_loss is not None: rec.update(margin_loss=float(margin_loss.detach()))"
 metric2="        if margin_loss is not None: rec.update(margin_loss=float(margin_loss.detach()), membership_loss=float(membership_loss.detach()), **membership_meta)"
 assert new.count(metric)==1;new=new.replace(metric,metric2,1)
 (H/'CONTROLLER.py').write_text(new,encoding='utf-8',newline='\n');(H/'CONTROLLER_DIFF.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='SF14/CONTROLLER.py',tofile='SF19/CONTROLLER.py')),encoding='utf-8',newline='\n')
 p=json.loads((SF14/'PROTOCOL.json').read_text());ps=[(87050,87017,'9e293af6d16cb642ba8e2bf1aeb5ad392ff4b27399ebbf0b7b8fd2ef7339b81d',0.007036717671962123),(87051,87018,'839f7f5a60b33a1736376c8a68a02985fb05ba3230e56b7aaf20d1eba5b35102',0.006672408242356376),(87052,87019,'eb6a725173ff26516d876914ea4357cce7e3758616c3d659ff059d2354404517',0.007839491795780695)]
 p.update({'study':'SF19_CANDIDATE_MEMBERSHIP_HINGE_V1','created_utc':datetime.now(timezone.utc).isoformat(),'seeds':[x[0]for x in ps],'runs':[{'seed':s,'arm':'curriculum','parent_sf8_seed':pa,'parent_checkpoint':str(ROOT/f'sf8_margin_dose_comparison_v1/runs/seed_{pa}_low/checkpoint_200.pt'),'parent_checkpoint_sha256':hh,'d3_reference':d3}for s,pa,hh,d3 in ps],
 'hypothesis':'SF14 lost exact generation because pairwise discrimination did not keep either valid answer candidate above the full vocabulary. A bounded relative candidate-membership hinge can restore top-1 answer generation without restoring first-answer-token CE and its replicated D3 pressure.',
 'manipulated_variable':'Relative to sealed SF14, add one first-position hinge: lambda_membership=0.25 times max(0,1.0-[max(logit(correct),logit(distractor))-max_noncandidate_logit]). Correct-vs-distractor margin remains independently active. First-answer CE remains zero.',
 'membership_objective':{'lambda':.25,'M_nats':1.0,'candidate_set':'the frozen correct and distractor first-name token IDs for each record','noncandidate_set':'all other 1022 vocabulary IDs, including EOS','reduction':'mean over 36 English records','first_answer_CE':0.0,'later_CE':1.0,'pairwise_margin':'unchanged lambda .25 M1','KL':'unchanged broad 160-position forward KL lambda1'},
 'classification':{'priority1':'MECHANICAL_INCOMPLETE if execution/integrity prevents classification','priority2':'RETENTION_REGRESSION if any frozen TRAIN16, language, or binding gate fails','priority3':'CANDIDATE_MEMBERSHIP_OBJECTIVE_SUPPORTED if >=2/3 reach u200 and pass every frozen endpoint gate including D3<=.010','priority4':'CANDIDATE_MEMBERSHIP_OBJECTIVE_INSUFFICIENT if >=2/3 fail D3 while retention and widening movement survive','priority5':'WIDENING_STALLED_UNDER_MEMBERSHIP if >=2/3 keep D3 and retention green through u200 but fail both widening endpoint gates','otherwise':'MIXED_OR_UNRESOLVED'},
 'interpretation':'Success supports only sufficiency of this bounded relative membership package under SF19. Failure does not rule out all objective reformulation or establish capacity limits.','next_action_rule':'Stop after SF19; no rescue or automatic next treatment.'});p.pop('first_answer_ce_ablation',None)
 wj(H/'PROTOCOL.json',p);wj(H/'PROVENANCE.json',{'study':p['study'],'predecessor':str(SF14),'predecessor_receipt_sha256':sha(SF14/'FREEZE_RECEIPT.json'),'predecessor_manifest_sha256':sha(SF14/'SHA256SUMS.txt'),'unchanged_payload_hashes':{n:sha(H/n)for n in fs},'sole_scientific_change':'bounded candidate-membership hinge at first answer position','fresh_seeds':p['seeds'],'final_sacred':'LOCKED_NOT_ACCESSED'});print('SF19_BUILD_COMPLETE')
if __name__=='__main__':main()
