"""Build prospective SF14 solely from sealed SF13 plus the first-answer CE mask."""
import difflib,hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path

H=Path(__file__).resolve().parent
ROOT=H.parent
SF13=ROOT/'sf13_broad_coverage_kl_retention_v1'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def write_json(p,o): p.write_text(json.dumps(o,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')

def main():
 assert not any(p.name!='BUILD.py' for p in H.iterdir())
 receipt=json.loads((SF13/'FREEZE_RECEIPT.json').read_text())
 assert sha(SF13/'FREEZE_RECEIPT.json')==(SF13/'FREEZE_RECEIPT.sha256').read_text().split()[0]
 assert sha(SF13/'SHA256SUMS.txt')==receipt['manifest_sha256']
 for line in (SF13/'SHA256SUMS.txt').read_text().splitlines():
  hh,n=line.split('  ',1); assert sha(SF13/n)==hh,n
 files=['D3_SELECTION.json','DEV_ORDER.json','DEV_SURFACE.json','EXTERNAL_INPUTS.json','KL_POOL_MANIFEST.json','KL_POOL.json','SCHEDULE.json','SF13_KL_SCHEDULE.json','SF2_ENGINE.py','SF2_PROTOCOL.json','TRAIN.json','TRAIN16_RETENTION.json']
 for n in files: shutil.copyfile(SF13/n,H/n)
 for d in ['data','sources']: shutil.copytree(SF13/d,H/d)

 old=(SF13/'CONTROLLER.py').read_text(encoding='utf-8')
 new=old.replace('"""SF13: broad-coverage retention-position sampling; identical SF11 loss and all other machinery."""','"""SF14: SF13 machinery with first-answer-token causal CE ablated on English updates."""')
 insert=r'''

NAME_TOKEN_IDS = (314, 536, 925, 512)


def mask_first_answer_ce_labels(labels, ids_in_batch, idx):
    """Mask only the first response/name target; preserve later response and EOS CE."""
    masked = labels.clone()
    for k, rid in enumerate(ids_in_batch):
        r = idx[rid]
        pos = len(r['prompt_token_ids'])  # label aligned to final prompt-token logit
        expected = r['candidate_token_ids'][r['correct_index']][0]
        assert expected in NAME_TOKEN_IDS
        assert int(masked[k, pos]) == expected
        masked[k, pos] = -100
    return masked
'''
 marker='\ndef verify(sealed=True):\n'; assert marker in new; new=new.replace(marker,insert+marker,1)
 new=new.replace('SF13_PROSPECTIVE_PREFLIGHT_PASS','SF14_PROSPECTIVE_PREFLIGHT_PASS')
 new=new.replace('SF13_PRE_PARENT_LOAD_PASS','SF14_PRE_PARENT_LOAD_PASS')
 target="x, y = E.pad_batch(batch, u['pad'])\n            x = x.to(device); y = y.to(device)"
 repl="x, y = E.pad_batch(batch, u['pad'])\n            y = mask_first_answer_ce_labels(y, u['ids'], idx)\n            assert int((y != -100).sum()) == len(u['ids']) * 4\n            x = x.to(device); y = y.to(device)"
 assert new.count(target)==1; new=new.replace(target,repl,1)
 new=new.replace("if ce is not None: rec.update(ce=float(ce.detach()), kl=float(kl_val.detach()), shard=u.get('shard'))","if ce is not None: rec.update(ce=float(ce.detach()), kl=float(kl_val.detach()), shard=u.get('shard'), first_answer_ce_masked=len(u['ids']), ce_supervised_tokens=len(u['ids'])*4)",1)
 new=new.replace("'kl_pool': sha(H / 'KL_POOL.json'), 'margin_M': MARGIN_M, 'lambda_margin': LAMBDA_MARGIN}","'kl_pool': sha(H / 'KL_POOL.json'), 'margin_M': MARGIN_M, 'lambda_margin': LAMBDA_MARGIN,\n                  'first_answer_token_ce_weight': 0.0, 'later_response_ce_weight': 1.0}",1)
 assert new.count('mask_first_answer_ce_labels(y, u[\'ids\'], idx)')==1
 (H/'CONTROLLER.py').write_text(new,encoding='utf-8',newline='\n')
 diff=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='SF13/CONTROLLER.py',tofile='SF14/CONTROLLER.py'))
 (H/'CONTROLLER_DIFF.patch').write_text(diff,encoding='utf-8',newline='\n')

 p=json.loads((SF13/'PROTOCOL.json').read_text())
 p.update({'study':'SF14_FIRST_ANSWER_TOKEN_CE_ABLATION_V1','created_utc':datetime.now(timezone.utc).isoformat(),'seeds':[87035,87036,87037],
  'runs':[
   {'seed':87035,'arm':'curriculum','parent_checkpoint':str(ROOT/'sf8_margin_dose_comparison_v1/runs/seed_87017_low/checkpoint_200.pt'),'parent_checkpoint_sha256':'9e293af6d16cb642ba8e2bf1aeb5ad392ff4b27399ebbf0b7b8fd2ef7339b81d','d3_reference':0.007036717671962123},
   {'seed':87036,'arm':'curriculum','parent_checkpoint':str(ROOT/'sf8_margin_dose_comparison_v1/runs/seed_87018_low/checkpoint_200.pt'),'parent_checkpoint_sha256':'839f7f5a60b33a1736376c8a68a02985fb05ba3230e56b7aaf20d1eba5b35102','d3_reference':0.006672408242356376},
   {'seed':87037,'arm':'curriculum','parent_checkpoint':str(ROOT/'sf8_margin_dose_comparison_v1/runs/seed_87019_low/checkpoint_200.pt'),'parent_checkpoint_sha256':'eb6a725173ff26516d876914ea4357cce7e3758616c3d659ff059d2354404517','d3_reference':0.007839491795780695}],
  'hypothesis':'Direct causal CE on the first answer-name token is sufficient to cause the replicated D3 name-probability leakage during widening supervision.',
  'manipulated_variable':'On every frozen English batch, set only the causally aligned first answer-name target label to ignore_index=-100 before the unchanged mean causal CE. The remaining three candidate tokens plus final EOS remain supervised. The first-token margin remains active and unchanged.',
  'english_objective':'SF13 objective unchanged except first answer-name token CE weight=0: mean causal CE over remaining response tokens plus EOS +1.0 full-vocabulary forward KL on frozen broad-coverage160 positions +.25 first-answer-token hinge with M=1.0. No R_name.',
  'first_answer_ce_ablation':{'weight':0.0,'scope':'all 36 records in widening English updates, including the four scheduled TRAIN16 preservation records','implementation':'change the aligned first response label from its frozen name token ID to ignore_index=-100, then use unchanged F.cross_entropy mean over the remaining four labels per record','later_candidate_and_eos_weight':1.0,'margin_at_first_token':'unchanged','binding_updates':'unchanged'},
  'classification':{'priority1':'MECHANICAL_INCOMPLETE if execution or integrity prevents classification','priority2':'RETENTION_REGRESSION if any frozen TRAIN16, language, or binding retention gate fails','priority3':'FIRST_TOKEN_CE_ABLATION_SUPPORTED if >=2/3 branches reach u200 and pass all frozen endpoint gates including D3<=.010','priority4':'FIRST_TOKEN_CE_ABLATION_INSUFFICIENT if >=2/3 branches reproduce D3>.010 while TRAIN16/language/binding remain preserved and both DEV_SURFACE and DEV_ORDER exact improve over own u0','otherwise':'MIXED_OR_UNRESOLVED'},
  'interpretation':'Success supports sufficiency of removing direct first-answer-name CE pressure under this setup; failure means that pressure was insufficient to explain or contain the replicated D3 regression. Neither result proves a fundamental incompatibility, necessity/sufficiency of margin, an internal mechanism, or a capacity limit.',
  'next_action_rule':'Freeze and report SF14 only. Do not propose or execute another treatment in this study.'})
 write_json(H/'PROTOCOL.json',p)
 write_json(H/'PROVENANCE.json',{'study':'SF14_FIRST_ANSWER_TOKEN_CE_ABLATION_V1','predecessor':str(SF13),'predecessor_receipt_sha256':sha(SF13/'FREEZE_RECEIPT.json'),'predecessor_manifest_sha256':sha(SF13/'SHA256SUMS.txt'),'unchanged_payload_hashes':{n:sha(H/n) for n in files},'sole_scientific_change':'first answer-name causal CE label masked on English updates','fresh_seeds':[87035,87036,87037],'final_sacred':'LOCKED_NOT_ACCESSED'})
 print('SF14_BUILD_COMPLETE')
if __name__=='__main__': main()
