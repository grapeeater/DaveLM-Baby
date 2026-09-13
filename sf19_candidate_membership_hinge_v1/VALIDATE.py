"""Bounded SF19 preflight. No checkpoint load, optimizer, inference, or update."""
import ast,hashlib,json,platform
from collections import Counter
from pathlib import Path
import torch,tokenizers
from tokenizers import Tokenizer
import CONTROLLER as C
import SF2_ENGINE as E
H=Path(__file__).resolve().parent;SF14=H.parent/'sf14_first_answer_token_ce_ablation_v1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 checks={};p=C.verify(sealed=False);old=json.loads((SF14/'PROTOCOL.json').read_text());unchanged=['D3_SELECTION.json','DEV_ORDER.json','DEV_SURFACE.json','EXTERNAL_INPUTS.json','KL_POOL_MANIFEST.json','KL_POOL.json','SCHEDULE.json','SF13_KL_SCHEDULE.json','SF2_ENGINE.py','SF2_PROTOCOL.json','TRAIN.json','TRAIN16_RETENTION.json']
 assert all(sha(H/n)==sha(SF14/n)for n in unchanged);checks['byte_identical_sf14_payload']=unchanged
 for r in p['runs']:assert sha(r['parent_checkpoint'])==r['parent_checkpoint_sha256']
 checks['parents']={str(r['seed']):r['parent_checkpoint_sha256']for r in p['runs']};assert platform.python_version()==p['runtime']['python']and torch.__version__==p['runtime']['torch']and tokenizers.__version__==p['runtime']['tokenizers']and torch.cuda.is_available();checks['runtime']={'python':platform.python_version(),'torch':torch.__version__,'tokenizers':tokenizers.__version__,'gpu':torch.cuda.get_device_name(0)}
 tok=Tokenizer.from_file(p['tokenizer']);assert sha(p['tokenizer'])==p['tokenizer_sha256'];checks['tokenizer_sha256']=sha(p['tokenizer'])
 schedule,items,idx,train16,ds,do,pool,kl,_=C.load_inputs();assert Counter(u['kind']for u in schedule)=={'english':180,'binding':20};b=next(u for u in schedule if u['kind']=='english');x,y=E.pad_batch([idx[i]for i in b['ids']],b['pad']);ym=C.mask_first_answer_ce_labels(y,b['ids'],idx);assert int((y!=-100).sum())==180 and int((ym!=-100).sum())==144
 logits=torch.zeros((36,b['pad'],1024));loss,meta=C.candidate_membership_hinge(logits,b['ids'],idx);assert float(loss)==1.0 and meta['membership_active_fraction']==1.0 and meta['membership_mean_gap']==0.0
 for k,rid in enumerate(b['ids']):
  r=idx[rid];pos=len(r['prompt_token_ids']);ct=r['candidate_token_ids'][r['correct_index']][0];logits[k,pos,ct]=2.0
 loss2,meta2=C.candidate_membership_hinge(logits,b['ids'],idx);assert float(loss2)==0.0 and meta2['membership_active_fraction']==0.0 and abs(meta2['membership_mean_gap']-2.0)<1e-7
 pair=C.margin_hinge_term(torch.zeros_like(logits),b['ids'],idx);assert float(pair)==1.0;assert len(E.pick_kl_entries(kl['entries'],1))==160
 checks['objective_isolation']={'first_answer_CE_positions':0,'later_CE_positions':144,'membership_terms':36,'membership_mock_active_loss':1.0,'membership_mock_satisfied_loss':0.0,'lambda_membership':.25,'M_membership':1.0,'pairwise_margin_active':True,'pairwise_lambda':.25,'pairwise_M':1.0,'broad_KL_positions':160}
 assert p['gates']==old['gates']and p['optimizer']==old['optimizer']and p['sampling']==old['sampling']and p['margin']==old['margin'];src=(H/'CONTROLLER.py').read_text();tree=ast.parse(src);assert src.count('candidate_membership_hinge(logits, u[\'ids\'], idx)')==1 and src.count('mask_first_answer_ce_labels(y, u[\'ids\'], idx)')==1 and 'R_name'not in src and 'autograd.grad'not in src
 assert len([n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='measure_d3'])==1;checks['unchanged_settings']={'gates':True,'optimizer':True,'scope':True,'schedule':True,'binding':True,'D3_training_calls':0}
 ou=json.loads((SF14/'U0_PREFLIGHT_RECORD.json').read_text());mapping={'87050':'87035','87051':'87036','87052':'87037'};branches={}
 for n,o in mapping.items():q=dict(ou['branches'][o]);q['sf19_seed']=int(n);q['source_sf14_seed']=int(o);branches[n]=q
 (H/'U0_PREFLIGHT_RECORD.json').write_text(json.dumps({'study':p['study'],'record_type':'INHERITED_ACTUAL_U0_PARENT_REPRODUCTION','source_path':str(SF14/'U0_PREFLIGHT_RECORD.json'),'source_sha256':sha(SF14/'U0_PREFLIGHT_RECORD.json'),'basis':'Identical parents/data/evaluator; sealed branches rerun U0 before optimizer construction.','branches':branches},indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
 checks.update({'reporter_smoke':json.loads((H/'REPORTER_SMOKE.json').read_text())['status'],'checkpoint_loaded':False,'optimizer_created':False,'updates':0,'final_accessed':False,'sacred_accessed':False});out={'status':'SF19_PROSPECTIVE_PREFLIGHT_PASS','checks':checks,'warnings':['The new relative membership hinge uses the established lambda=.25 and M=1.0 without a dose sweep. First-answer CE remains zero.','U0 is inherited from actual identical-parent evaluation and rerun by every branch.']};(H/'PREFLIGHT.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n');(H/'PREFLIGHT.md').write_text('# SF19 prospective preflight\n\n**SF19_PROSPECTIVE_PREFLIGHT_PASS**\n\nOnly a bounded candidate-membership hinge was added to sealed SF14. Mock active/satisfied cases, zero first-token CE, later CE144, pairwise margin, broad KL160, parents, tokenizer/runtime, data, schedule, scope, binding, gates, reporter, and locks passed. No checkpoint loaded, no optimizer, zero updates.\n',encoding='utf-8',newline='\n');print(out['status'])
if __name__=='__main__':main()
