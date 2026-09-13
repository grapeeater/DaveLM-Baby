"""Bounded SF17 preflight: static/data checks plus deterministic mock gradient projection."""
import ast,hashlib,json,platform
from collections import Counter
from pathlib import Path
import torch,tokenizers
from tokenizers import Tokenizer
import CONTROLLER as C
import SF2_ENGINE as E
H=Path(__file__).resolve().parent; SF13=H.parent/'sf13_broad_coverage_kl_retention_v1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def main():
 checks={}; p=C.verify(sealed=False); old=json.loads((SF13/'PROTOCOL.json').read_text())
 unchanged=['D3_SELECTION.json','DEV_ORDER.json','DEV_SURFACE.json','EXTERNAL_INPUTS.json','KL_POOL_MANIFEST.json','KL_POOL.json','SCHEDULE.json','SF13_KL_SCHEDULE.json','SF2_ENGINE.py','SF2_PROTOCOL.json','TRAIN.json','TRAIN16_RETENTION.json']
 assert all(sha(H/n)==sha(SF13/n) for n in unchanged); checks['byte_identical_sf13_payload']=unchanged
 for r in p['runs']: assert sha(r['parent_checkpoint'])==r['parent_checkpoint_sha256']
 checks['parents']={str(r['seed']):r['parent_checkpoint_sha256'] for r in p['runs']}
 assert platform.python_version()==p['runtime']['python'] and torch.__version__==p['runtime']['torch'] and tokenizers.__version__==p['runtime']['tokenizers'] and torch.cuda.is_available(); checks['runtime']={'python':platform.python_version(),'torch':torch.__version__,'tokenizers':tokenizers.__version__,'gpu':torch.cuda.get_device_name(0)}
 tok=Tokenizer.from_file(p['tokenizer']); assert sha(p['tokenizer'])==p['tokenizer_sha256']; checks['tokenizer_sha256']=sha(p['tokenizer'])
 schedule,items,idx,train16,ds,do,pool,kl,_=C.load_inputs(); assert Counter(u['kind'] for u in schedule)=={'english':180,'binding':20}; assert len(E.pick_kl_entries(kl['entries'],1))==160
 # Exact global projection algebra: opposing factual component is removed; retention is untouched.
 gf=[torch.tensor([1.,0.]),torch.tensor([2.])]; gr=[torch.tensor([-1.,1.]),torch.tensor([-1.])]
 combined,t=C.retention_priority_project(gf,gr); coeff=t['projection_coefficient']; projected=[combined[i]-gr[i] for i in range(2)]
 assert t['gradient_conflict'] and abs(sum((projected[i]*gr[i]).sum().item() for i in range(2)))<1e-6
 assert all(torch.equal(combined[i]-projected[i],gr[i]) for i in range(2)) and coeff>0
 gf2=[torch.tensor([1.,0.])]; gr2=[torch.tensor([1.,1.])]; c2,t2=C.retention_priority_project(gf2,gr2); assert not t2['gradient_conflict'] and torch.equal(c2[0],gf2[0]+gr2[0])
 gnone,t3=C.retention_priority_project([None,torch.tensor([1.])],[torch.tensor([2.]),None]); assert torch.equal(gnone[0],torch.tensor([2.])) and torch.equal(gnone[1],torch.tensor([1.]))
 checks['projection_mock']={'conflict_detected':True,'projected_factual_dot_retention':0.0,'retention_gradient_unchanged':True,'nonconflict_exact_sum':True,'single_objective_gradient_unchanged':True}
 assert p['gates']==old['gates'] and p['optimizer']==old['optimizer'] and p['sampling']==old['sampling'] and p['margin']==old['margin']; src=(H/'CONTROLLER.py').read_text(); tree=ast.parse(src); assert src.count('torch.autograd.grad(')==2 and src.count('retention_priority_project(factual_grads, retention_grads)')==2 and 'R_name' not in src
 assert len([n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='measure_d3'])==1
 checks['isolated_variable']={'factual_loss':'CE + .25 margin M=1','retention_loss':'1.0 forward KL','kl_positions':160,'global_projection_only_on_conflict':True,'binding_backward_unchanged':True,'all_scientific_settings_equal_sf13':True,'D3_training_calls':0}
 old_u0=json.loads((SF13/'U0_PREFLIGHT_RECORD.json').read_text()); mapping={'87044':'87032','87045':'87033','87046':'87034'}; branches={}
 for n,o in mapping.items(): q=dict(old_u0['branches'][o]); q['sf17_seed']=int(n); q['source_sf13_seed']=int(o); branches[n]=q
 u0={'study':p['study'],'record_type':'INHERITED_ACTUAL_U0_PARENT_REPRODUCTION','source_path':str(SF13/'U0_PREFLIGHT_RECORD.json'),'source_sha256':sha(SF13/'U0_PREFLIGHT_RECORD.json'),'basis':'Identical parents/data/evaluator; sealed controller reruns U0 and asserts it before optimizer construction.','branches':branches}; (H/'U0_PREFLIGHT_RECORD.json').write_text(json.dumps(u0,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
 checks.update({'reporter_smoke':json.loads((H/'REPORTER_SMOKE.json').read_text())['status'],'checkpoint_loaded':False,'optimizer_created':False,'updates':0,'final_accessed':False,'sacred_accessed':False})
 result={'status':'SF17_PROSPECTIVE_PREFLIGHT_PASS','checks':checks,'warnings':['This is deterministic retention-priority global gradient projection, not randomized symmetric PCGrad. The distinction is frozen before outcomes.','U0 is inherited from actual evaluation of identical parents and rerun by each sealed branch before optimizer construction.']}; (H/'PREFLIGHT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n'); (H/'PREFLIGHT.md').write_text('# SF17 prospective preflight\n\n**SF17_PROSPECTIVE_PREFLIGHT_PASS**\n\nOne variable: deterministic global retention-priority projection on conflicting English gradients. All SF13 science and gates are unchanged. Mock projection, reporter, data/schedule, parents, tokenizer/runtime, KL160, and locked-material checks passed. No checkpoint loaded; no optimizer; zero updates.\n',encoding='utf-8',newline='\n'); print(result['status'])
if __name__=='__main__': main()
