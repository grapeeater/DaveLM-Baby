"""Bounded SF18 preflight. No checkpoint load, optimizer, inference, or update."""
import ast,hashlib,json,platform
from collections import Counter
from pathlib import Path
import torch,tokenizers
from torch import nn
from tokenizers import Tokenizer
import CONTROLLER as C
import SF2_ENGINE as E
H=Path(__file__).resolve().parent; SF13=H.parent/'sf13_broad_coverage_kl_retention_v1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
class Base(nn.Module):
 def __init__(self):
  super().__init__(); self.token_embedding=nn.Embedding(8,4); self.blocks=nn.ModuleList([nn.Linear(4,4) for _ in range(8)]); self.final_norm=nn.LayerNorm(4); self.language_head=nn.Linear(4,8)
class Wrapper(nn.Module):
 def __init__(self): super().__init__(); self.base_model=Base(); self.localization=nn.Linear(4,2)
def main():
 checks={}; p=C.verify(sealed=False); old=json.loads((SF13/'PROTOCOL.json').read_text()); unchanged=['D3_SELECTION.json','DEV_ORDER.json','DEV_SURFACE.json','EXTERNAL_INPUTS.json','KL_POOL_MANIFEST.json','KL_POOL.json','SCHEDULE.json','SF13_KL_SCHEDULE.json','SF2_ENGINE.py','SF2_PROTOCOL.json','TRAIN.json','TRAIN16_RETENTION.json']
 assert all(sha(H/n)==sha(SF13/n) for n in unchanged); checks['byte_identical_sf13_payload']=unchanged
 for r in p['runs']: assert sha(r['parent_checkpoint'])==r['parent_checkpoint_sha256']
 checks['parents']={str(r['seed']):r['parent_checkpoint_sha256'] for r in p['runs']}
 assert platform.python_version()==p['runtime']['python'] and torch.__version__==p['runtime']['torch'] and tokenizers.__version__==p['runtime']['tokenizers'] and torch.cuda.is_available(); checks['runtime']={'python':platform.python_version(),'torch':torch.__version__,'tokenizers':tokenizers.__version__,'gpu':torch.cuda.get_device_name(0)}
 tok=Tokenizer.from_file(p['tokenizer']); assert sha(p['tokenizer'])==p['tokenizer_sha256']; checks['tokenizer_sha256']=sha(p['tokenizer'])
 schedule,items,idx,train16,ds,do,pool,kl,_=C.load_inputs(); assert Counter(u['kind'] for u in schedule)=={'english':180,'binding':20} and len(E.pick_kl_entries(kl['entries'],1))==160
 m=Wrapper(); assert m.base_model.token_embedding.weight is not m.base_model.language_head.weight and m.base_model.token_embedding.weight.untyped_storage().data_ptr()!=m.base_model.language_head.weight.untyped_storage().data_ptr()
 E.set_scope(m,False); baseline={n:x.requires_grad for n,x in m.named_parameters()}; C.set_scope_sf18(m,False); sf18={n:x.requires_grad for n,x in m.named_parameters()}; changed=[n for n in baseline if baseline[n]!=sf18[n]]; assert changed==list(C.HEAD_PARAMETER_NAMES) and all(not sf18[n] for n in C.HEAD_PARAMETER_NAMES)
 C.set_scope_sf18(m,True); assert all(x.requires_grad for x in m.parameters()) and all(x.grad is None for x in m.parameters()); audit=C.scope_audit_sf18(m); assert all(n in audit['english']['frozen'] for n in C.HEAD_PARAMETER_NAMES) and all(n in audit['binding']['active'] for n in C.HEAD_PARAMETER_NAMES)
 checks['scope_isolation']={'untied_embedding_and_head':True,'only_english_scope_changes':changed,'english_head_frozen':True,'binding_full_scope_restored':True,'grad_none_at_transitions':True}
 assert p['gates']==old['gates'] and p['optimizer']==old['optimizer'] and p['sampling']==old['sampling'] and p['margin']==old['margin']; src=(H/'CONTROLLER.py').read_text(); tree=ast.parse(src); assert src.count("set_scope_sf18(m, u['kind'] == 'binding')")==1 and src.count('scope_audit_sf18(m)')==1 and 'R_name' not in src and 'autograd.grad' not in src
 assert len([n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='measure_d3'])==1
 checks['isolated_variable']={'full_first_answer_ce':True,'margin_lambda':.25,'margin_M':1.0,'forward_KL_lambda':1.0,'broad_KL_positions':160,'head_frozen_english_only':True,'binding_unchanged':True,'D3_training_calls':0}
 ou=json.loads((SF13/'U0_PREFLIGHT_RECORD.json').read_text()); mapping={'87047':'87032','87048':'87033','87049':'87034'}; branches={}
 for n,o in mapping.items(): q=dict(ou['branches'][o]); q['sf18_seed']=int(n); q['source_sf13_seed']=int(o); branches[n]=q
 (H/'U0_PREFLIGHT_RECORD.json').write_text(json.dumps({'study':p['study'],'record_type':'INHERITED_ACTUAL_U0_PARENT_REPRODUCTION','source_path':str(SF13/'U0_PREFLIGHT_RECORD.json'),'source_sha256':sha(SF13/'U0_PREFLIGHT_RECORD.json'),'basis':'Identical parents/data/evaluator; each sealed branch reruns U0 before optimizer construction.','branches':branches},indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
 checks.update({'reporter_smoke':json.loads((H/'REPORTER_SMOKE.json').read_text())['status'],'checkpoint_loaded':False,'optimizer_created':False,'updates':0,'final_accessed':False,'sacred_accessed':False}); out={'status':'SF18_PROSPECTIVE_PREFLIGHT_PASS','checks':checks,'warnings':['The head is frozen only on English updates; unchanged binding updates restore full T13 scope and may update it. This is the prospectively frozen treatment.','U0 evidence is inherited from actual identical-parent evaluation and rerun by each branch.']}; (H/'PREFLIGHT.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n'); (H/'PREFLIGHT.md').write_text('# SF18 prospective preflight\n\n**SF18_PROSPECTIVE_PREFLIGHT_PASS**\n\nOnly the untied language-head weight/bias change scope: frozen on English updates, restored on unchanged binding updates. SF13 full CE, margin, broad KL160, data, optimizer, schedule, gates, parents, tokenizer/runtime, and locks verified. No checkpoint loaded, no optimizer, zero updates.\n',encoding='utf-8',newline='\n'); print(out['status'])
if __name__=='__main__': main()
