"""Mechanical SF14 preflight. No checkpoint load, optimizer, inference, or update."""
import ast,hashlib,json,platform,sys
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
 assert platform.python_version()==p['runtime']['python'] and torch.__version__==p['runtime']['torch'] and tokenizers.__version__==p['runtime']['tokenizers'] and torch.cuda.is_available()
 checks['runtime']={'python':platform.python_version(),'torch':torch.__version__,'tokenizers':tokenizers.__version__,'device':torch.cuda.get_device_name(0)}
 tok=Tokenizer.from_file(p['tokenizer']); assert sha(p['tokenizer'])==p['tokenizer_sha256']
 ids={n:tok.encode(' '+n+'.').ids[0] for n in ['Alex','Owen','Mia','Nora']}; assert ids=={'Alex':314,'Owen':536,'Mia':925,'Nora':512}
 checks['tokenizer']={'sha256':sha(p['tokenizer']),'name_first_token_ids':ids}
 schedule,items,idx,train16,ds,do,pool,kl,_=C.load_inputs(); assert Counter(u['kind'] for u in schedule)=={'english':180,'binding':20}
 batch=next(u for u in schedule if u['kind']=='english'); x,y=E.pad_batch([idx[i] for i in batch['ids']],batch['pad']); assert int((y!=-100).sum())==36*5
 ym=C.mask_first_answer_ce_labels(y,batch['ids'],idx); assert int((ym!=-100).sum())==36*4
 for k,rid in enumerate(batch['ids']):
  r=idx[rid]; pos=len(r['prompt_token_ids']); c=r['candidate_token_ids'][r['correct_index']]
  assert len(c)==4 and int(y[k,pos])==c[0] and int(ym[k,pos])==-100
  assert ym[k,pos+1:pos+4].tolist()==c[1:] and int(ym[k,pos+4])==3
 logits=torch.zeros((36,batch['pad'],1024)); margin_before=C.margin_hinge_term(logits,batch['ids'],idx); margin_after=C.margin_hinge_term(logits,batch['ids'],idx)
 assert torch.equal(margin_before,margin_after) and float(margin_after)==1.0
 sel=E.pick_kl_entries(kl['entries'],1); assert len(sel)==160 and len({ri for ri,_ in sel})==160
 checks['isolated_variable']={'original_ce_labels':180,'sf14_ce_labels':144,'masked_first_name_labels':36,'remaining_candidate_plus_eos_labels_per_record':4,'margin_same_first_position':True,'margin_mock_value':1.0,'kl_positions':160,'kl_distinct_source_rows':160}
 assert p['gates']==old['gates'] and p['optimizer']==old['optimizer'] and p['sampling']==old['sampling'] and p['margin']==old['margin']
 src=(H/'CONTROLLER.py').read_text(); assert 'R_name' not in src and src.count('mask_first_answer_ce_labels(y, u[\'ids\'], idx)')==1
 tree=ast.parse(src); calls=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='measure_d3']; assert len(calls)==1
 checks['unchanged_settings']={'gates':True,'optimizer':True,'schedule':True,'margin':True,'kl':True,'binding':True,'D3_training_calls':0}
 u0old=json.loads((SF13/'U0_PREFLIGHT_RECORD.json').read_text()); mapping={'87035':'87032','87036':'87033','87037':'87034'}; branches={}
 for new,prior in mapping.items():
  q=dict(u0old['branches'][prior]); q['sf14_seed']=int(new); q['source_sf13_preflight_seed']=int(prior); branches[new]=q
 u0={'study':'SF14_FIRST_ANSWER_TOKEN_CE_ABLATION_V1','record_type':'INHERITED_ACTUAL_U0_PARENT_REPRODUCTION','source_path':str(SF13/'U0_PREFLIGHT_RECORD.json'),'source_sha256':sha(SF13/'U0_PREFLIGHT_RECORD.json'),'basis':'Same three parent bytes and byte-identical evaluator/data; the post-seal controller independently reruns and asserts U0 before optimizer creation.','branches':branches}
 (H/'U0_PREFLIGHT_RECORD.json').write_text(json.dumps(u0,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
 checks.update({'reporter_smoke':json.loads((H/'REPORTER_SMOKE.json').read_text())['status'],'checkpoint_loaded':False,'optimizer_created':False,'updates':0,'final_accessed':False,'sacred_accessed':False})
 result={'status':'SF14_PROSPECTIVE_PREFLIGHT_PASS','checks':checks,'warnings':['Masking all first-answer labels leaves mean CE over four active labels per record; this is the prospectively frozen ablation semantics.','U0 evidence is inherited from actual SF13 evaluation of the identical parents/data/evaluator and is asserted again by each sealed run before optimizer construction.']}
 (H/'PREFLIGHT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n'); print(result['status'])
if __name__=='__main__': main()
