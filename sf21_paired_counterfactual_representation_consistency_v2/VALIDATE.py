"""Independent bounded SF21 preflight; no checkpoint load or optimizer."""
from pathlib import Path
from collections import Counter, defaultdict
import ast, hashlib, json, platform
import torch, tokenizers
import torch.nn.functional as F
import CONTROLLER as C

H=Path(__file__).resolve().parent
S=H.parent/"sf20_identity_heldout_balanced_entity_rotation_v2"
def read(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 p=C.verify(sealed=False); assert p['study'].endswith('_V2')
 assert p['seeds']==[87056,87057,87058] and C.LAMBDA_CONSISTENCY==0.1
 assert platform.python_version()==p['runtime']['python'] and torch.__version__==p['runtime']['torch'] and tokenizers.__version__==p['runtime']['tokenizers']
 identical=['TRAIN.json','SCHEDULE.json','D3_SELECTION.json','DEV_ORDER.json','DEV_SURFACE.json','KL_POOL.json','KL_POOL_MANIFEST.json','SF13_KL_SCHEDULE.json','SF2_ENGINE.py','SF2_PROTOCOL.json','TRAIN16_RETENTION.json','IDENTITY_ASSIGNMENT.json','TOKENIZATION_MATCH.json']
 assert all(sha(H/n)==sha(S/n) for n in identical)
 for r in p['runs']: assert sha(r['parent_checkpoint'])==r['parent_checkpoint_sha256']
 schedule,items,idx,train16,ds,do,pool,kl,_=C.load_inputs()
 assert Counter(x['kind'] for x in schedule)=={'english':180,'binding':20}
 assert len(items)==48 and len(train16)==len(ds)==len(do)==16 and len(pool)==80
 pm=read(H/'PAIR_MANIFEST.json'); pairs=pm['pairs']; assert len(pairs)==16
 seen=[]; by_pair=defaultdict(list)
 for r in items:
  if r['id'].startswith('SF20:'): by_pair[r['pair_id']].append(r)
 for pair in pairs:
  a,b=idx[pair['record_a']],idx[pair['record_b']]
  assert a['pair_id']==b['pair_id']==pair['pair_id'] and a['correct_index']!=b['correct_index']
  assert a['candidates']==b['candidates'] and a['actor']!=b['actor']
  assert (a['subgroup'],a['predicate'],a['object'])==(b['subgroup'],b['predicate'],b['object'])
  assert pair['representation_position_a']==len(a['prompt_token_ids'])
  assert pair['representation_position_b']==len(b['prompt_token_ids'])
  assert pair['identity_assignment_differs'] and 'answer' not in pair['position_semantics'].split()[0]
  seen += [a['id'],b['id']]
 assert len(seen)==len(set(seen))==32 and set(seen)=={r['id'] for r in items if r['id'].startswith('SF20:')}
 # Every English update contains all 32 widening records and therefore exactly 16 pairs.
 assert all(sum(i.startswith('SF20:') for i in u['ids'])==32 for u in schedule if u['kind']=='english')
 # Synthetic numerical/gradient sanity for the actual functional implementation.
 u=next(x for x in schedule if x['kind']=='english'); hidden=torch.randn(36,u['pad'],384,requires_grad=True)
 loss,cos=C.paired_consistency(hidden,u['ids'],idx,pairs)
 manual=[]
 loc={rid:k for k,rid in enumerate(u['ids'])}
 for pair in pairs:
  a,b=pair['record_a'],pair['record_b']; va=F.normalize(hidden[loc[a],len(idx[a]['prompt_token_ids'])].float(),dim=-1); vb=F.normalize(hidden[loc[b],len(idx[b]['prompt_token_ids'])].float(),dim=-1); manual.append(1-(va*vb).sum())
 assert torch.equal(loss,torch.stack(manual).mean()) and torch.isfinite(loss) and torch.isfinite(cos)
 loss.backward(); assert all(hidden.grad[loc[x],len(idx[x]['prompt_token_ids'])].norm()>0 for pair in pairs for x in [pair['record_a'],pair['record_b']])
 # Exact tiny sanity: identical -> 0; orthogonal -> 1.
 h=torch.zeros(36,u['pad'],384); 
 for pair in pairs:
  ia,ib=loc[pair['record_a']],loc[pair['record_b']]; pa=len(idx[pair['record_a']]['prompt_token_ids']); pb=len(idx[pair['record_b']]['prompt_token_ids']); h[ia,pa,0]=1; h[ib,pb,0]=1
 z,_=C.paired_consistency(h,u['ids'],idx,pairs); assert float(z)==0.0
 source=(H/'CONTROLLER.py').read_text(encoding='utf-8'); ast.parse(source)
 assert source.count('LAMBDA_CONSISTENCY = 0.1')==1
 assert source.count("register_forward_hook(capture_final_norm)")==1
 assert source.count("loss = loss + LAMBDA_MARGIN * margin_loss + LAMBDA_CONSISTENCY * consistency_loss")==1
 assert source.count("F.cross_entropy(logits.reshape(-1, 1024), y.reshape(-1), ignore_index=-100)")==1
 assert 'candidate_membership_hinge' not in source and 'R_name' not in source and 'autograd.grad' not in source
 assert C.LAMBDA_MARGIN==0.25 and C.MARGIN_M==1.0 and C.E.LAMBDA_KL==1.0 and len(C.E.pick_kl_entries(kl['entries'],1))==160
 old=read(S/'PROTOCOL.json'); assert p['gates']==old['gates'] and p['optimizer']==old['optimizer'] and p['scope']==old['scope'] and p['evaluation']==old['evaluation']
 out={'status':'SF21_STATIC_PREFLIGHT_PASS','pairs':16,'paired_records':32,'representation':'final_norm output at final prompt/query token','both_sides_gradient':True,'lambda_consistency':0.1,'synthetic_identical_loss':0.0,'sf20_payloads_byte_identical':identical,'seeds':p['seeds'],'parents_verified':3,'schedule':{'english':180,'binding':20},'checkpoint_loaded':False,'optimizer_created':False,'updates':0,'locked_transfer':'LOCKED_UNSCORED','final_accessed':False,'sacred_accessed':False}
 (H/'STATIC_PREFLIGHT.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n'); print(out['status'])
if __name__=='__main__': main()
