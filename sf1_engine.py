"""Frozen single-fact acquisition controller. No sampling during execution."""
import argparse,json,hashlib,sys,os,random,math,tempfile
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'sources'))
import hr3_block3_runtime as rt
from PINNED_MASKING import set_scope,prepare_example,pad_batch
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def verify(b):
 assert sha(b/'RECEIPT.json')==(b/'RECEIPT.sha256').read_text().split()[0]
 assert sha(b/'SHA256SUMS.txt')==read(b/'RECEIPT.json')['manifest_sha256']
 for line in (b/'SHA256SUMS.txt').read_text().splitlines():
  h,n=line.split('  ',1);assert sha(b/n)==h,n
 p=read(b/'PROTOCOL.json')
 for path,h in p['external_hashes'].items():assert sha(path)==h,path
 assert sha(p['parent'])==p['parent_sha256'];assert sha(p['tokenizer'])==p['tokenizer_sha256']
 return p
def scope_audit(m):
 out={}
 for binding in [False,True]:
  set_scope(m,binding);on=[];off=[]
  for n,x in m.named_parameters():
   expected=binding or (n.startswith('base_model.') and not any(n.startswith(f'base_model.blocks.{i}.') for i in range(4)))
   assert x.requires_grad==expected and x.grad is None
   (on if expected else off).append(n)
  out['binding' if binding else 'english']={'active':on,'frozen':off}
 return out
def preflight(b):
 p=read(b/'PROTOCOL.json');train=read(b/'TRAIN.json');schedule=read(b/'SCHEDULE.json');index={r['id']:r for r in train}
 assert len(train)==16 and len(schedule)==200
 rehearsal=read(b/'data/binding_rehearsal.json')['quartets']
 for u in schedule:
  assert u['kind']==('binding' if u['update']%10==0 else 'english')
  if u['kind']=='english':
   rs=[index[i] for i in u['ids']];assert len(rs)==32
   assert all(u['ids'].count(i)==2 for i in index)
   x,y=pad_batch(rs,u['pad']);assert x.shape==y.shape and x.shape[1]<=256
   for j,r in enumerate(rs):
    z=[2]+r['prompt_token_ids']+r['candidate_token_ids'][r['correct_index']]+[3]
    labels=y[j].tolist(); assert [(i,t) for i,t in enumerate(labels) if t!=-100]==[(len(r['prompt_token_ids'])+i,t) for i,t in enumerate(z[-5:])]
    assert x[j,:len(z)-1].tolist()==z[:-1] and all(t==-100 for t in labels[len(z)-1:])
  else:assert len(rt.binding_docs_for_batch(rehearsal,u['quartets'],u['documents']))==32
 # Alignment prediction equivalence on artificial logits; changing masked positions cannot change loss.
 r=train[0];x,y=pad_batch([r],1+len(r['prompt_token_ids'])+5);v=torch.zeros(1,x.shape[1],1024)
 for i,t in enumerate(y[0]):
  if t!=-100:v[0,i,t]=10
 loss=F.cross_entropy(v.reshape(-1,1024),y.reshape(-1),ignore_index=-100)
 v[0,y[0]==-100]=50
 assert torch.equal(loss,F.cross_entropy(v.reshape(-1,1024),y.reshape(-1),ignore_index=-100))
 # Exact CPU restart round trip through production atomic writer; no model or optimizer.
 random.seed(p['seed']);torch.manual_seed(p['seed']);state={'completed_update':17,'rng':rt.capture_rng(),'mock':torch.arange(8)}
 with tempfile.TemporaryDirectory(dir=b) as d:
  file=Path(d)/'mock.pt';rt.atomic_torch_save(state,file);back=torch.load(file,weights_only=False);assert back['completed_update']+1==18 and torch.equal(back['mock'],state['mock']);rt.restore_rng(back['rng'])
 # Mock objects run the authoritative scope controller.
 class P:
  def __init__(self):self.grad=None;self.requires_grad=True
  def requires_grad_(self,b):self.requires_grad=b
 class M:
  def named_parameters(self):return self.ps.items()
 m=M();m.ps={f'base_model.blocks.{i}.weight':P() for i in range(8)}
 m.ps.update({n:P() for n in ['base_model.token_embedding.weight','base_model.position_embedding.weight','base_model.final_norm.weight','base_model.language_head.weight','localizer.u','wq.weight','wk.weight','wv.weight','wo.weight']});scope_audit(m)
 return {'schedule':200,'english_batches':180,'binding_batches':20,'masking':'PASS','alignment':'PASS','scope_mock':'PASS','atomic_restart':'PASS','optimizer_created':False,'model_loaded':False}
@torch.no_grad()
def score(m,r,tok,device):
 prefix=[2]+r['prompt_token_ids'];scores=[];details=[];first={};ci=r['correct_index']
 for cand in r['candidate_token_ids']:
  z=m.base_model(torch.tensor([prefix+cand+[3]],device=device))[0];lp=z.log_softmax(-1);k=len(prefix)-1
  vals=[float(lp[k+j,t]) for j,t in enumerate(cand+[3])];scores.append(sum(vals[:-1]));details.append(vals)
  if not first:
   fids=[c[0] for c in r['candidate_token_ids']];first={'eos_probability':float(lp[k,3].exp()),'top1':int(z[k].argmax()),'first_candidate_mass':float(lp[k,fids].exp().sum())}
 gen=rt.greedy_ids(m,r['prompt_token_ids'],device,32)[len(prefix):];target=r['candidate_token_ids'][ci];margin=scores[ci]-scores[1-ci]
 return {'id':r['id'],'family_id':r['family_id'],'pair_id':r['pair_id'],'subgroup':r['subgroup'],'prompt':r['prompt'],'candidates':r['candidates'],'correct_index':ci,'scores':scores,'token_scores_including_eos':details,'margin':margin,'correct':margin>0,'mass':sum(math.exp(s) for s in scores),'exact':gen==target+[3],'generated_ids':gen,'generated_text':tok.decode(gen,skip_special_tokens=True),**first}
def summarize(rows):
 f=defaultdict(list);p=defaultdict(list)
 for r in rows:f[r['family_id']].append(r);p[r['pair_id']].append(r)
 assert all(len(v)==2 for v in p.values())
 return {'n':len(rows),'correct':sum(r['correct'] for r in rows),'ties':sum(r['margin']==0 for r in rows),'exact':sum(r['exact'] for r in rows),'reversals':sum(all(r['correct'] for r in v) for v in p.values()),'pairs':len(p),'families':sum(all(r['correct'] for r in v) for v in f.values()),'family_count':len(f),'mean_margin':sum(r['margin'] for r in rows)/len(rows),'min_margin':min(r['margin'] for r in rows),'mean_mass':sum(r['mass'] for r in rows)/len(rows),'immediate_eos':sum(r['generated_ids']==[3] for r in rows)}
def panel(m,b,out,label,items,tok,device):
 path=out/(label+'_RAW.jsonl');done={}
 if path.exists():
  for l in path.read_text(encoding='utf-8').splitlines():
   v=json.loads(l);assert v['id'] not in done;done[v['id']]=v
 m.eval()
 for r in items:
  if r['id'] in done:continue
  v=score(m,r,tok,device)
  with path.open('a',encoding='utf-8',newline='\n') as h:h.write(json.dumps(v,ensure_ascii=False)+'\n');h.flush();os.fsync(h.fileno())
  done[r['id']]=v
 vals=list(done.values());assert set(done)=={r['id'] for r in items};agg=summarize(vals);gg=defaultdict(list)
 for r in vals:gg[r['subgroup']].append(r)
 agg['subgroups']={n:summarize(v) for n,v in gg.items()};rt.atomic_json(agg,out/(label+'_RESULT.json'));return agg
def checks(m,b,device):
 m.eval();result={};pin=rt.pinned_binding(b)
 with torch.no_grad():
  for n in ['pilot0','pilot1']:
   qs=read(b/f'data/binding_dev_{n}.json')['quartets'];raw=pin.binding_eval(m,[d for q in qs for d in q['docs']],device);s=rt.binding_summary(raw);result[n]={'summary':s,'gate':rt.binding_gate(s),'raw':raw}
  dev=[json.loads(l) for l in (b/'data/ENGLISH_DEV.jsonl').read_text(encoding='utf-8').splitlines() if l]
  lang=rt.aligned_dev_loss(m,dev,device)
 return {'binding':result,'language':lang}
def acquire(s):return s['correct']==16 and s['exact']==16 and s['reversals']==8 and s['families']==4
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--bundle',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--mode',choices=['preload','train'],required=True);ap.add_argument('--resume',action='store_true');a=ap.parse_args();b=a.bundle
 p=verify(b);rt.configure_runtime(p['seed']);assert os.environ.get('PYTHONHASHSEED')==str(p['seed'])
 import platform,tokenizers
 assert platform.python_version()=='3.12.14' and tokenizers.__version__=='0.23.1'
 schedule=read(b/'SCHEDULE.json');train=read(b/'TRAIN.json');idx={r['id']:r for r in train};pool=read(b/'data/binding_rehearsal.json')['quartets']
 for u in schedule:
  if u['kind']=='english':assert all(i in idx for i in u['ids'])
  else:rt.binding_docs_for_batch(pool,u['quartets'],u['documents'])
 if a.mode=='preload':print('SF1_PRE_PARENT_LOAD_PASS');return
 out=a.out;out.mkdir(exist_ok=True);device=torch.device('cuda');m=rt.load_model(Path(p['parent']),device,b);scope=scope_audit(m)
 rt.atomic_json(scope,out/'PARAMETER_SCOPE.json');tok=Tokenizer.from_file(p['tokenizer']);provenance={'parent':p['parent_sha256'],'protocol':sha(b/'PROTOCOL.json'),'schedule':sha(b/'SCHEDULE.json'),'receipt':sha(b/'RECEIPT.json')}
 if (out/'PROVENANCE.json').exists():assert read(out/'PROVENANCE.json')==provenance
 else:rt.atomic_json(provenance,out/'PROVENANCE.json')
 restart=out/'restart.pt';completed=0
 if a.resume:
  assert restart.exists();state=torch.load(restart,map_location=device,weights_only=False);assert state['provenance']==provenance;completed=state['completed'];assert 0<=completed<=200;m.load_state_dict(state['model'])
 else:assert not restart.exists()
 # Parent acquisition/language/binding are scored before optimizer construction.
 if completed==0:
  baseline=panel(m,b,out,'update0_acquisition',train,tok,device)
  if not (out/'update0_checks.json').exists():rt.atomic_json(checks(m,b,device),out/'update0_checks.json')
 basecheck=read(out/'update0_checks.json');assert all(v['gate'] for v in basecheck['binding'].values())
 opt=torch.optim.AdamW(m.parameters(),lr=5e-5,betas=(.9,.999),eps=1e-8,weight_decay=.05,amsgrad=False,foreach=False,fused=False)
 if a.resume:opt.load_state_dict(state['optimizer']);rt.restore_rng(state['rng'])
 pin=rt.pinned_binding(b)
 def commit(u):rt.atomic_torch_save({'completed':u,'model':m.state_dict(),'optimizer':opt.state_dict(),'rng':rt.capture_rng(),'scope':'binding' if u%10==0 else 'english','provenance':provenance},restart)
 def checkpoint_and_eval(u):
  cp=out/f'checkpoint_{u}.pt'
  if not cp.exists():rt.atomic_torch_save({'model_state_dict':m.state_dict(),'update':u,'provenance':provenance},cp)
  cpmeta=out/f'checkpoint_{u}.sha256'
  if cpmeta.exists():assert cpmeta.read_text().split()[0]==sha(cp)
  else:cpmeta.write_text(sha(cp)+'\n')
  cq=out/f'update{u}_checks.json'
  if not cq.exists():rt.atomic_json(checks(m,b,device),cq)
  q=read(cq);panel(m,b,out,f'update{u}_acquisition',train,tok,device)
  ok=all(v['gate'] for v in q['binding'].values());lang=q['language']['loss']<=basecheck['language']['loss']+.25
  if not ok or not lang:
   rt.atomic_json({'status':'STOP_REGRESSION','update':u,'binding_pass':ok,'language_cost_pass':lang,'transfer_opened':False},out/'STATUS.json');return False
  return True
 if completed in [100,200] and not checkpoint_and_eval(completed):return
 if completed==0:commit(0)
 for u in schedule[completed:]:
  opt.zero_grad(set_to_none=True);set_scope(m,u['kind']=='binding');m.train()
  inactive={n:(x.detach().clone(),{k:v.clone() if torch.is_tensor(v) else v for k,v in opt.state.get(x,{}).items()}) for n,x in m.named_parameters() if not x.requires_grad}
  if u['kind']=='english':
   batch=[idx[i] for i in u['ids']];x,y=pad_batch(batch,u['pad']);x=x.to(device);y=y.to(device);loss=F.cross_entropy(m.base_model(x).reshape(-1,1024),y.reshape(-1),ignore_index=-100)
  else:loss,_=rt.binding_loss(m,rt.binding_docs_for_batch(pool,u['quartets'],u['documents']),device,pin)
  loss.backward();assert all(x.grad is None for x in m.parameters() if not x.requires_grad)
  norm=torch.nn.utils.clip_grad_norm_([x for x in m.parameters() if x.grad is not None],2.0);opt.step()
  for n,x in m.named_parameters():
   if n not in inactive:continue
   old,st=inactive[n];assert torch.equal(x,old);now=opt.state.get(x,{});assert now.keys()==st.keys()
   for k,v in st.items():assert torch.equal(now[k],v) if torch.is_tensor(v) else now[k]==v
  number=u['update'];commit(number)
  with (out/'TRAIN_METRICS.jsonl').open('a',encoding='utf-8') as h:h.write(json.dumps({'update':number,'kind':u['kind'],'loss':float(loss.detach()),'grad_norm':float(norm)})+'\n');h.flush();os.fsync(h.fileno())
  if number%25==0:print('committed',number,flush=True)
  if number in [100,200] and not checkpoint_and_eval(number):return
 cp=out/'checkpoint_200.pt';s=read(out/'update200_acquisition_RESULT.json')
 if not acquire(s):rt.atomic_json({'status':'SF1_ACQUISITION_FAIL','completed':200,'checkpoint_sha256':sha(cp),'transfer_opened':False},out/'STATUS.json');return
 # Final checkpoint committed/hashed and both regression gates passed before opening transfer.
 for label in ['HELDOUT','ALTERNATE','COPY','COMPETING']:
  panel(m,b,out,label.lower(),read(b/(label+'.json')),tok,device)
 assert sha(p['parent'])==p['parent_sha256'];verify(b)
 rt.atomic_json({'status':'SF1_TRAINING_FAMILY_ACQUISITION_PASS_DIAGNOSTICS_COMPLETE','completed':200,'checkpoint_sha256':sha(cp),'transfer_opened':True,'final_accessed':False},out/'STATUS.json')
if __name__=='__main__':main()
