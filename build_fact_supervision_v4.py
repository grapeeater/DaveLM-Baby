"""Build executable v4 freeze; construction only, never imports torch or checkpoints."""
import hashlib, json, random, shutil, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(r'C:\DaveLM-CADAVER')
SRC=ROOT/'fact_supervision_87001_corrected_v3'
OUT=ROOT/'fact_supervision_87001_corrected_v4'
SEED=87002
PARENT=ROOT/'language_pilot_1_early_block_protection_seed8380/pilot_run/checkpoints/seed_8380/latest.pt'
TOK=Path(r'C:\DaveLM-v0.9/tokenizer/v0_7/davelm_tokenizer.json')
BIND=ROOT/'language_pilot_1_early_block_protection_seed8380/binding_rehearsal.json'
PILOTRUN=ROOT/'language_pilot_1_early_block_protection_seed8380/run.py'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def write(name, value):
 p=OUT/name; assert not p.exists(),p
 if isinstance(value,str): value=value.encode('utf-8')
 elif not isinstance(value,bytes): value=(json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False)+'\n').encode('utf-8')
 p.write_bytes(value)
def main():
 assert not OUT.exists()
 assert sha(SRC/'SHA256SUMS.txt')=='d78a807280bbf7ca80ef725ede117e50f8741ab00bcde357c2696a15b4c41960'
 OUT.mkdir()
 # Scientific payloads are copied byte-for-byte from the preserved v3 source.
 for n in ('FAMILIES.json','ITEMS.jsonl','TRANSFER_ITEMS.jsonl','NATURALISTIC.jsonl','LEXICON.json','MANIFEST.json','PREFLIGHT.json','PROTOCOL.md','construction_check.json','V3D_HISTORICAL_STATUS_ERRATUM.md','harness.py','independent_validator.py'):
  shutil.copyfile(SRC/n,OUT/n)
 families=json.loads((SRC/'FAMILIES.json').read_text())
 rows=[json.loads(x) for x in (SRC/'ITEMS.jsonl').read_text().splitlines()]
 fam={f['family_id']:f for f in families}
 train=[f for f in families if f['partition']=='train']
 obj=[f['family_id'] for f in train if f['stratum']=='object']; pred=[f['family_id'] for f in train if f['stratum']=='predicate']
 assert len(obj)==len(pred)==24 and len(set(obj))==len(set(pred))==24
 by=defaultdict(list)
 for r in rows:
  if r['partition']=='train': by[(r['arm'],r['family_id'])].append(r)
 for k,v in by.items(): assert len(v)==8; v.sort(key=lambda r:r['id'])
 def placements(ids,seed):
  rng=random.Random(seed); out=[]
  for _ in range(37):
   x=list(ids); rng.shuffle(x); out+=x
  x=list(ids); rng.shuffle(x); out+=x[:12]
  assert len(out)==900 and sorted(Counter(out).values())==[37]*12+[38]*12
  return out
 op=placements(obj,8700201); pp=placements(pred,8700202)
 pool=json.loads(BIND.read_text())['quartets']; qids=[q['quartet_id'] for q in pool]; qmap={q['quartet_id']:q for q in pool}
 assert len(pool)==len(qids)==80 and len(set(qids))==80 and sum(len(q['docs']) for q in pool)==320 and all(len(q['docs'])==4 for q in pool)
 rng=random.Random(8700203); qp=[]
 for _ in range(5):
  x=list(qids);rng.shuffle(x);qp+=x
 assert len(qp)==400 and set(Counter(qp).values())=={5}
 bindb=[qp[i:i+8] for i in range(0,400,8)]; assert len(bindb)==50
 schedule=[]; ei=bi=0
 for u in range(1,501):
  if u%10==0:
   qs=bindb[bi];bi+=1
   schedule.append({'global_update':u,'kind':'binding','binding_batch_index':bi,'quartet_ids':qs,'document_ids':[d['doc_id'] for q in qs for d in qmap[q]['docs']]})
  else:
   of=op[2*ei:2*ei+2]; pf=pp[2*ei:2*ei+2];ei+=1
   arms={}
   mx=0
   for arm in ('factual','control'):
    rs=[r for fid in of+pf for r in by[arm,fid]]
    ids=[r['id'] for r in rs]; assert len(ids)==32
    arms[arm]={'family_ids':{'object':of,'predicate':pf},'item_ids':ids}
    mx=max(mx,max(1+len(r['prompt_token_ids'])+len(r['candidate_token_ids'][0])+1 for r in rs))
   assert mx<=256
   schedule.append({'global_update':u,'kind':'english','english_batch_index':ei,'arms':arms,'shared_pad_to_length':mx})
 assert ei==450 and bi==50
 write('MATERIALIZED_SCHEDULE_SEED87002.json',{'seed':SEED,'object_schedule_seed':8700201,'predicate_schedule_seed':8700202,'binding_schedule_seed':8700203,'updates':schedule})
 shutil.copyfile(PILOTRUN,OUT/'PINNED_PILOT1_BINDING_IMPLEMENTATION.py')
 write('AUTHORITATIVE_SOURCES.json',{'pilot1_binding_objective':{'path':str(PILOTRUN),'sha256':sha(PILOTRUN),'symbols':['rowpos','loc_loss','LAM','answer CE binding path']},'binding_pool':{'path':str(BIND),'sha256':sha(BIND)},'parent_checkpoint':{'path':str(PARENT),'sha256':sha(PARENT)},'tokenizer':{'path':str(TOK),'sha256':sha(TOK)}})
 write('TRAINING_PROTOCOL.md','''# Executable seed-87002 protocol\n\nExactly 500 updates are read literally from MATERIALIZED_SCHEDULE_SEED87002.json: 50 cycles of nine English and one binding update. Each English batch has two complete object and two complete predicate families (32 records); both arms use the same identities and shared padding. Binding batches contain eight frozen quartets / 32 documents.\n\nEach arm is a fresh process with PYTHONHASHSEED=87002, Python/random/NumPy/torch CPU/CUDA seeds 87002, deterministic algorithms enabled, float32 only, no TF32, no DataLoader and no workers. AdamW remains one continuous optimizer per arm: lr 5e-5, betas (.9,.999), eps 1e-8, wd .05, amsgrad/foreach/fused false; no scheduler.\n\nEnglish labels supervise candidate response tokens plus final EOS only. Binding uses the pinned Pilot1 `rowpos`, `loc_loss`, `LAM`, and causal answer-CE implementation. At every scope change zero grads to None, set scope, verify inactive gradients None, backward, clip only gradient-bearing parameters at 2.0, then step.\n\nDEV is factual DEV only at 0,100,500. Primary can begin only after both committed final arm hashes; Confirmation is not touched. Restart state is atomic and carries model, optimizer, completed update, scope, all RNG states, and protocol/schedule/parent hashes.\n''')
 trainer='''# Frozen v4 trainer. This source consumes only the literal schedule; it is not executed during freeze.\nimport os,random,json,hashlib,tempfile\nfrom pathlib import Path\nimport torch\nimport torch.nn.functional as F\nfrom harness import prepare_example,pad_batch\nfrom PINNED_PILOT1_BINDING_IMPLEMENTATION import rowpos,loc_loss,LAM\ndef scope(model,kind):\n for n,p in model.named_parameters():\n  p.grad=None; p.requires_grad_(kind==\"binding\" or (n.startswith(\"base_model.\") and not any(n.startswith(f\"base_model.blocks.{i}.\") for i in range(4))))\ndef english_loss(model,rows,pad):\n x,y=pad_batch(rows,pad); z=model.base_model(x); return F.cross_entropy(z.reshape(-1,z.size(-1)),y.reshape(-1),ignore_index=-100)\ndef atomic_state(path,state):\n tmp=path.with_suffix(path.suffix+\".tmp\"); torch.save(state,tmp)\n with open(tmp,\"rb\") as f: os.fsync(f.fileno())\n os.replace(tmp,path)\ndef step(model,opt,kind,batch):\n opt.zero_grad(set_to_none=True); scope(model,kind)\n if kind==\"english\": loss=english_loss(model,batch[\"rows\"],batch[\"pad\"])\n else:\n  docs=batch[\"docs\"]; x=torch.tensor([d[\"full_document_token_ids\"] for d in docs],device=next(model.parameters()).device); q=torch.tensor([d[\"qdp\"] for d in docs],device=x.device); a=torch.tensor([d[\"answer_causal_position\"] for d in docs],device=x.device); y=torch.tensor([d[\"target_value_token\"] for d in docs],device=x.device); out,extra=model(x,q,a); loss=F.cross_entropy(out[torch.arange(len(docs),device=x.device),a],y)+LAM*loc_loss(extra[\"localization_attention\"],docs)\n loss.backward(); active=[p for p in model.parameters() if p.grad is not None]; torch.nn.utils.clip_grad_norm_(active,2.0); opt.step(); return float(loss)\n'''
 write('TRAINER.py',trainer)
 evaluator='''# Frozen v4 evaluator source; no auto-Primary/Confirmation path.\nimport torch\ndef candidate_ll(logits,prompt_len,candidate):\n lp=logits[0,prompt_len:prompt_len+len(candidate)].double().log_softmax(-1); return float(lp[torch.arange(len(candidate)),torch.tensor(candidate)].sum())\ndef correct(a,b): return a>b\n'''
 write('EVALUATOR.py',evaluator)
 mock='''import json,collections\nfrom pathlib import Path\nd=Path(__file__).parent; s=json.loads((d/\"MATERIALIZED_SCHEDULE_SEED87002.json\").read_text()); u=s[\"updates\"]; assert len(u)==500; assert [x[\"global_update\"] for x in u if x[\"kind\"]==\"binding\"]==list(range(10,501,10)); assert sum(x[\"kind\"]==\"english\" for x in u)==450; assert sum(x[\"kind\"]==\"binding\" for x in u)==50\nfor x in u:\n if x[\"kind\"]==\"english\":\n  assert x[\"shared_pad_to_length\"]<=256\n  for a in (\"factual\",\"control\"): assert len(x[\"arms\"][a][\"item_ids\"])==32\n else: assert len(x[\"quartet_ids\"])==8 and len(x[\"document_ids\"])==32\nprint(\"PASS schedule/masking-source/static persistence contract\")\n'''
 write('MOCK_TESTS.py',mock)
 write('RUNTIME.json',{'python':'3.12','torch':'2.12.0+rocm7.14.0','tokenizers':'0.23.1','determinism':'torch.use_deterministic_algorithms(True); no autocast; no TF32'})
 # payload manifest intentionally excludes receipt and its detached hash.
 payload=sorted(p for p in OUT.iterdir() if p.is_file() and p.name not in ('SHA256SUMS.txt','FREEZE_RECEIPT.json','FREEZE_RECEIPT.sha256'))
 (OUT/'SHA256SUMS.txt').write_bytes(''.join(f'{sha(p)}  {p.name}\n' for p in payload).encode())
 mh=sha(OUT/'SHA256SUMS.txt')
 receipt={'status':'PASS_EXECUTABLE_PRETRAINING_FREEZE','bundle':'fact_supervision_87001_corrected_v4','manifest_sha256':mh,'payload_count':len(payload),'non_circular_chain':'FREEZE_RECEIPT.sha256 -> FREEZE_RECEIPT.json -> SHA256SUMS.txt -> payload','checkpoint_loaded':False,'optimizer_created':False,'optimizer_update':False,'behavioral_inference':False,'sacred_access':False}
 write('FREEZE_RECEIPT.json',receipt)
 (OUT/'FREEZE_RECEIPT.sha256').write_text(sha(OUT/'FREEZE_RECEIPT.json')+'  FREEZE_RECEIPT.json\n',encoding='utf-8')
 print(json.dumps({'manifest_sha256':mh,'receipt_sha256':sha(OUT/'FREEZE_RECEIPT.json'),'payload_count':len(payload)}))
if __name__=='__main__':main()
