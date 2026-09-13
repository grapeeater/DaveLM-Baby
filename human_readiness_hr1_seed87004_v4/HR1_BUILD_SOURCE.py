import json, hashlib, re, random
from pathlib import Path
from tokenizers import Tokenizer

ROOT=Path(r'C:\DaveLM-CADAVER')
OUT=ROOT/'human_readiness_hr1_seed87004'
PILOT=ROOT/'language_pilot_1_early_block_protection_seed8380'
TOK=Path(r'C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json')
PARENT=ROOT/'treatment13_orthogonal_shared_unbounded_seed8380/checkpoints/orthogonal_shared_unbounded/seed_8380/latest.pt'
SEED=87004

def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def stories(path):
 return [json.loads(x)['text'] for x in path.read_text(encoding='utf-8').splitlines()]

def split(text):
 return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.replace('\n',' ')) if s.strip()]

def materialize(src, out, tok, limit=96):
 rows=[]
 for si,story in enumerate(stories(src)):
  for ji,s in enumerate(split(story)):
   ids=tok.encode(s).ids
   if 1 <= len(ids) <= limit:
    rows.append({'id':f'{src.stem}:{si:05d}:{ji:03d}','text':s,'token_ids':ids,'length':len(ids)})
 out.write_text(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in rows),encoding='utf-8',newline='')
 return rows

def main():
 OUT.mkdir(exist_ok=True)
 tok=Tokenizer.from_file(str(TOK))
 tr=materialize(PILOT/'language_train.jsonl',OUT/'ENGLISH_TRAIN.jsonl',tok)
 dv=materialize(PILOT/'language_dev.jsonl',OUT/'ENGLISH_DEV.jsonl',tok)
 # Materialized deterministic English batches: 64 complete sentence records, sampled without replacement per batch.
 rng=random.Random(SEED+1); order=list(range(len(tr))); rng.shuffle(order)
 batches=[]
 for i in range(0,450,1):
  start=(i*64) % len(order); ids=[tr[order[(start+j)%len(order)]]['id'] for j in range(64)]
  batches.append({'update':i+1,'record_ids':ids,'batch_size':64,'max_tokens':max(tr[order[(start+j)%len(order)]]['length'] for j in range(64))+2})
 (OUT/'ENGLISH_SCHEDULE.json').write_text(json.dumps({'seed':SEED+1,'batches':batches},indent=2),encoding='utf-8')
 # Binding schedule is a frozen reference to Pilot1 rehearsal, with deterministic quartet permutations.
 bind=json.loads((PILOT/'binding_rehearsal.json').read_text(encoding='utf-8'))['quartets']
 assert len(bind)==80 and all(len(q['docs'])==4 for q in bind)
 brng=random.Random(SEED+3); qs=list(range(80)); placements=[]
 for _ in range(5):
  p=qs[:]; brng.shuffle(p); placements.extend(p)
 bb=[{'update':10*(i+1),'quartet_ids':placements[i*8:(i+1)*8],'documents':sum(([d['doc_id'] for d in bind[q]['docs']] for q in placements[i*8:(i+1)*8]),[])} for i in range(50)]
 (OUT/'BINDING_SCHEDULE.json').write_text(json.dumps({'seed':SEED+3,'batches':bb},indent=2),encoding='utf-8')
 protocol={'version':'HR-1','seed':SEED,'parent_path':str(PARENT),'parent_sha256':sha(PARENT),'tokenizer_path':str(TOK),'tokenizer_sha256':sha(TOK),'updates':500,'english_updates':450,'binding_updates':50,'cadence':'9 English then 1 binding, repeated 50 cycles','english_batch_size':64,'english_data':'Pilot1 TinyStories train stories split deterministically at sentence boundaries; 1-96 tokenizer tokens; materialized records','english_objective':'full-vocabulary causal CE over sentence token targets and final EOS; BOS once; no intermediate EOS supervision','binding_pool_path':str(PILOT/'binding_rehearsal.json'),'binding_pool_sha256':sha(PILOT/'binding_rehearsal.json'),'binding_batch_size':32,'binding_objective':'authoritative Pilot1 answer CE plus hard-min permutation-invariant localization; weight LAM=1.0536573711078283','english_scope':'embeddings, positional embeddings, blocks 4-7, final norm, language head; blocks 0-3 and localizer/retrieval frozen','binding_scope':'full established T13 scope','optimizer':{'type':'AdamW','lr':5e-5,'betas':[0.9,0.999],'eps':1e-8,'weight_decay':0.05,'amsgrad':False,'foreach':False,'fused':False},'clip_norm':2.0,'device':'cuda','deterministic':True,'dev_points':[0,100,500],'final_battery':'sealed; inaccessible during HR-1 development','stopping':'stop on preregistered DEV readiness or binding failure; no final-battery access'}
 (OUT/'HR1_PROTOCOL.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
 # Validate provenance and deterministic materialization without loading a checkpoint or creating an optimizer.
 assert len(tr)>0 and len(dv)>0 and len(batches)==450 and len(bb)==50
 assert all(len(x['record_ids'])==64 for x in batches)
 assert all(len(x['documents'])==32 for x in bb)
 manifest={}
 for p in sorted(OUT.iterdir()):
  if p.is_file(): manifest[p.name]={'sha256':sha(p),'bytes':p.stat().st_size}
 (OUT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 print(json.dumps({'status':'HR1_READY_FOR_EXECUTION','train_records':len(tr),'dev_records':len(dv),'english_batches':450,'binding_batches':50,'parent_sha256':protocol['parent_sha256'],'tokenizer_sha256':protocol['tokenizer_sha256']},indent=2))
if __name__=='__main__': main()
