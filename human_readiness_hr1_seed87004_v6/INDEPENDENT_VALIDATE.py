import json,hashlib
from pathlib import Path
R=Path(r'C:\DaveLM-CADAVER\human_readiness_hr1_seed87004')
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 p=json.loads((R/'HR1_PROTOCOL.json').read_text()); tr=[json.loads(x) for x in (R/'ENGLISH_TRAIN.jsonl').read_text(encoding='utf-8').splitlines()]; dv=[json.loads(x) for x in (R/'ENGLISH_DEV.jsonl').read_text(encoding='utf-8').splitlines()]; es=json.loads((R/'ENGLISH_SCHEDULE.json').read_text()); bs=json.loads((R/'BINDING_SCHEDULE.json').read_text());
 assert p['updates']==500 and p['english_updates']==450 and p['binding_updates']==50
 assert len(es['batches'])==450 and len(bs['batches'])==50
 tids={x['id'] for x in tr}; dids={x['id'] for x in dv}; assert tids.isdisjoint(dids) and len(tids)==len(tr) and len(dids)==len(dv)
 assert all(len(b['record_ids'])==64 and set(b['record_ids'])<=tids for b in es['batches'])
 assert all(len(b['documents'])==32 and len(b['quartet_ids'])==8 for b in bs['batches'])
 assert [b['update'] for b in es['batches']]==list(range(1,451)); assert [b['update'] for b in bs['batches']]==list(range(10,501,10))
 assert 'FINAL_ITEMS.jsonl' not in str(R) and not (R/'FINAL_ITEMS.jsonl').exists()
 out={'status':'INDEPENDENT_VALIDATION_PASS','train_records':len(tr),'dev_records':len(dv),'english_batches':450,'binding_batches':50,'updates':500,'final_battery_accessed':False,'parent_sha256':p['parent_sha256'],'tokenizer_sha256':p['tokenizer_sha256'],'hr1_builder_sha256':sha(Path(r'C:\DaveLM-CADAVER\hr1_build.py'))}
 (R/'INDEPENDENT_VALIDATION.json').write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))
main()
