import json,hashlib
from pathlib import Path
r=Path(r'C:\\DaveLM-CADAVER\\human_readiness_hr2_seed87005_v7')
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=json.loads((r/'HR2_PROTOCOL.json').read_text()); tr=[json.loads(x) for x in (r/'ENGLISH_TRAIN.jsonl').read_text(encoding='utf8').splitlines()]; dv=[json.loads(x) for x in (r/'ENGLISH_DEV.jsonl').read_text(encoding='utf8').splitlines()]; es=json.loads((r/'ENGLISH_SCHEDULE.json').read_text())['batches']; bs=json.loads((r/'BINDING_SCHEDULE.json').read_text())['batches']; assert len(es)==450 and len(bs)==50 and [x['update'] for x in bs]==list(range(10,501,10)); assert len({x['id'] for x in tr})==len(tr) and len({x['id'] for x in dv})==len(dv) and {x['id'] for x in tr}.isdisjoint({x['id'] for x in dv}); assert all(len(x['record_ids'])==64 for x in es) and all(len(x['documents'])==32 for x in bs); assert p['parent_sha256']==h(r/'HR1_PARENT.pt'); assert p['english_schedule_sha256']==h(r/'ENGLISH_SCHEDULE.json') and p['binding_schedule_sha256']==h(r/'BINDING_SCHEDULE.json'); print(json.dumps({'status':'HR2_VALIDATION_PASS','train_records':len(tr),'dev_records':len(dv),'english_batches':450,'binding_batches':50,'final_battery_accessed':False},indent=2))
main()

