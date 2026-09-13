import json,collections,re
from pathlib import Path
rs=[json.loads(x) for x in Path(r'C:\DaveLM-CADAVER\fact_supervision_87001_eval_v1\RAW_RESULTS.jsonl').read_text().splitlines()]
for r in rs:
 m=re.search(r':a(\d+)q(\d+)o(\d+)$',r['id']);r.update(assignment=int(m.group(1)),query=int(m.group(2)),fact_order=int(m.group(3)))
def a(xs):
 fam=collections.defaultdict(list)
 for r in xs:fam[r['family_id']].append(r)
 tot=rev=0
 for f in fam.values():
  mp={(r['assignment'],r['query'],r['fact_order']):r for r in f}
  for r in f:
   y=mp.get((1-r['assignment'],r['query'],r['fact_order']))
   if y:tot+=1;rev+=r['correct'] and y['correct']
 return [len(xs),sum(r['correct'] for r in xs),sum(r['greedy']['exact'] for r in xs),rev//2,tot//2]
out={}
for ck in ('Pilot1 parent','Factual','Control'):
 out[ck]={s:a([r for r in rs if r['checkpoint']==ck and r['stratum']==s]) for s in ('object','predicate')}
print(json.dumps(out,indent=2))
