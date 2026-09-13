import json,re,collections
from pathlib import Path
p=Path(r'C:\DaveLM-CADAVER\fact_supervision_87001_eval_v1\RAW_RESULTS.jsonl'); rows=[json.loads(x) for x in p.read_text().splitlines()]
for r in rows:
 m=re.search(r':a(\d+)q(\d+)o(\d+)$',r['id']); r['assignment']=int(m.group(1)); r['query']=int(m.group(2)); r['fact_order']=int(m.group(3))
def agg(rs):
 fam=collections.defaultdict(list)
 for r in rs:fam[r['family_id']].append(r)
 rev=0; tot=0; comp=0
 for fs in fam.values():
  comp+=int(all(r['correct'] for r in fs)); mp={(r['assignment'],r['query'],r['fact_order']):r for r in fs}
  for r in fs:
   k=(1-r['assignment'],r['query'],r['fact_order'])
   if k in mp: tot+=1; rev+=int(r['correct'] and mp[k]['correct'])
 return {'items':len(rs),'correct':sum(r['correct'] for r in rs),'accuracy':sum(r['correct'] for r in rs)/len(rs),'families_complete':comp,'families_total':len(fam),'reversal_pairs_counted':tot//2,'reversal_both_correct':rev//2,'reversal_rate':(rev/tot if tot else None),'mean_margin':sum(r['margin'] for r in rs)/len(rs),'greedy_exact':sum(r['greedy']['exact'] for r in rs)}
out={}
for name in ('Pilot1 parent','Factual','Control'):out[name]=agg([r for r in rows if r['checkpoint']==name])
print(json.dumps(out,indent=2)); Path(p.with_name('AGGREGATES_CORRECTED.json')).write_text(json.dumps(out,indent=2))
