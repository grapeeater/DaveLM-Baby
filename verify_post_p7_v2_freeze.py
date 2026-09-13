"""Read-only independent final-byte verification, no checkpoint deserialization."""
import hashlib,json,stat
from collections import Counter
from pathlib import Path
from post_p7_v2_independent_validator import parse,validate
ROOT=Path(__file__).resolve().parent
D=ROOT/'post_p7_language_report_card_v2_seed8391'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for line in (D/'SHA256SUMS.txt').read_text().splitlines():
    digest,name=line.split('  ')
    assert sha(D/name)==digest,name
for name,meta in json.loads((D/'FREEZE_RECEIPT.json').read_text())['artifacts'].items():
    assert sha(D/name)==meta['sha256'] and (D/name).stat().st_size==meta['bytes']
for p in D.iterdir():
    assert not p.stat().st_mode & stat.S_IWRITE
    assert b'\r' not in p.read_bytes()
items=[json.loads(l) for l in (D/'ITEMS.jsonl').read_text().splitlines()]
result=validate(items)
actual_balance={}
for section in ('near_distribution','counterfactual','surface_form','distractor'):
    rows=[r for r in items if r['section']==section]
    identities=Counter();positions=Counter();actions=Counter()
    for r in rows:
        name,facts,key=parse(r['prompt']);identities[name]+=1
        positions[next(i for i,f in enumerate(facts) if f[1:]==key)]+=1
        actions[key[0]]+=1
    assert len(set(identities.values()))==len(set(positions.values()))==len(set(actions.values()))==1
    actual_balance[section]={'correct_subject':dict(identities),'relevant_fact_position':dict(positions),'queried_action':dict(actions)}
# Matched diagnostic core families must have identical underlying facts/answers.
byid={r['item_id']:r for r in items}
for r in items:
    if r['section'] not in ('surface_form','distractor'):continue
    fid=r['family_id'].split(':')[1]
    base=byid[f"near_distribution:{fid}:a{r['assignment']}q{r['query']}o{r['fact_order']}d-1"]
    assert parse(base['prompt'])==parse(r['prompt'])
assert sha(ROOT/'post_p7_language_report_card_v1_seed8380'/'SHA256SUMS.txt')=='bca8d2882059f4c029d969a88de5fe90e0973f33a96ad111770955cd910ebdcd'
audit=json.loads((D/'OVERLAP_AUDIT.json').read_text())
for s in audit['sources']:assert sha(Path(s['path']))==s['sha256']
print(json.dumps({'status':'PASS_FINAL_READ_ONLY_CHECK','independent_semantics':result,'actual_text_balance':actual_balance,'source_count':len(audit['sources']),'sources_with_atomic_overlap':sum(s['items_with_any_fact_substring_overlap_diagnostic']>0 for s in audit['sources']),'detached_sha256':sha(D/'SHA256SUMS.txt'),'items_sha256':sha(D/'ITEMS.jsonl'),'erratum_sha256':sha(ROOT/'P7_LINEAGE_AND_REPORT_CARD_ERRATUM_20260905.md')},indent=2))
