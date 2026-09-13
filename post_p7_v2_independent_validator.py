"""Independent text-only semantic validator. No builder imports or model access."""
import re
from collections import Counter

def parse(prompt):
    body, query = prompt.split('\n')
    records = []
    for sentence in body.split('. '):
        words = sentence.rstrip('.').split()
        if 'watched' in words:
            assert words == ['Mia','watched','a','bird','by','the','window']
            continue
        if words[0] == 'The':
            assert len(words)==7 and words[3]=='was' and words[5]=='by'
            color,obj,verb,agent=words[1],words[2],words[4],words[6]
        else:
            agent=words[0]
            assert words[2]=='the' and len(words)==5
            verb,color,obj=words[1],words[3],words[4]
        assert agent in ('Sam','Tom') and verb in ('found','carried')
        records.append((agent,verb,color,obj))
    assert len(records)==2 and len(set(records))==2
    assert len({r[0] for r in records})==2
    assert len({r[1] for r in records})==2
    assert len({r[2:] for r in records})==2
    q=query.split()
    if q[0]=='Who':
        assert len(q)==5 and q[2]=='the'
        key=(q[1],q[3],q[4].rstrip('?'))
    else:
        assert q[:3]==['The','person','who'] and q[4]=='the' and q[-1]=='was' and len(q)==8
        key=(q[3],q[5],q[6])
    matches=[r[0] for r in records if r[1:]==key]
    assert len(matches)==1
    return matches[0], records, key

def validate(items):
    controlled=[r for r in items if r['section']!='naturalistic']
    for r in controlled:
        answer,_,_=parse(r['prompt'])
        assert r['candidates'][r['correct_index']]==' '+answer+'.',r['item_id']
    groups={}
    for r in controlled: groups.setdefault(r['family_id'],[]).append(r)
    for fid,rs in groups.items():
        extra=2 if rs[0]['section']=='distractor' else 1
        assert len(rs)==8*extra
        for field in ('assignment','query','fact_order','correct_index'):
            assert Counter(r[field] for r in rs)=={0:4*extra,1:4*extra}
        by={(r['assignment'],r['query'],r['fact_order'],r['distractor_position']):r for r in rs}
        assert len(by)==len(rs)
        for k,r in by.items():
            a,q,o,d=k
            answer,facts,query=parse(r['prompt'])
            other=by[(a,q,1-o,d)]
            ans2,facts2,query2=parse(other['prompt'])
            assert answer==ans2 and facts==list(reversed(facts2)) and query==query2
            ans2,facts2,query2=parse(by[(1-a,q,o,d)]['prompt'])
            assert answer!=ans2 and query==query2
            assert {(v,c,b) for _,v,c,b in facts}=={(v,c,b) for _,v,c,b in facts2}
            assert all(x[0]!=y[0] for x,y in zip(facts,facts2))
            ans2,facts2,query2=parse(by[(a,1-q,o,d)]['prompt'])
            assert answer!=ans2 and facts==facts2 and query!=query2
            if extra==2:
                assert parse(by[(a,q,o,1-d)]['prompt'])==parse(r['prompt'])
    # Adversarial checks: every inverted key must fail this independent semantic rule.
    rejected=sum(r['candidates'][1-r['correct_index']]!=' '+parse(r['prompt'])[0]+'.' for r in controlled)
    assert rejected==len(controlled)
    return {'controlled_keys_checked':len(controlled),'families_checked':len(groups),
            'key_disagreements':0,'contradictory_or_ambiguous_controlled_prompts':0,
            'inverted_keys_rejected':rejected,'fact_order_assignment_query_and_distractor_transformations':'PASS'}
