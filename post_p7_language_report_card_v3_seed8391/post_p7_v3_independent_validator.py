"""Independent final-text parser; no builder imports and no model access."""
import re
from collections import Counter

def parse(prompt):
    body,q=prompt.split('\n'); rec=[]
    for s in body.split('. '):
        w=s.rstrip('.').split()
        if w==['Mia','watched','a','bird','by','the','window']: continue
        if w[0]=='The':
            assert len(w)==7 and w[3]=='was' and w[5]=='by'; color,obj,verb,agent=w[1],w[2],w[4],w[6]
        else:
            assert len(w)==5 and w[2]=='the'; agent,verb,color,obj=w[0],w[1],w[3],w[4]
        assert agent in ('Sam','Tom') and verb in ('found','carried'); rec.append((agent,verb,color,obj))
    assert len(rec)==2 and len({r[0] for r in rec})==2 and len({r[1] for r in rec})==2
    if q.startswith('Who '):
        z=q[4:-1].split(); assert len(z)==4 and z[0] in ('found','carried') and z[1]=='the'; key=(z[0],z[2],z[3])
    else:
        z=q.split()
        if z[:4]==['Based','on','these','facts,']: z=z[4:]
        assert z[:3] in (['The','person','who'],['the','person','who']) and z[-1]=='was', q; key=(z[3],z[5],z[6])
    hits=[r[0] for r in rec if r[1:]==key]; assert len(hits)==1; return hits[0],rec,key

def validate(items):
    ctrl=[r for r in items if r['section']!='naturalistic']; groups={}
    for r in ctrl:
        ans,rec,key=parse(r['prompt']); assert r['candidates'][r['correct_index']]==' '+ans+'.'; groups.setdefault(r['family_id'],[]).append(r)
    for fid,rs in groups.items():
        mult=2 if rs[0]['section']=='distractor' else 1
        if mult==1:
            assert len(rs)==24
            formats=set(r['format'] for r in rs); assert formats=={'continuation','cloze','qa'}
            for fmt in formats:
                sub=[r for r in rs if r['format']==fmt]; assert len(sub)==8
                assert Counter(r['assignment'] for r in sub)=={0:4,1:4}; assert Counter(r['query'] for r in sub)=={0:4,1:4}; assert Counter(r['fact_order'] for r in sub)=={0:4,1:4}
            rs=[r for r in rs if r['format']=='continuation']
        else: assert len(rs)==8*mult
        assert Counter(r['assignment'] for r in rs)=={0:4*mult,1:4*mult}
        assert Counter(r['query'] for r in rs)=={0:4*mult,1:4*mult}
        assert Counter(r['fact_order'] for r in rs)=={0:4*mult,1:4*mult}
        keys={(r['assignment'],r['query'],r['fact_order'],r.get('distractor_position',-1)):r for r in rs}; assert len(keys)==len(rs)
        for k,r in keys.items():
            a,q,o,d=k; other=keys[(a,q,1-o,d)]; assert parse(r['prompt'])[1]==list(reversed(parse(other['prompt'])[1])); assert parse(r['prompt'])[0]==parse(other['prompt'])[0]
            swapped=keys[(1-a,q,o,d)]; assert parse(r['prompt'])[0]!=parse(swapped['prompt'])[0]
            queried=keys[(a,1-q,o,d)]; assert parse(r['prompt'])[0]!=parse(queried['prompt'])[0]
            if mult==2: assert parse(keys[(a,q,o,1-d)]['prompt'])[0]==parse(r['prompt'])[0]
    return {'controlled_keys_checked':len(ctrl),'families_checked':len(groups),'key_disagreements':0,'contradictory_or_ambiguous_prompts':0,'transformations':'PASS'}
