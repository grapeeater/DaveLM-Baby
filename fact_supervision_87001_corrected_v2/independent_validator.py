"""Final-text grammar and empirical-independence checks. Never imports builder/model."""
from collections import Counter, defaultdict

NAMES = {'Alex', 'Mia', 'Nora', 'Owen'}
VERBS = {'found', 'carried'}
COLORS = {'blue', 'green', 'red', 'yellow'}
OBJECTS = {'ball', 'book', 'box', 'car', 'fish', 'hat', 'kite', 'toy'}

def parse(prompt):
    body, cue = prompt.split('\n')
    facts = []
    for sentence in body.split('. '):
        w = sentence.rstrip('.').split()
        if w == ['Sam', 'watched', 'a', 'bird', 'by', 'the', 'window']:
            continue
        if w[0] == 'The':
            assert len(w) == 7 and w[3] == 'was' and w[5] == 'by'
            a, v, c, o = w[6], w[4], w[1], w[2]
        else:
            assert len(w) == 5 and w[2] == 'the'
            a, v, c, o = w[0], w[1], w[3], w[4]
        assert a in NAMES and v in VERBS and c in COLORS and o in OBJECTS
        facts.append((a, v, c, o))
    assert len(facts) == 2 and len({f[0] for f in facts}) == 2
    w = cue.split()
    if w[0] == 'Who':
        assert len(w) == 5 and w[2] == 'the' and w[4].endswith('?')
        key = w[1], w[3], w[4][:-1]
    else:
        assert len(w) == 8 and w[:3] == ['The', 'person', 'who'] and w[4] == 'the' and w[7] == 'was'
        key = w[3], w[5], w[6]
    assert key[0] in VERBS and key[1] in COLORS and key[2] in OBJECTS
    return facts, key

def check_row(row):
    facts, key = parse(row['prompt'])
    assert sorted(f[0] for f in facts) == [s[1:-1] for s in row['candidates']]
    hits = [a for a, v, c, o in facts if (v, c, o) == key]
    if row['arm'] == 'factual':
        assert len(hits) == 1
        assert row['candidates'][row['correct_index']] == ' ' + hits[0] + '.'
        assert row['answer_status'] == 'entailed_stated_actor'
    else:
        assert row['arm'] == 'control' and not hits
        assert all(f[2:] != key[1:] for f in facts)
        assert 'correct_index' not in row and 'correct_answer' not in row
        assert row['answer_status'] == 'underdetermined'
        assert row['practice_target'] in row['candidates']
    if row['stratum'] == 'object':
        assert facts[0][1] == facts[1][1] == key[0]
        assert facts[0][2:] != facts[1][2:]
    else:
        assert facts[0][2:] == facts[1][2:]
        assert {f[1] for f in facts} == VERBS
    return facts, key

def validate(rows):
    groups = defaultdict(list)
    for r in rows:
        check_row(r)
        groups[(r['partition'], r['family_id'], r['arm'])].append(r)
    for (partition, fid, arm), rs in groups.items():
        assert len(rs) == 8
        by = {(r['assignment'], r['query'], r['fact_order']): r for r in rs}
        assert len(by) == 8
        for field in ('assignment', 'query', 'fact_order'):
            assert Counter(r[field] for r in rs) == {0: 4, 1: 4}
        targets = [r['candidates'][r['correct_index']] if arm == 'factual' else r['practice_target'] for r in rs]
        assert Counter(targets) == {c: 4 for c in rs[0]['candidates']}
        for (a, q, o), r in by.items():
            fs, key = check_row(r)
            ofs, okey = check_row(by[a, q, 1-o])
            assert fs == ofs[::-1] and key == okey
            flip = by[1-a, q, o]
            if arm == 'control':
                assert r['prompt'] == flip['prompt']
                assert r['practice_target'] != flip['practice_target']
            else:
                ffs, fkey = check_row(flip)
                assert fkey == key and r['correct_index'] != flip['correct_index']
                assert [f[1:] for f in fs] == [f[1:] for f in ffs]
                assert all(x[0] != y[0] for x, y in zip(fs, ffs))
                qfs, qkey = check_row(by[a, 1-q, o])
                assert fs == qfs and key != qkey
                assert r['correct_index'] != by[a, 1-q, o]['correct_index']
    for r in rows:
        if r['arm'] != 'factual':
            continue
        peer = next(z for z in groups[(r['partition'], r['family_id'], 'control')]
                    if (z['assignment'], z['query'], z['fact_order']) == (r['assignment'], r['query'], r['fact_order']))
        assert r['prompt'].split('\n')[1] == peer['prompt'].split('\n')[1]
        assert r['candidates'] == peer['candidates']
        assert r['candidates'][r['correct_index']] == peer['practice_target']
    return {'rows_checked': len(rows), 'family_arm_groups': len(groups), 'answer_key_disagreements': 0,
            'control_independence': 'exact target twins for each input and candidate pair',
            'transformations': 'PASS', 'cross_arm_cues_targets_candidates': 'PASS'}

def check_control_batch(rows):
    counts = defaultdict(Counter)
    for r in rows:
        check_row(r)
        counts[(r['prompt'], tuple(r['candidates']))][r['practice_target']] += 1
    for (_, cs), counts_for_prefix in counts.items():
        assert counts_for_prefix[cs[0]] == counts_for_prefix[cs[1]] > 0
    return len(counts)
