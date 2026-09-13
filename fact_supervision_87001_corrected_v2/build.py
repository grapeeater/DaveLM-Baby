"""Prospective construction only. No checkpoint loader, inference, optimizer, or training.

All prospective choices are recorded before materialization. Any scientific validation
failure produces a failure receipt, never a repaired/resampled frozen battery.
"""
import hashlib
import itertools
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import independent_validator as independent

ROOT = Path(r'C:\DaveLM-CADAVER')
HERE = Path(__file__).resolve().parent
SEED = 87001
NAMES = ['Alex', 'Mia', 'Nora', 'Owen']
OBJECTS = ['ball', 'book', 'box', 'car', 'fish', 'hat', 'kite', 'toy']
COLORS = ['blue', 'green', 'red', 'yellow']
VERBS = ['found', 'carried']
TOKENIZER = Path(r'C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json')
TOKENIZER_HASH = 'e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b'
COUNTS = {'train': 24, 'dev': 6, 'primary': 12, 'confirmation': 12}


def canonical(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(x):
    return hashlib.sha256(x.encode('utf-8')).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def norm(s):
    return ' '.join(s.casefold().split())


def decoded_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, list):
        for x in obj:
            yield from decoded_strings(x)
    elif isinstance(obj, dict):
        for x in obj.values():
            yield from decoded_strings(x)
        prompt = obj.get('prompt')
        if isinstance(prompt, str):
            for k in ('response', 'continuation', 'generation', 'correct_candidate', 'correct_answer', 'answer', 'text'):
                if isinstance(obj.get(k), str):
                    yield prompt + obj[k]
            for x in obj.get('candidates', []):
                if isinstance(x, str):
                    yield prompt + x


class Corpus:
    def __init__(self, tok):
        audit = ROOT / 'post_p7_language_report_card_v3d_seed8391' / 'OVERLAP_AUDIT.json'
        previous = json.loads(audit.read_text(encoding='utf-8'))['sources']
        paths = {Path(r['path']): r['sha256'] for r in previous}
        # Explicitly bounded, nonsacred English evaluation artifacts added after the old audit.
        for directory in list(ROOT.glob('post_p7_language_report_card*')) + list(ROOT.glob('english_context_characterization*')) + list(ROOT.glob('post_p7_v3d_stage1_execution*')):
            if not directory.is_dir():
                continue
            for name in ('ITEMS.jsonl', 'PRIORS.jsonl', 'RAW_SCORES.jsonl', 'RAW_SCORES.json', 'RAW_GENERATIONS.json', 'RAW_GENERATIONS.jsonl'):
                p = directory / name
                if p.exists():
                    paths.setdefault(p, None)
        self.records = []
        self.sources = []
        for path, expected in sorted(paths.items(), key=lambda x: str(x[0])):
            allowed_binding = {ROOT / d / f for d in ('language_pilot_0_tinystories_seed8380', 'language_pilot_1_early_block_protection_seed8380') for f in ('binding_dev.json', 'binding_rehearsal.json')}
            assert 'treatment' not in str(path).lower()
            assert 'binding' not in str(path).lower() or path in allowed_binding
            sha = file_hash(path)
            assert expected is None or sha == expected, f'Historical input mismatch: {path}'
            text = path.read_text(encoding='utf-8-sig')
            if path.suffix.lower() == '.json':
                obj = json.loads(text)
                strings = list(decoded_strings(obj))
                if path in allowed_binding:
                    strings.extend(tok.decode(d['full_document_token_ids'], skip_special_tokens=False) for q in obj['quartets'] for d in q['docs'])
            elif path.suffix.lower() == '.jsonl':
                strings = [s for line in text.splitlines() if line.strip() for s in decoded_strings(json.loads(line))]
            else:
                strings = [text]
            normalized = '\0'.join(norm(s) for s in strings)
            self.records.append((str(path), normalized))
            self.sources.append({'path': str(path), 'sha256': sha, 'decoded_strings': len(strings)})
        self.joined = '\0'.join(s for _, s in self.records)
        self.exact_strings = {x for _, s in self.records for x in s.split('\0') if x}

    def any_overlap(self, rows):
        for r in rows:
            if norm(r['prompt']) in self.exact_strings:
                return True
            for c in r.get('candidates', []):
                if norm(r['prompt'] + c) in self.exact_strings:
                    return True
        return False

    def audit(self, rows):
        hits = []
        for r in rows:
            tests = [('prompt', r['prompt'])] + [('completed:' + str(i), r['prompt'] + c) for i, c in enumerate(r.get('candidates', []))]
            for field, s in tests:
                needle = norm(s)
                if needle in self.exact_strings:
                    hits.append({'id': r['id'], 'field': field, 'sources': [p for p, content in self.records if needle in content]})
        return {'normalization': 'Decode JSON/JSONL string values, reconstruct recognized prompt+response/candidate pairs; casefold and collapse Unicode whitespace; NUL between records',
                'sources': self.sources, 'hits': hits,
                'limitation': 'Exact non-overlap does not exclude semantic, template, or near-duplicate exposure; sacred material intentionally not read.'}


def descriptors(partition):
    residues = {'train': {0, 1}, 'dev': {2}, 'primary': {3}, 'confirmation': {3}}[partition]
    return sorted(f'{c} {o}' for i, o in enumerate(OBJECTS)
                  for j, c in enumerate(COLORS) if (i+j) % 4 in residues)


def family_identity(f):
    return {k: f[k] for k in ('stratum', 'names', 'descriptions', 'predicates')}


def universe(partition, stratum):
    ds = descriptors(partition)
    for ns in itertools.combinations(NAMES, 2):
        if stratum == 'object':
            for pair in itertools.combinations(ds, 2):
                for v in VERBS:
                    yield {'stratum': stratum, 'names': list(ns), 'descriptions': list(pair), 'predicates': [v, v]}
        else:
            for d in ds:
                yield {'stratum': stratum, 'names': list(ns), 'descriptions': [d, d], 'predicates': VERBS}


def facts_for(f, a):
    return [(f['names'][i ^ a], f['predicates'][i], f['descriptions'][i]) for i in (0, 1)]


def render(facts, key, order, frame='cloze', distractor=None):
    sentences = [f'{n} {v} the {d}.' if frame != 'passive' else f'The {d} was {v} by {n}.' for n, v, d in facts]
    if order:
        sentences.reverse()
    if distractor == 'before':
        sentences.insert(0, 'Sam watched a bird by the window.')
    elif distractor == 'after':
        sentences.append('Sam watched a bird by the window.')
    v, d = key
    cue = f'Who {v} the {d}?' if frame == 'qa' else f'The person who {v} the {d} was'
    return ' '.join(sentences) + '\n' + cue


def rendered_answer(prompt):
    """Builder derivation uses regex over final text, not assignment metadata."""
    body, cue = prompt.split('\n')
    query = re.fullmatch(r'(?:The person who |Who )(found|carried) the (\w+ \w+)(?: was|\?)', cue)
    assert query
    v, desc = query.groups()
    actors = []
    for s in re.findall(r'[^.]+\.', body):
        s = s.strip()
        active = re.fullmatch(r'(Alex|Mia|Nora|Owen) (found|carried) the (\w+ \w+)\.', s)
        passive = re.fullmatch(r'The (\w+ \w+) was (found|carried) by (Alex|Mia|Nora|Owen)\.', s)
        if active:
            n, fv, fd = active.groups()
        elif passive:
            fd, fv, n = passive.groups()
        else:
            assert s == 'Sam watched a bird by the window.'
            continue
        if (fv, fd) == (v, desc):
            actors.append(n)
    return actors


def make_factual(f, a, q, o, frame='cloze', distractor=None):
    key = f['predicates'][q], f['descriptions'][q]
    prompt = render(facts_for(f, a), key, o, frame, distractor)
    hits = rendered_answer(prompt)
    assert len(hits) == 1
    cs = [' ' + n + '.' for n in f['names']]
    row = {'id': f"{f['family_id']}:factual:{frame}:{distractor or 'none'}:a{a}q{q}o{o}",
           'partition': f['partition'], 'family_id': f['family_id'], 'stratum': f['stratum'],
           'arm': 'factual', 'frame': frame, 'distractor': distractor,
           'assignment': a, 'query': q, 'fact_order': o, 'prompt': prompt,
           'candidates': cs, 'correct_index': cs.index(' ' + hits[0] + '.'),
           'answer_status': 'entailed_stated_actor'}
    return row


def choose_families(overlap_check=None):
    selected, used, exclusions = [], set(), []
    for partition, count in COUNTS.items():
        for stratum in ('object', 'predicate'):
            ordered = sorted(universe(partition, stratum), key=lambda f: digest(f'{SEED}|{partition}|{canonical(f)}'))
            quota = Counter()
            chosen = []
            for f in ordered:
                identity = canonical(f)
                if identity in used:
                    continue
                pair = tuple(f['names'])
                per_pair = count // 6
                if quota[pair] >= per_pair:
                    continue
                if stratum == 'object':
                    verb = f['predicates'][0]
                    if partition != 'dev' and quota[(pair, verb)] >= per_pair // 2:
                        continue
                    if partition == 'dev' and quota[verb] >= 3:
                        continue
                f = dict(f, partition=partition, family_id=f'{partition}:{stratum}:{digest(identity)[:16]}')
                if overlap_check is not None:
                    rendered = [make_factual(f, a, q, o, frame) for a, q, o, frame in itertools.product((0, 1), (0, 1), (0, 1), ('cloze', 'qa', 'passive'))]
                    if overlap_check(rendered):
                        exclusions.append(f['family_id'])
                        continue
                chosen.append(f)
                used.add(identity)
                quota[pair] += 1
                if stratum == 'object':
                    quota[(pair, f['predicates'][0])] += 1
                    quota[f['predicates'][0]] += 1
                if len(chosen) == count:
                    break
            assert len(chosen) == count, f'Insufficient eligible families: {partition}/{stratum}'
            selected.extend(chosen)
    return selected, exclusions


def tokenize_row(row, tok):
    p = tok.encode(row['prompt'], add_special_tokens=False).ids
    assert tok.decode(p, skip_special_tokens=False) == row['prompt']
    cs = []
    for c in row['candidates']:
        ids = tok.encode(c, add_special_tokens=False).ids
        assert len(ids) == 4 and ids[-1] == tok.token_to_id('.')
        assert tok.decode(ids, skip_special_tokens=False) == c
        assert tok.encode(row['prompt'] + c, add_special_tokens=False).ids == p + ids
        assert tok.decode(p + ids, skip_special_tokens=False) == row['prompt'] + c
        assert max(p + ids) < 1024 and min(p + ids) >= 0 and not ({0, 1, 2, 3, 4} & set(p + ids))
        assert 1 + len(p) + len(ids) + 1 <= 256
        cs.append(ids)
    row['prompt_token_ids'], row['candidate_token_ids'] = p, cs
    return row


def construct_rows(families, tok):
    rows = []
    for index, f in enumerate(families):
        factual = {(a, q, o): tokenize_row(make_factual(f, a, q, o), tok)
                   for a, q, o in itertools.product((0, 1), repeat=3)}
        control = {}
        for q in (0, 1):
            allowed = [d for d in descriptors(f['partition']) if d != f['descriptions'][q]]
            pairs = list(itertools.permutations(allowed, 2)) if f['stratum'] == 'object' else [(d, d) for d in allowed]
            b = q ^ (index % 2)
            key = f['predicates'][q], f['descriptions'][q]
            def control_facts(ds):
                return [(f['names'][i ^ b], f['predicates'][i], ds[i]) for i in (0, 1)]
            def cost(ds):
                delta = [len(tok.encode(render(control_facts(ds), key, o), add_special_tokens=False).ids)
                         - len(factual[a, q, o]['prompt_token_ids']) for a, o in itertools.product((0, 1), repeat=2)]
                return max(map(abs, delta)), sum(map(abs, delta)), digest(f'{SEED}|control|{f["family_id"]}|{q}|{canonical(ds)}')
            best = min(pairs, key=cost)
            for a, o in itertools.product((0, 1), repeat=2):
                source = factual[a, q, o]
                prompt = render(control_facts(best), key, o)
                assert rendered_answer(prompt) == []
                r = {k: v for k, v in source.items() if k not in ('correct_index', 'prompt_token_ids', 'candidate_token_ids')}
                r.update(id=source['id'].replace(':factual:', ':control:'), prompt=prompt,
                         arm='control', answer_status='underdetermined', practice_target=source['candidates'][source['correct_index']],
                         nuisance_assignment=b)
                control[a, q, o] = tokenize_row(r, tok)
        rows.extend(factual.values())
        rows.extend(control.values())
    return rows


def transfer_rows(families, tok):
    controlled, natural = [], []
    selected_natural = {}
    used_natural_prompts = set()
    for partition in ('primary', 'confirmation'):
        fs = [f for f in families if f['partition'] == partition]
        for f in fs:
            for a, q, o, frame in itertools.product((0, 1), (0, 1), (0, 1), ('qa', 'passive')):
                r = tokenize_row(make_factual(f, a, q, o, frame), tok)
                independent.check_row(r)
                controlled.append(r)
        for stratum in ('object', 'predicate'):
            subset = sorted((f for f in fs if f['stratum'] == stratum), key=lambda f: canonical(family_identity(f)))[:3]
            for f in subset:
                for a, q, o, position in itertools.product((0, 1), (0, 1), (0, 1), ('before', 'after')):
                    r = tokenize_row(make_factual(f, a, q, o, distractor=position), tok)
                    independent.check_row(r)
                    controlled.append(r)
        candidates = sorted((f for f in fs if f['stratum'] == 'object'), key=lambda f: canonical(family_identity(f)))
        chosen = None
        for combo in itertools.combinations(candidates, 8):
            prompt_sets = [{f'{n} {v} the {d} and then went home.' for n, v, d in facts_for(f, 0)} for f in combo]
            local = set().union(*prompt_sets)
            if len(local) != 16 or local & used_natural_prompts:
                continue
            chosen = list(combo)
            break
        assert chosen is not None, f'Unable to select eight unique naturalistic families for {partition}'
        local = {f'{n} {v} the {d} and then went home.' for f in chosen for n, v, d in facts_for(f, 0)}
        selected_natural[partition] = [f['family_id'] for f in chosen]
        used_natural_prompts |= local
        for f in chosen:
            for i, (n, v, d) in enumerate(facts_for(f, 0)):
                prompt = f'{n} {v} the {d} and then went home.'
                ids = tok.encode(prompt, add_special_tokens=False).ids
                assert tok.decode(ids, skip_special_tokens=False) == prompt
                natural.append({'id': f"{f['family_id']}:natural:{i}", 'partition': partition,
                                'family_id': f['family_id'], 'prompt': prompt, 'prompt_token_ids': ids,
                                'scoring': 'HUMAN_REVIEW_REQUIRED; descriptive only'})
    return controlled, natural, selected_natural


def uniqueness(rows, transfers, natural):
    factual = [r for r in rows if r['arm'] == 'factual'] + transfers
    counts = Counter(r['prompt'] for r in factual)
    duplicates = {p: n for p, n in counts.items() if n != 1}
    nc = Counter(r['prompt'] for r in natural)
    nd = {p: n for p, n in nc.items() if n != 1}
    return {'factual_controlled_count': len(factual), 'unique_factual_prompts': len(counts),
            'duplicate_factual_prompts': duplicates, 'naturalistic_count': len(natural),
            'unique_naturalistic_prompts': len(nc), 'duplicate_naturalistic_prompts': nd}


def main():
    from tokenizers import Tokenizer
    assert file_hash(TOKENIZER) == TOKENIZER_HASH
    tok = Tokenizer.from_file(str(TOKENIZER))
    assert not json.loads(TOKENIZER.read_text(encoding='utf-8'))['model']['byte_fallback']
    corpus = Corpus(tok)
    print('Verified and decoded corpus sources:', len(corpus.sources), flush=True)
    families, exclusions = choose_families(corpus.any_overlap)
    rows = construct_rows(families, tok)
    transfers, natural, selected_natural = transfer_rows(families, tok)
    checks = independent.validate(rows)
    check_unique = uniqueness(rows, transfers, natural)
    overlap = corpus.audit(rows + transfers + natural)
    deltas = defaultdict(list)
    byid = {r['id']: r for r in rows}
    for r in rows:
        if r['arm'] == 'factual':
            control = byid[r['id'].replace(':factual:', ':control:')]
            deltas[r['partition'] + ':' + r['stratum']].append(len(control['prompt_token_ids']) - len(r['prompt_token_ids']))
    length_checks = {k: {'minimum': min(v), 'maximum': max(v), 'mean_signed': sum(v)/len(v)} for k, v in deltas.items()}
    issues = []
    if overlap['hits']:
        issues.append('Final rendered material overlaps listed historical text')
    if check_unique['duplicate_factual_prompts']:
        issues.append('Duplicate factual controlled prompts')
    if check_unique['duplicate_naturalistic_prompts']:
        issues.append('Duplicate naturalistic prompts under prescribed first-eight-family construction')
    if any(abs(z) > 4 for v in deltas.values() for z in v):
        issues.append('Paired context length differs by more than four tokens')
    if any(abs(sum(v)/len(v)) > 1 for k, v in deltas.items() if k.startswith('train:')):
        issues.append('TRAIN mean signed context length difference exceeds one token')
    result = {'status': 'STOP_NOT_FROZEN' if issues else 'CONSTRUCTION_CHECKS_PASS_NOT_FROZEN',
              'seed': SEED, 'issues': issues, 'independent_validation': checks, 'uniqueness': check_unique,
              'length_checks': length_checks, 'tokenizer_sha256': file_hash(TOKENIZER),
              'family_counts': dict(Counter(f['partition'] + ':' + f['stratum'] for f in families)),
              'selected_naturalistic_family_ids': selected_natural,
              'checkpoint_loaded': False, 'inference': False, 'training': False, 'sacred_access': False,
              'corpus_overlap_audit': overlap, 'excluded_family_ids': exclusions, 'freeze_created': False,
              'source_hashes': {p.name: file_hash(p) for p in HERE.glob('*.py')}}
    out = HERE / 'construction_check.json'
    assert not out.exists(), 'Never overwrite a completed construction check'
    out.write_bytes((json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode())
    # Prospective identities are preserved for auditing a stop, not declared a frozen exam.
    payload = HERE / 'construction_candidates.json'
    assert not payload.exists()
    payload.write_bytes((json.dumps({'families': families, 'rows': rows, 'transfers': transfers, 'naturalistic': natural}, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
