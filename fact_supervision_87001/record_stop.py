"""Record the failed prospective construction and an additive historical erratum."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def norm(s):
    return ' '.join(s.casefold().split())


def read_json(p):
    return json.loads(p.read_text(encoding='utf-8'))


def write_new(name, value):
    p = HERE/name
    assert not p.exists()
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False)+'\n'
    p.write_bytes(text.encode('utf-8'))


def verify_dir(path, expected):
    receipt = path/'SHA256SUMS.txt'
    assert sha(receipt) == expected
    count = 0
    for line in receipt.read_text(encoding='utf-8').splitlines():
        h, name = line.split('  ', 1)
        p = path/name
        assert p.resolve().parent == path.resolve()
        assert sha(p) == h, p
        count += 1
    return {'path': str(path), 'detached_sha256': expected, 'files_verified': count}


def main():
    result = read_json(HERE/'construction_check.json')
    data = read_json(HERE/'construction_candidates.json')
    assert result['status'] == 'STOP_NOT_FROZEN'
    by_prompt = defaultdict(list)
    for r in data['naturalistic']:
        by_prompt[r['prompt']].append({'id': r['id'], 'partition': r['partition']})
    duplicates = {p: rs for p, rs in by_prompt.items() if len(rs)>1}
    parts = {p: {'items': len(rs := [r for r in data['naturalistic'] if r['partition'] == p]),
                 'unique': len({r['prompt'] for r in rs})} for p in ('primary', 'confirmation')}
    intersect = sorted({r['prompt'] for r in data['naturalistic'] if r['partition']=='primary'} &
                       {r['prompt'] for r in data['naturalistic'] if r['partition']=='confirmation'})
    stop = {'status': 'STOP_NOT_FROZEN_NOT_READY_FOR_TRAINING', 'reason': result['issues'],
            'naturalistic_partition_counts': parts, 'duplicate_prompts': duplicates,
            'primary_confirmation_shared_prompts': intersect,
            'all_scientific_rows_preserved': 'construction_candidates.json',
            'passing_checks': result['independent_validation'],
            'historical_overlap_hits': len(result['corpus_overlap_audit']['hits']),
            'inventoried_historical_sources': len(result['corpus_overlap_audit']['sources']),
            'not_completed': ['final schedules', 'mock harness test execution', 'complete runtime and training runner freeze', 'pretraining scientific freeze'],
            'no_checkpoint_loaded': True, 'no_inference': True, 'no_training': True, 'no_sacred_access': True,
            'mechanical_corrections': [
                'Explicit UTF-8 tokenizer JSON reading replaced Windows implicit cp1252; tokenizer bytes unchanged.',
                'The corpus reader explicitly allowlists the four historical Pilot 0/1 nonsacred binding JSON files present in the old audit. It decodes their document token arrays; no treatment or sacred files are opened.'
            ],
            'required_review': 'Approve a revised naturalistic construction/uniqueness rule. No revised rule has been selected or executed.'}
    write_new('STOP_RECEIPT.json', stop)

    v3 = ROOT/'post_p7_language_report_card_v3d_seed8391'
    execution = ROOT/'post_p7_v3d_stage1_execution_retry'
    integrity = [verify_dir(v3, '9893514299870cd27377a0f64de6e44cc230d13c69ff6869f18a04a4079483b6'),
                 verify_dir(execution, '408083cafd04cbdd9f0fa5a680def347c36ab4354536ffcf737c3da811d78b5d')]
    v2rows = [json.loads(s) for s in (ROOT/'post_p7_language_report_card_v2_seed8391'/'ITEMS.jsonl').read_text(encoding='utf-8').splitlines()]
    v3rows = [json.loads(s) for s in (v3/'ITEMS.jsonl').read_text(encoding='utf-8').splitlines()]
    prompts = {norm(r['prompt']) for r in v2rows}
    contexts = {norm(r['prompt'].split('\n')[0]) for r in v2rows}
    exact = Counter(r['section'] for r in v3rows if norm(r['prompt']) in prompts)
    primary = [r for r in v3rows if r['section'] in ('near_distribution', 'counterfactual', 'surface_form')]
    context_overlap = sum(norm(r['prompt'].split('\n')[0]) in contexts for r in primary)
    priors = [r for r in v3rows if r['section']=='query_only_prior']
    prior_substring = sum(any(norm(r['prompt']) in p for p in prompts) for r in priors)
    raw = read_json(execution/'RAW_SCORES.json')
    assert all(r['correct'] is None and r['margin'] is None for r in raw if r['section']=='query_only_prior')
    prior_raw_count = sum(r['section']=='query_only_prior' for r in raw)
    first_dir = ROOT/'post_p7_v3d_stage1_execution'
    runner = ROOT/'execute_stage1_v3d.py'
    historical = {'integrity': integrity, 'v2_items_sha256': sha(ROOT/'post_p7_language_report_card_v2_seed8391'/'ITEMS.jsonl'),
                  'exact_prompt_overlap_by_section': dict(exact), 'primary_context_overlap': context_overlap,
                  'query_only_substring_overlap': prior_substring, 'raw_prior_records_without_answer_key': prior_raw_count,
                  'first_attempt_files': sorted(p.name for p in first_dir.iterdir() if p.is_file()),
                  'currently_available_runner_sha256': sha(runner),
                  'runner_historical_identity': 'Current source is explanatory evidence; executed source hash was not saved in the historical output manifest.'}
    write_new('V3D_ERRATUM_EVIDENCE.json', historical)
    erratum = f'''# Additive v3d historical-status erratum

This new record supersedes stronger freshness/completeness interpretations. No v3d
artifact, prior report, checkpoint, or historical checksum was changed.

Retain the six-checkpoint likelihood and two-pool binding measurements as descriptive
evidence on that frozen battery. Withdraw the claim that v3d was a fully fresh
prospective transfer panel. Do not use it to select a parent or establish new acquisition.

## Independently reproduced overlap limitation

Decoded v2-to-v3d exact prompt matches by section: {json.dumps(dict(exact), sort_keys=True)}.
Of 192 primary items, {context_overlap} reused a v2 factual context. Of 16 query-only
prompts, {prior_substring} occurred as substrings of v2 prompts. Duplicate exposure is
not itself evidence of training contamination, but precludes the fully fresh panel claim.
The old builder searched serialized file text instead of decoded JSON strings; it also
exempted naturalistic/query-only overlaps and wrote zero aggregate overlap counts.
Its zero-overlap report is not a reliable decoded-text freshness audit.

## Missing diagnostics and invalid query-only summary

The preserved execution omits first-step EOS probability/rank, common TinyStories DEV
loss, per-token likelihoods, and the requested format-separated summary. Raw query-only
records ({prior_raw_count}) have null correctness and margins because they have no factual
answer key. Their reported zero correct/zero mean margin are aggregation artifacts,
not zero-accuracy measurements. Candidate log-likelihoods remain usable as descriptive priors.
Current runner source also uses punctuation/special-token generation stops; its short
outputs must not be interpreted as EOS-only 32-token generations without this qualification.

## Repeat execution and provenance

The first execution directory contains only a precheck. The autopsy/session history
reports that English computation completed before a binding-wrapper failure, with raw
English results unsaved, and the retry repeated English inference. That first attempt
cannot be reconstructed or compared from preserved raw scores. The current source saves
English only after binding; this supports the identified persistence failure but is not
a substitute for an executed-source snapshot. Withdraw a pristine single-pass claim.

The successful output manifest preserves result hashes and checkpoint receipts but
does not bind a saved executed runner or runtime/package inventory. The currently
available runner hash is recorded separately as current explanatory evidence, not
retroactively asserted to be the executed hash. The v3d manifest additionally retains
the internal version label "3c". These provenance limitations are preserved, not repaired.

Both detached historical checksum receipts and their listed files verified during
this additive audit. Exact overlap counts and source paths/hashes are in
V3D_ERRATUM_EVIDENCE.json. No historical files were edited or behavior rerun.
'''
    write_new('V3D_HISTORICAL_STATUS_ERRATUM.md', erratum)
    report = f'''# Prospective construction stopped before freeze

The implementation has reached a scientific design conflict. It is NOT a frozen
training/evaluation package and must not be used for training or model selection.

The approved first-eight-object-families rule yields 16 naturalistic entries per
holdout but only {parts['primary']['unique']} unique primary and
{parts['confirmation']['unique']} unique confirmation prompts. Across both holdouts,
32 entries have only 26 unique prompts; {len(intersect)} prompts occur in both holdouts.
Different two-fact families can share one fact. Reducing each fact to a single-sentence
naturalistic prompt therefore loses the family distinction. This is a design conflict,
not an answer-key implementation error.

No entries were removed, replaced, resampled, or rewritten to pass. Review is needed
before changing the naturalistic selection rule, counts, or uniqueness requirement.

## Completed construction checks

- 108 semantic families: train 48, DEV 12, primary 24, confirmation 24; each half object/predicate.
- 1,728 base rows across factual/control arms: 768 train, 192 DEV, 384 primary, 384 confirmation.
- Independent validation passed all base answer keys, 8-member structures, assignment/query/order transformations, matched targets/cues, and exact balanced control target twins.
- 960 additional factual transfer items were rendered and independently parsed (QA, passive wording, distractor placement).
- All 1,824 factual prompts across base and transfer sections are unique.
- All candidate completions are four tokens. UTF-8 text round trips, prompt/candidate boundary concatenation, and completed context limits passed with the required tokenizer hash.
- TRAIN factual/control prompt token lengths match exactly; maximum absolute mismatch elsewhere is one token.
- No exact normalized prompt/completion hits in the 66 inventoried historical sources. This does not establish semantic or near-duplicate non-contamination.

Construction candidates, identifiers, final text, answers, token IDs, and full validation
details are preserved in construction_candidates.json and construction_check.json.
The independent validator imports no builder code. Control twins are intentionally
duplicated inputs with opposite balanced practice targets, not factual answer labels.

## Work not completed after the stop

The final schedules, mock test execution, complete runtime/training harness freeze,
and pretraining freeze receipt remain incomplete. The helper/test source is provisional,
not approved for execution against Baby. The accompanying erratum is additive only.

No checkpoint was loaded. No inference, training, backward pass, optimizer creation,
or sacred access occurred. Historical checkpoints, datasets, and results remain unchanged.

Mechanical startup corrections: explicit UTF-8 tokenizer JSON decoding; explicit
allowlist for the four nonsacred historical binding corpus files already listed in the
old audit. No scientific construction rule changed.
'''
    write_new('STOP_REPORT.md', report)
    files = sorted(p for p in HERE.iterdir() if p.is_file())
    sums = ''.join(f'{sha(p)}  {p.name}\n' for p in files)
    write_new('STOP_SHA256SUMS.txt', sums)
    receipt_hash = sha(HERE/'STOP_SHA256SUMS.txt')
    # Preserve this failed construction as evidence, not as a usable frozen exam.
    for p in HERE.iterdir():
        if p.is_file():
            p.chmod(0o444)
    print(json.dumps({'status': stop['status'], 'partition_counts': parts,
                      'cross_holdout_shared_prompts': len(intersect), 'stop_receipt_sha256': receipt_hash,
                      'erratum': str(HERE/'V3D_HISTORICAL_STATUS_ERRATUM.md')}, indent=2))


if __name__ == '__main__':
    main()
