"""Tokenizer-independent semantic and mock-tensor tests; no Baby imports or loading."""
import copy
import json
import math
import tempfile
from pathlib import Path

import torch
import harness
import independent_validator as validator


def rejected(call):
    try:
        call()
    except (AssertionError, ValueError, RuntimeError, KeyError):
        return True
    raise AssertionError('Negative test was incorrectly accepted')


def run(rows):
    sample = next(r for r in rows if r['arm'] == 'factual' and r['stratum'] == 'object')
    negatives = []
    wrong = copy.deepcopy(sample)
    wrong['correct_index'] = 1 - wrong['correct_index']
    rejected(lambda: validator.check_row(wrong)); negatives.append('wrong_final_answer')
    wrong = copy.deepcopy(sample)
    body, cue = wrong['prompt'].split('\n')
    verb = 'carried' if ' found ' in cue else 'found'
    cue_words = cue.split(); cue_words[3] = verb
    wrong['prompt'] = body + '\n' + ' '.join(cue_words)
    rejected(lambda: validator.check_row(wrong)); negatives.append('query_predicate_not_asserted')
    wrong = copy.deepcopy(sample)
    wrong['prompt'] = wrong['prompt'].replace('the ', '', 1)
    rejected(lambda: validator.check_row(wrong)); negatives.append('malformed_fact')
    wrong = copy.deepcopy(sample)
    wrong['arm'] = 'control'; wrong['answer_status'] = 'underdetermined'
    wrong['practice_target'] = wrong['candidates'][wrong.pop('correct_index')]
    rejected(lambda: validator.check_row(wrong)); negatives.append('informative_control')
    subset = copy.deepcopy([r for r in rows if r['family_id'] == sample['family_id']])
    r0 = next(r for r in subset if r['arm'] == 'factual' and r['assignment'] == 0 and r['query'] == 0 and r['fact_order'] == 0)
    r1 = next(r for r in subset if r['arm'] == 'factual' and r['assignment'] == 1 and r['query'] == 0 and r['fact_order'] == 0)
    r1['prompt'], r1['correct_index'] = r0['prompt'], r0['correct_index']
    rejected(lambda: validator.validate(subset)); negatives.append('broken_assignment_reversal')
    control = [r for r in rows if r['family_id'] == sample['family_id'] and r['arm'] == 'control']
    validator.check_control_batch(control)
    rejected(lambda: validator.check_control_batch(control[:-1])); negatives.append('unbalanced_control_target_twins')

    x, y = harness.prepare_example(sample)
    c = sample['candidate_token_ids'][sample['correct_index']]
    assert x[0] == 2 and y[-5:] == c + [3]
    assert all(v == -100 for v in y[:-5])
    assert sum(v != -100 for v in y) == 5 and 3 not in x
    padded_x, padded_y = harness.pad_batch([sample, control[0]], max(len(x), len(harness.prepare_example(control[0])[0]))+4)
    assert int((padded_y != -100).sum()) == 10
    assert torch.all(padded_x[:, -4:] == 0) and torch.all(padded_y[:, -4:] == -100)
    z = torch.zeros((1, len(sample['prompt_token_ids'])+8, 1024), dtype=torch.float32)
    scores = harness.candidate_scores(z, len(sample['prompt_token_ids']), c)
    assert math.isclose(scores['candidate_ll'], -4*math.log(1024), abs_tol=1e-12)
    assert math.isclose(scores['word_only_ll'], -3*math.log(1024), abs_tol=1e-12)
    assert math.isclose(scores['eos_terminated_ll'], -5*math.log(1024), abs_tol=1e-12)
    # Nonuniform logits ensure offset and target token indexing are tested independently.
    for j, token in enumerate(c+[3]):
        z[0, len(sample['prompt_token_ids'])+j, token] = 1+j
    values = harness.candidate_scores(z, len(sample['prompt_token_ids']), c)['token_log_probabilities']
    assert all(math.isclose(v, k-math.log(math.exp(k)+1023), abs_tol=1e-12) for k,v in enumerate(values, 1))
    eos = harness.eos_diagnostic(torch.zeros(1024))
    assert eos['rank'] == 1 and not eos['wins_argmax'] and math.isclose(eos['probability'], 1/1024)
    probe = torch.zeros(1024); probe[3] = 2
    assert harness.eos_diagnostic(probe)['wins_argmax']

    class Mock:
        def __init__(self):
            self.params = {n: torch.nn.Parameter(torch.tensor([1.])) for n in (
                'base_model.token_emb.weight', 'base_model.pos_emb.weight',
                *[f'base_model.blocks.{i}.weight' for i in range(8)],
                'base_model.ln_f.weight', 'base_model.head.weight', 'localizer.weight', 'retrieval.weight')}
        def named_parameters(self):
            return self.params.items()
    model = Mock()
    for p in model.params.values():
        p.grad = torch.ones_like(p)
    harness.set_scope(model, binding=False)
    for name,p in model.named_parameters():
        expected = name.startswith('base_model.') and not any(name.startswith(f'base_model.blocks.{i}.') for i in range(4))
        assert p.requires_grad == expected and p.grad is None
    harness.set_scope(model, binding=True)
    assert all(p.requires_grad and p.grad is None for p in model.params.values())
    with tempfile.TemporaryDirectory(prefix='davelm_journal_mock_') as td:
        path = Path(td)/'journal.jsonl'
        j = harness.Journal(path)
        assert j.start('one')
        rejected(lambda: harness.Journal(path).start('two'))
        j.finish('one', {'raw': '  verbatim\n'})
        j = harness.Journal(path)
        assert not j.start('one') and j.completed['one']['raw'] == '  verbatim\n'
        assert j.start('two'); j.finish('two', {'result': 2})
    return {'status': 'PASS', 'negative_tests': negatives, 'mock_tensor_checks': [
        'five response/EOS targets only', 'right padding ignored', 'candidate likelihood positions and full-vocabulary denominator',
        'EOS probability/rank/tie', 'scope and gradient clearing', 'durable no-replay journal'],
        'baby_imported': False, 'checkpoint_loaded': False, 'backward_called': False, 'optimizer_created': False,
        'torch_version': torch.__version__}


if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    data = json.loads((here/'construction_candidates.json').read_text(encoding='utf-8'))
    result = run(data['rows'])
    path = here/'MOCK_TEST_RESULTS.json'
    assert not path.exists()
    path.write_bytes((json.dumps(result, sort_keys=True, indent=2)+'\n').encode())
    print(json.dumps(result, indent=2))
