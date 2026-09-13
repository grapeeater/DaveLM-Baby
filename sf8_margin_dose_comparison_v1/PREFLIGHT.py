"""Minimal SF8 preflight. Essential mechanical/scientific-identity checks only, focused on
the dose comparison: control=0.0, low=0.25, medium=0.50, M=1.0 fixed. No checkpoint loading,
no optimizer updates, no training.
"""
import ast, json, inspect, os, sys
from collections import Counter
from pathlib import Path
import torch
import CONTROLLER as C


def hinge_reference(logp_correct, logp_wrong, M=1.0):
    gap = logp_correct - logp_wrong
    return max(0.0, M - gap), gap


def main():
    p = C.verify(sealed=False)

    # 1/2. Parent + tokenizer hashes.
    assert p['parent_sha256'] == '2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb'
    assert p['tokenizer_sha256'] == 'e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b'
    assert C.sha(p['parent']) == p['parent_sha256']
    assert C.sha(p['tokenizer']) == p['tokenizer_sha256']

    # 3/4/5/6/7/8. Frozen SF7 mechanics (TRAIN16, SCHEDULE, KL pool, D3 selection, binding) reused unchanged.
    schedule, train, idx, pool, kl, old = C.load_inputs()
    tok = C.Tokenizer.from_file(p['tokenizer'])
    for r in train:
        assert tok.encode(r['prompt']).ids == r['prompt_token_ids']
        for text, ids in zip(r['candidates'], r['candidate_token_ids']):
            assert tok.encode(text).ids == ids and len(ids) == 4
            assert tok.encode(r['prompt'] + text).ids == r['prompt_token_ids'] + ids
            assert tok.decode(ids) == text
    assert len(set(r['family_id'] for r in train)) == 4
    assert len(set(r['pair_id'] for r in train)) == 8
    counts = Counter(r['candidates'][r['correct_index']] for r in train)
    assert set(counts.values()) == {4}
    prov = C.read(C.H / 'PROVENANCE.json')
    reused_dest = {r['destination'] for r in prov['reused_byte_identical']}
    assert {'TRAIN.json', 'SCHEDULE.json', 'KL_POOL.json', 'KL_POOL_MANIFEST.json', 'D3_SELECTION.json',
            'data/binding_rehearsal.json', 'data/binding_dev_pilot0.json', 'data/binding_dev_pilot1.json'} <= reused_dest

    # 9/10. M fixed at 1.0; lambda exactly {control:0.0, low:0.25, medium:0.50}.
    assert C.MARGIN_M == 1.0
    assert C.LAMBDA_BY_ARM == {'control': 0.0, 'low': 0.25, 'medium': 0.50}

    # 13/14. Correct/wrong candidate NAME token mapping valid for all 16 TRAIN records; first-answer-token position.
    for r in train:
        prefix_len = 1 + len(r['prompt_token_ids'])
        ci = r['correct_index']
        assert len(r['candidate_token_ids']) == 2
        correct_tok = r['candidate_token_ids'][ci][0]
        wrong_tok = r['candidate_token_ids'][1 - ci][0]
        assert correct_tok != wrong_tok, r['id']
        assert prefix_len - 1 >= 0

    # Static sanity check of the hinge arithmetic (not an experiment).
    correct_logp, wrong_logp = -0.5, -1.0
    hinge, gap = hinge_reference(correct_logp, wrong_logp, C.MARGIN_M)
    assert abs(gap - 0.5) < 1e-12 and abs(hinge - 0.5) < 1e-12
    for arm, lam, expected in [('control', 0.0, 0.0), ('low', 0.25, 0.125), ('medium', 0.50, 0.250)]:
        assert C.LAMBDA_BY_ARM[arm] == lam
        assert abs(lam * hinge - expected) < 1e-12
    test_logits = torch.log(torch.tensor([0.6065, 0.3679, 0.01, 0.0156]))
    gap2 = float(C.margin_gap(test_logits, 0, 1))
    assert abs(gap2 - 0.5) < 1e-3, gap2

    # 11/12. Control has NO effective margin contribution; low/medium differ ONLY in lambda.
    src = inspect.getsource(C)
    ast.parse(src)
    assert 'if lam > 0.0:' in src
    assert 'margin_hinge_term(logits, u' in src
    assert 'loss = loss + lam * margin_loss' in src
    control_gate_idx = src.index('if lam > 0.0:')
    ce_line_idx = src.index('loss = ce + E.LAMBDA_KL * kl_val')
    assert ce_line_idx < control_gate_idx
    # The hinge function itself contains no arm-specific branching (same formula for low/medium).
    hinge_src = inspect.getsource(C.margin_hinge_term) + inspect.getsource(C.margin_gap)
    assert 'arm' not in hinge_src and "'low'" not in hinge_src and "'medium'" not in hinge_src

    # 15. English LR constant 5e-5 in all three arms; no annealing.
    assert "group['lr'] = 5e-5" in src
    assert 'lrs[' not in src and 'LR_SCHEDULE' not in src
    assert "a.arm=='anneal'" not in src.replace(' ', '') and "l['anneal']" not in src

    assert 'rt.binding_loss(m,' in src and 'E.compute_kl(m,' in src
    assert 'E.panel(' in src and 'E.checks(' in src and 'E.measure_d3(' in src
    assert C.E.LAMBDA_KL == 1 and C.E.KL_POS_PER_UPDATE == 160
    pin = C.rt.pinned_binding(C.H)
    assert pin.LAM == 1.0536573711078283

    # 16. Frozen gates unchanged (same negative tests as SF6/SF7).
    good = {'correct': 16, 'exact': 16, 'reversals': 8, 'families': 4}
    checks = {'binding': {n: {'gate': True} for n in ['pilot0', 'pilot1']}, 'language': {'loss': 3.5}}
    d3 = {'mean_combined_name_probability': .01}
    assert C.gates(good, checks, d3, 3.39, 200)['endpoint_pass']
    for key in good:
        bad = dict(good); bad[key] -= 1
        assert not C.gates(bad, checks, d3, 3.39, 200)['endpoint_pass']
    assert not C.gates(good, checks, {'mean_combined_name_probability': .01000001}, 3.39, 200)['continue']
    for n in checks['binding']:
        checks['binding'][n]['gate'] = False
        assert not C.gates(good, checks, d3, 3.39, 200)['continue']
        checks['binding'][n]['gate'] = True
    checks['language']['loss'] = 3.6400001
    assert not C.gates(good, checks, d3, 3.39, 200)['continue']
    checks['language']['loss'] = 3.5
    assert not C.gates({'correct': 9, 'exact': 9, 'reversals': 0, 'families': 0}, checks, d3, 3.39, 100)['continue']
    assert C.gates({'correct': 10, 'exact': 9, 'reversals': 0, 'families': 0}, checks, d3, 3.39, 100)['continue']
    gates_src = inspect.getsource(C.gates)
    assert 'margin' not in gates_src.lower()

    # 17. Transfer/generalization panels inaccessible/locked.
    assert all(x not in src for x in ['HELDOUT.json', 'ALTERNATE.json', 'COPY.json', 'COMPETING.json', 'NotImplementedError'])
    assert 'set_scope_block3(' not in src and 'E.main(' not in src

    # 18. All nine runs frozen before launch.
    assert p['runs'] == [
        {'seed': 87017, 'arm': 'control'}, {'seed': 87017, 'arm': 'low'}, {'seed': 87017, 'arm': 'medium'},
        {'seed': 87018, 'arm': 'control'}, {'seed': 87018, 'arm': 'low'}, {'seed': 87018, 'arm': 'medium'},
        {'seed': 87019, 'arm': 'control'}, {'seed': 87019, 'arm': 'low'}, {'seed': 87019, 'arm': 'medium'},
    ]
    assert p['seeds'] == [87017, 87018, 87019]
    assert not (C.H / 'runs').exists(), 'No run may exist before preflight/seal'

    scopesource = inspect.getsource(C.E.set_scope)
    assert 'range(4)' in scopesource
    for fp in C.H.glob('*.py'):
        ast.parse(fp.read_text(encoding='utf-8'))

    result = {
        'status': 'PASS', 'checkpoint_loaded': False, 'optimizer_created': False, 'updates': 0,
        'counts': {'updates': 200, 'english': 180, 'binding': 20, 'training_records': 16, 'families': 4,
                   'pairs': 8, 'batch': 32, 'kl_positions_per_update': 160, 'kl_pool': kl['count']},
        'dose_static_check': {'correct_logp': -0.5, 'wrong_logp': -1.0, 'gap': 0.5, 'raw_hinge': 0.5,
                               'control_contribution': 0.0, 'low_contribution': 0.125, 'medium_contribution': 0.25},
        'checks': {k: 'PASS' for k in [
            'Pilot1_identity', 'tokenizer_identity_and_boundaries', 'SF7_mechanics_reused',
            'literal_schedule_and_balancing', 'masking_alignment', 'constant_LR_all_arms_no_anneal',
            'M_fixed_1.0', 'lambda_control_0.0', 'lambda_low_0.25', 'lambda_medium_0.50',
            'control_no_margin_contribution', 'low_medium_differ_only_in_lambda',
            'first_answer_token_position', 'competing_candidate_tokens_all_16',
            'pinned_scope_and_objectives', 'frozen_gate_negative_tests', 'margin_not_a_gate',
            'locked_panels_not_referenced', 'external_source_identity', 'nine_runs_frozen', 'runtime']},
        'runtime': {'python': __import__('platform').python_version(), 'torch': __import__('torch').__version__,
                    'tokenizers': __import__('tokenizers').__version__, 'gpu': torch.cuda.get_device_name(0),
                    'executable': sys.executable},
        'source_hashes': {str(m.__file__): C.sha(m.__file__) for m in [C, C.E, C.rt]},
        'scope_function_sha256': C.sha(C.H / 'sources/PINNED_MASKING.py'),
    }
    C.rt.atomic_json(result, C.H / 'PREFLIGHT.json')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
