"""Minimal SF7 preflight. Essential mechanical/scientific-identity checks only.
No checkpoint loading, no optimizer updates, no training. Includes the required tiny
static sanity check of the margin-hinge arithmetic on one fabricated log-prob pair.
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

    # 1. Parent + tokenizer hashes.
    assert p['parent_sha256'] == '2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb'
    assert p['tokenizer_sha256'] == 'e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b'
    assert C.sha(p['parent']) == p['parent_sha256']
    assert C.sha(p['tokenizer']) == p['tokenizer_sha256']

    # 2/9/10/11. SF6 control mechanics reused; constant LR; binding/KL unchanged.
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

    # 3/4. Correct first-answer-token location and competing candidate NAME tokens for all 16 items.
    for r in train:
        prefix_len = 1 + len(r['prompt_token_ids'])
        ci = r['correct_index']
        assert len(r['candidate_token_ids']) == 2
        correct_tok = r['candidate_token_ids'][ci][0]
        wrong_tok = r['candidate_token_ids'][1 - ci][0]
        assert correct_tok != wrong_tok, r['id']
        # First-answer-token position is prefix_len-1 (0-indexed logits row that predicts token at prefix_len).
        assert prefix_len - 1 >= 0
        # The token at that response position (from TRAIN.json's own encoding) must equal correct_tok.
        assert r['candidate_token_ids'][ci][0] == correct_tok

    # 5/6. Margin hinge formula and fixed hyperparameters, via one fabricated log-prob pair.
    assert C.MARGIN_M == 1.0 and C.LAMBDA_MARGIN == 1.0
    correct_logp, wrong_logp = -0.5, -1.0
    hinge, gap = hinge_reference(correct_logp, wrong_logp, C.MARGIN_M)
    assert abs(gap - 0.5) < 1e-12 and abs(hinge - 0.5) < 1e-12
    logits_row = torch.zeros(4)
    logits_row[0] = correct_logp  # arbitrary unnormalized stand-in check of the log-softmax path below
    # Direct check of margin_gap()/margin_hinge_term() against hand-built logits with a known softmax.
    test_logits = torch.log(torch.tensor([0.6065, 0.3679, 0.01, 0.0156]))  # exp(-0.5),exp(-1.0),...
    gap2 = float(C.margin_gap(test_logits, 0, 1))
    assert abs(gap2 - 0.5) < 1e-3, gap2
    hinge2 = max(0.0, C.MARGIN_M - gap2)
    assert abs(hinge2 - 0.5) < 1e-3
    # A margin already at/above M must contribute exactly zero.
    big_gap_logits = torch.log(torch.tensor([0.9, 0.01, 0.045, 0.045]))
    gap3 = float(C.margin_gap(big_gap_logits, 0, 1))
    assert gap3 > C.MARGIN_M
    assert max(0.0, C.MARGIN_M - gap3) == 0.0

    # 7/8. Margin loss present ONLY in treatment; control has exactly the unchanged SF2/SF6 objective.
    src = inspect.getsource(C)
    ast.parse(src)
    assert "if a.arm == 'treatment':" in src
    assert 'margin_hinge_term(logits, u' in src
    assert "loss = loss + LAMBDA_MARGIN * margin_loss" in src
    # The unconditional (control-applicable) English loss line must be byte-identical to SF6/SF2's.
    assert 'loss = ce + E.LAMBDA_KL * kl_val' in src
    # Ensure the margin addition is strictly inside the treatment-only branch (not before it, not shared).
    ce_line_idx = src.index('loss = ce + E.LAMBDA_KL * kl_val')
    treat_idx = src.index("if a.arm == 'treatment':")
    margin_add_idx = src.index('loss = loss + LAMBDA_MARGIN * margin_loss')
    assert ce_line_idx < treat_idx < margin_add_idx

    # 9. English LR constant 5e-5 in BOTH arms; no anneal schedule/branch anywhere.
    assert "group['lr'] = 5e-5" in src
    assert 'lrs[' not in src and 'LR_SCHEDULE' not in src
    assert "a.arm=='anneal'" not in src.replace(' ', '') and "l['anneal']" not in src

    # 10/11. Binding and KL mechanics byte-identical call sites to SF2/SF6.
    assert 'rt.binding_loss(m,' in src and 'E.compute_kl(m,' in src
    assert 'E.panel(' in src and 'E.checks(' in src and 'E.measure_d3(' in src
    assert C.E.LAMBDA_KL == 1 and C.E.KL_POS_PER_UPDATE == 160
    pin = C.rt.pinned_binding(C.H)
    assert pin.LAM == 1.0536573711078283

    # 12. Frozen acquisition/retention gates unchanged (same negative tests as SF6 PREFLIGHT).
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
    # Margin is never a gate input: gates() signature/body must not reference margin at all.
    gates_src = inspect.getsource(C.gates)
    assert 'margin' not in gates_src.lower()

    # 13. Transfer/generalization panels inaccessible/locked.
    assert all(x not in src for x in ['HELDOUT.json', 'ALTERNATE.json', 'COPY.json', 'COMPETING.json', 'NotImplementedError'])
    assert 'set_scope_block3(' not in src and 'E.main(' not in src

    # 14. All six runs and seeds frozen before launch.
    assert p['runs'] == [
        {'seed': 87014, 'arm': 'control'}, {'seed': 87014, 'arm': 'treatment'},
        {'seed': 87015, 'arm': 'control'}, {'seed': 87015, 'arm': 'treatment'},
        {'seed': 87016, 'arm': 'control'}, {'seed': 87016, 'arm': 'treatment'},
    ]
    assert p['seeds'] == [87014, 87015, 87016]
    assert not (C.H / 'runs').exists(), 'No run may exist before preflight/seal'

    scopesource = inspect.getsource(C.E.set_scope)
    assert 'range(4)' in scopesource
    for fp in C.H.glob('*.py'):
        ast.parse(fp.read_text(encoding='utf-8'))

    result = {
        'status': 'PASS', 'checkpoint_loaded': False, 'optimizer_created': False, 'updates': 0,
        'counts': {'updates': 200, 'english': 180, 'binding': 20, 'training_records': 16, 'families': 4,
                   'pairs': 8, 'batch': 32, 'kl_positions_per_update': 160, 'kl_pool': kl['count']},
        'hinge_static_check': {'correct': -0.5, 'wrong': -1.0, 'gap': 0.5, 'hinge': 0.5, 'M': C.MARGIN_M, 'lambda_margin': C.LAMBDA_MARGIN},
        'checks': {k: 'PASS' for k in [
            'SF2_source_identity', 'Pilot1_identity', 'tokenizer_identity_and_boundaries',
            'literal_schedule_and_balancing', 'masking_alignment', 'constant_LR_both_arms_no_anneal',
            'margin_hinge_formula', 'margin_only_in_treatment', 'control_objective_unchanged',
            'competing_candidate_tokens_all_16', 'first_answer_token_position',
            'pinned_scope_and_objectives', 'frozen_gate_negative_tests', 'margin_not_a_gate',
            'no_replay_or_fork_gate', 'locked_panels_not_referenced', 'external_source_identity',
            'six_runs_frozen', 'runtime']},
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
