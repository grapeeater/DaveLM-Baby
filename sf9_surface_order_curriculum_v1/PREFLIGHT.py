"""Minimal SF9 preflight. Verifies parent checkpoint identities, curriculum integrity/disjointness,
and the frozen treatment (lambda=.25, M=1) + retention/gates. No checkpoint loading into an optimizer,
no training.
"""
import ast, json, inspect, os, sys
from pathlib import Path
import torch
import CONTROLLER as C


def main():
    p = C.verify(sealed=False)
    # 1/2 parent anchor + tokenizer
    assert p['parent_sha256'] == '2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb'
    assert p['tokenizer_sha256'] == 'e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b'
    # parent checkpoints present and hash-verified
    for r in p['runs']:
        assert C.sha(r['parent_checkpoint']) == r['parent_checkpoint_sha256'], r['seed']

    schedule, items, idx, train16, dev_surface, dev_order, pool, kl, old = C.load_inputs()

    # curriculum disjointness from frozen panels
    frozen = []
    for label in ['HELDOUT', 'ALTERNATE', 'COPY', 'COMPETING']:
        frozen += C.read(Path(r"C:\DaveLM-CADAVER\single_fact_acquisition_sf1_seed87011") / f"{label}.json")
    frozen_ids = {it['id'] for it in frozen}
    for it in items:
        assert it['id'] not in frozen_ids
    for it in dev_surface + dev_order:
        assert it['id'] not in frozen_ids and it['id'] not in {x['id'] for x in items}

    # 9/10 margin dose
    assert C.MARGIN_M == 1.0 and C.LAMBDA_MARGIN == 0.25
    src = inspect.getsource(C)
    ast.parse(src)
    assert 'loss = loss + LAMBDA_MARGIN * margin_loss' in src
    assert 'loss = ce + E.LAMBDA_KL * kl_val' in src
    assert "group['lr'] = 5e-5" in src and 'lrs[' not in src and 'LR_SCHEDULE' not in src
    assert C.E.LAMBDA_KL == 1 and C.E.KL_POS_PER_UPDATE == 160
    assert 'rt.binding_loss(m,' in src and 'E.compute_kl(m,' in src
    pin = C.rt.pinned_binding(C.H)
    assert pin.LAM == 1.0536573711078283
    assert 'margin_hinge_term(logits, u' in src

    # candidate name-token mapping valid for all curriculum items
    for it in items + dev_surface + dev_order:
        assert len(it['candidate_token_ids']) == 2
        assert it['candidate_token_ids'][0][0] != it['candidate_token_ids'][1][0]
        assert len(it['candidate_token_ids'][it['correct_index']]) == 4

    # gates negative tests (prospective thresholds)
    good_ret = {'correct': 16, 'exact': 16, 'reversals': 8, 'families': 4}
    good_surf = {'correct': 15, 'exact': 13}
    good_ord = {'exact': 13, 'subgroups': {'order0:fact': {'exact': 4}, 'order1:fact': {'exact': 4}, 'order0:copy': {'exact': 4}, 'order1:copy': {'exact': 4}}}
    checks = {'binding': {n: {'gate': True} for n in ['pilot0', 'pilot1']}, 'language': {'loss': 3.5}}
    d3 = {'mean_combined_name_probability': .01}
    assert C.gates(good_ret, good_surf, good_ord, checks, d3, 3.39, 200)['endpoint_pass']
    assert C.gates(good_ret, {'correct': 15, 'exact': 12}, good_ord, checks, d3, 3.39, 200)['endpoint_pass'] is False
    bad_ord = {'exact': 13, 'subgroups': {'order0:fact': {'exact': 3}, 'order1:fact': {'exact': 3}, 'order0:copy': {'exact': 4}, 'order1:copy': {'exact': 4}}}
    assert not C.gates(good_ret, good_surf, bad_ord, checks, d3, 3.39, 200)['endpoint_pass']
    bad_ret = {'correct': 15, 'exact': 15, 'reversals': 8, 'families': 4}
    assert not C.gates(bad_ret, good_surf, good_ord, checks, d3, 3.39, 200)['continue']
    assert not C.gates(good_ret, good_surf, good_ord, checks, {'mean_combined_name_probability': .0100001}, 3.39, 200)['continue']
    assert not C.gates(good_ret, good_surf, good_ord, checks, d3, 3.39, 100)['endpoint_pass']  # not endpoint at 100

    # 17 transfer/final locks
    assert all(x not in src for x in ['HELDOUT.json', 'ALTERNATE.json', 'COPY.json', 'COMPETING.json', 'NotImplementedError'])
    assert 'set_scope_block3(' not in src and 'E.main(' not in src

    # 18 runs frozen
    assert [r['seed'] for r in p['runs']] == [87020, 87021, 87022]
    assert p['seeds'] == [87020, 87021, 87022]
    assert not (C.H / 'runs').exists(), 'No run may exist before preflight/seal'

    for fp in C.H.glob('*.py'):
        ast.parse(fp.read_text(encoding='utf-8'))

    result = {
        'status': 'PASS', 'checkpoint_loaded': False, 'optimizer_created': False, 'updates': 0,
        'counts': {'updates': 200, 'english': 180, 'binding': 20, 'curriculum_items': len(items),
                   'train16': len(train16), 'dev_surface': len(dev_surface), 'dev_order': len(dev_order),
                   'kl_positions_per_update': 160, 'kl_pool': kl['count']},
        'checks': {k: 'PASS' for k in [
            'parent_checkpoint_identity', 'tokenizer_identity', 'curriculum_disjoint_from_frozen_panels',
            'dev_panels_disjoint', 'margin_M_1.0', 'lambda_margin_0.25', 'margin_only_first_answer_token',
            'candidate_token_mapping', 'constant_LR', 'KL_binding_scope_unchanged', 'frozen_gate_negative_tests',
            'locked_panels_not_referenced', 'runtime', 'runs_frozen']},
        'runtime': {'python': __import__('platform').python_version(), 'torch': __import__('torch').__version__,
                    'tokenizers': __import__('tokenizers').__version__, 'gpu': torch.cuda.get_device_name(0),
                    'executable': sys.executable},
        'source_hashes': {str(m.__file__): C.sha(m.__file__) for m in [C, C.E, C.rt]},
    }
    C.rt.atomic_json(result, C.H / 'PREFLIGHT.json')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
