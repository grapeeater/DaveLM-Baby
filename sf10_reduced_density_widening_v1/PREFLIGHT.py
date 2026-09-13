"""Minimal SF10 preflight. Verifies parent identities, curriculum/shard integrity and balance,
disjointness from frozen panels, the frozen treatment (lambda=.25, M=1) + retention/gates, and
that the reporter has already been smoke-tested. No checkpoint loading into an optimizer.
"""
import ast, json, inspect, os, sys
from pathlib import Path
import torch
import CONTROLLER as C


def main():
    p = C.verify(sealed=False)
    assert p['parent_sha256'] == '2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb'
    assert p['tokenizer_sha256'] == 'e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b'
    for r in p['runs']:
        assert C.sha(r['parent_checkpoint']) == r['parent_checkpoint_sha256'], r['seed']
    # parents must be the SF8 low checkpoints, NOT the D3-polluted SF9 checkpoints
    for r in p['runs']:
        assert 'sf8_margin_dose_comparison_v1' in r['parent_checkpoint']
        assert 'sf9_surface_order_curriculum_v1' not in r['parent_checkpoint']
    assert p['seeds'] == [87023, 87024, 87025]
    assert set(p['seeds']).isdisjoint({87020, 87021, 87022}), 'must not reuse SF9 seeds'

    schedule, items, idx, train16, dev_surface, dev_order, pool, kl, old = C.load_inputs()

    # curriculum content identical to SF9 (except added 'shard' tag)
    sf9_items = {it['id']: it for it in json.loads(Path(r"C:\DaveLM-CADAVER\sf9_surface_order_curriculum_v1\TRAIN.json").read_text(encoding='utf-8-sig'))}
    assert set(idx) == set(sf9_items)
    for iid, it in idx.items():
        stripped = {k: v for k, v in it.items() if k != 'shard'}
        assert stripped == sf9_items[iid], iid

    # dev panels reused verbatim from SF9 (historical development rulers)
    sf9_ds = json.loads(Path(r"C:\DaveLM-CADAVER\sf9_surface_order_curriculum_v1\DEV_SURFACE.json").read_text(encoding='utf-8-sig'))
    sf9_do = json.loads(Path(r"C:\DaveLM-CADAVER\sf9_surface_order_curriculum_v1\DEV_ORDER.json").read_text(encoding='utf-8-sig'))
    assert dev_surface == sf9_ds and dev_order == sf9_do
    sf9_t16 = json.loads(Path(r"C:\DaveLM-CADAVER\sf8_margin_dose_comparison_v1\TRAIN.json").read_text(encoding='utf-8-sig'))
    assert train16 == sf9_t16

    # disjointness from frozen SF1 transfer panels
    frozen_ids = set()
    for label in ['HELDOUT', 'ALTERNATE', 'COPY', 'COMPETING']:
        frozen_ids |= {it['id'] for it in C.read(Path(r"C:\DaveLM-CADAVER\single_fact_acquisition_sf1_seed87011") / f"{label}.json")}
    assert frozen_ids.isdisjoint(idx.keys())

    # exposure/shard balance (structural, re-verified independently of CURRICULUM.py)
    from collections import Counter
    per_shard_count = Counter(it['shard'] for it in items)
    assert per_shard_count == Counter({0: 36, 1: 36, 2: 36, 3: 36})
    for s in range(4):
        names = Counter(it['candidates'][it['correct_index']] for it in items if it['shard'] == s)
        assert set(names.values()) == {9}, (s, names)
    exposure = Counter()
    for u in schedule:
        if u['kind'] == 'english':
            exposure.update(u['ids'])
    assert set(exposure.values()) == {45}
    assert sum(len(u['ids']) for u in schedule if u['kind'] == 'english') == 36 * 180 == 6480

    # margin dose unchanged
    assert C.MARGIN_M == 1.0 and C.LAMBDA_MARGIN == 0.25
    src = inspect.getsource(C)
    ast.parse(src)
    assert 'loss = loss + LAMBDA_MARGIN * margin_loss' in src
    assert 'loss = ce + E.LAMBDA_KL * kl_val' in src
    assert "group['lr'] = 5e-5" in src and 'lrs[' not in src and 'LR_SCHEDULE' not in src
    assert C.E.LAMBDA_KL == 1 and C.E.KL_POS_PER_UPDATE == 160
    pin = C.rt.pinned_binding(C.H)
    assert pin.LAM == 1.0536573711078283
    assert 'margin_hinge_term(logits, u' in src

    # gates identical to sealed SF9 protocol (re-confirmed byte-for-byte)
    sf9_gates = json.loads(Path(r"C:\DaveLM-CADAVER\sf9_surface_order_curriculum_v1\PROTOCOL.json").read_text(encoding='utf-8-sig'))['gates']
    assert p['gates']['retention_acquisition'] == sf9_gates['retention_acquisition']
    assert p['gates']['dev_surface'] == sf9_gates['dev_surface']
    assert p['gates']['dev_order'] == sf9_gates['dev_order']
    assert p['gates']['d3'] == sf9_gates['d3']
    assert p['gates']['language'] == sf9_gates['language']

    good_ret = {'correct': 16, 'exact': 16, 'reversals': 8, 'families': 4}
    good_surf = {'correct': 15, 'exact': 13}
    good_ord = {'exact': 13, 'subgroups': {'order0:fact': {'exact': 4}, 'order1:fact': {'exact': 4}, 'order0:copy': {'exact': 4}, 'order1:copy': {'exact': 4}}}
    checks = {'binding': {n: {'gate': True} for n in ['pilot0', 'pilot1']}, 'language': {'loss': 3.5}}
    d3 = {'mean_combined_name_probability': .01}
    assert C.gates(good_ret, good_surf, good_ord, checks, d3, 3.39, 200)['endpoint_pass']
    assert not C.gates(good_ret, {'correct': 15, 'exact': 12}, good_ord, checks, d3, 3.39, 200)['endpoint_pass']
    bad_ord = {'exact': 13, 'subgroups': {'order0:fact': {'exact': 3}, 'order1:fact': {'exact': 3}, 'order0:copy': {'exact': 4}, 'order1:copy': {'exact': 4}}}
    assert not C.gates(good_ret, good_surf, bad_ord, checks, d3, 3.39, 200)['endpoint_pass']
    bad_ret = {'correct': 15, 'exact': 15, 'reversals': 8, 'families': 4}
    assert not C.gates(bad_ret, good_surf, good_ord, checks, d3, 3.39, 200)['continue']
    assert not C.gates(good_ret, good_surf, good_ord, checks, {'mean_combined_name_probability': .0100001}, 3.39, 200)['continue']

    # transfer/final locks
    assert all(x not in src for x in ['HELDOUT.json', 'ALTERNATE.json', 'COPY.json', 'COMPETING.json', 'NotImplementedError'])
    assert 'set_scope_block3(' not in src and 'E.main(' not in src

    # reporter smoke test must already have been run and passed (mechanical hygiene)
    smoke_src = (C.H / 'REPORTER_SMOKE_TEST.py').read_text(encoding='utf-8')
    assert 'ALL_SMOKE_TESTS_PASS' in smoke_src or 'def test_' in smoke_src  # source exists and defines tests
    import subprocess
    smoke = subprocess.run([sys.executable, '-B', str(C.H / 'REPORTER_SMOKE_TEST.py')], cwd=C.H, capture_output=True, text=True)
    assert smoke.returncode == 0 and 'ALL_SMOKE_TESTS_PASS' in smoke.stdout, smoke.stdout + smoke.stderr

    assert not (C.H / 'runs').exists(), 'No run may exist before preflight/seal'
    for fp in C.H.glob('*.py'):
        ast.parse(fp.read_text(encoding='utf-8'))

    result = {
        'status': 'PASS', 'checkpoint_loaded': False, 'optimizer_created': False, 'updates': 0,
        'counts': {'updates': 200, 'english': 180, 'binding': 20, 'curriculum_items': len(items),
                   'per_update_batch': 36, 'train16': len(train16), 'dev_surface': len(dev_surface),
                   'dev_order': len(dev_order), 'kl_positions_per_update': 160, 'kl_pool': kl['count']},
        'checks': {k: 'PASS' for k in [
            'parent_checkpoint_identity', 'parents_are_sf8_not_sf9', 'tokenizer_identity',
            'curriculum_identical_to_sf9_content', 'dev_panels_reused_verbatim', 'train16_reused_verbatim',
            'curriculum_disjoint_from_frozen_panels', 'shard_balance_36_per_shard_9_per_name',
            'per_item_exposure_45', 'margin_M_1.0', 'lambda_margin_0.25', 'gates_match_sf9_protocol',
            'frozen_gate_negative_tests', 'locked_panels_not_referenced', 'reporter_smoke_test_pass',
            'runtime', 'runs_frozen']},
        'runtime': {'python': __import__('platform').python_version(), 'torch': __import__('torch').__version__,
                    'tokenizers': __import__('tokenizers').__version__, 'gpu': torch.cuda.get_device_name(0),
                    'executable': sys.executable},
        'source_hashes': {str(m.__file__): C.sha(m.__file__) for m in [C, C.E, C.rt]},
    }
    C.rt.atomic_json(result, C.H / 'PREFLIGHT.json')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
