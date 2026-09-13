"""Pre-seal smoke test for REPORTER.py. Uses a synthetic fixture (no real training, no GPU) to
verify (1) load_trajectory() tolerates a missing update0_GATES.json (the exact SF9 post-hoc bug),
and a partial (u=0,100 only) trajectory, and (2) classify() produces the expected label for each
scripted scenario. Run and must PASS before SEAL.py. Not part of the sealed scientific pipeline.
"""
import json, sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import REPORTER as R


def make_result(correct, exact, reversals, families, subgroups=None):
    return {'correct': correct, 'exact': exact, 'reversals': reversals, 'families': families,
            'subgroups': subgroups or {}}


def test_load_trajectory_missing_update0_gates():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        # update0: no GATES.json (matches real controller behavior)
        (d / 'update0_train16_RESULT.json').write_text(json.dumps(make_result(16, 16, 8, 4)))
        (d / 'update0_devsurface_RESULT.json').write_text(json.dumps(make_result(11, 3, 3, 3)))
        (d / 'update0_devorder_RESULT.json').write_text(json.dumps(make_result(7, 2, 2, 2, {'order0:fact': {'exact': 2, 'n': 4}})))
        (d / 'update0_checks.json').write_text(json.dumps({'binding': {'pilot0': {'gate': True}, 'pilot1': {'gate': True}}, 'language': {'loss': 3.58}}))
        (d / 'd3_update0_summary.json').write_text(json.dumps({'mean_combined_name_probability': 0.007}))
        # update100: HAS GATES.json, d3 failed (simulating a stop)
        (d / 'update100_train16_RESULT.json').write_text(json.dumps(make_result(16, 16, 8, 4)))
        (d / 'update100_devsurface_RESULT.json').write_text(json.dumps(make_result(11, 8, 4, 4)))
        (d / 'update100_devorder_RESULT.json').write_text(json.dumps(make_result(10, 10, 4, 4,
            {'order0:fact': {'exact': 4, 'n': 4}, 'order1:fact': {'exact': 3, 'n': 4}, 'order0:copy': {'exact': 2, 'n': 4}, 'order1:copy': {'exact': 1, 'n': 4}})))
        (d / 'update100_checks.json').write_text(json.dumps({'binding': {'pilot0': {'gate': True}, 'pilot1': {'gate': True}}, 'language': {'loss': 3.70}}))
        (d / 'd3_update100_summary.json').write_text(json.dumps({'mean_combined_name_probability': 0.02}))
        (d / 'update100_GATES.json').write_text(json.dumps({'retention_acquisition': True, 'binding': {'pilot0': True, 'pilot1': True}, 'language': True, 'd3': False, 'continue': False, 'endpoint_pass': False}))
        traj = R.load_trajectory(d)
        assert set(traj.keys()) == {'0', '100'}, traj.keys()
        assert traj['0']['gates'] is None, 'update0 must tolerate missing GATES.json'
        assert traj['100']['gates']['d3'] is False
    print('test_load_trajectory_missing_update0_gates: PASS')


def test_classify_density_hypothesis_weakened():
    results = {f'seed_{s}_curriculum': {'status': {'completed': 100, 'status': 'STOP_REGRESSION',
        'gates': {'endpoint_pass': False, 'd3': False}}} for s in [1, 2, 3]}
    label, n = R.classify(results)
    assert label == 'DENSITY_HYPOTHESIS_WEAKENED', label
    print('test_classify_density_hypothesis_weakened: PASS')


def test_classify_full_endpoint_success():
    results = {f'seed_{s}_curriculum': {'status': {'completed': 200, 'status': 'ACQUISITION_SUCCESS',
        'gates': {'endpoint_pass': True, 'd3': True}}} for s in [1, 2]}
    results['seed_3_curriculum'] = {'status': {'completed': 200, 'status': 'ACQUISITION_FAIL', 'gates': {'endpoint_pass': False, 'd3': True}}}
    label, n = R.classify(results)
    assert label == 'DENSITY_HYPOTHESIS_SUPPORTED_FULL_ENDPOINT', label
    assert n == 2
    print('test_classify_full_endpoint_success: PASS')


def test_classify_retention_regression_non_d3():
    results = {'seed_1_curriculum': {'status': {'completed': 100, 'status': 'STOP_REGRESSION', 'gates': {'endpoint_pass': False, 'd3': True}}},
               'seed_2_curriculum': {'status': {'completed': 200, 'status': 'ACQUISITION_FAIL', 'gates': {'endpoint_pass': False, 'd3': True}}},
               'seed_3_curriculum': {'status': {'completed': 200, 'status': 'ACQUISITION_FAIL', 'gates': {'endpoint_pass': False, 'd3': True}}}}
    label, n = R.classify(results)
    assert label == 'RETENTION_REGRESSION', label
    print('test_classify_retention_regression_non_d3: PASS')


if __name__ == '__main__':
    test_load_trajectory_missing_update0_gates()
    test_classify_density_hypothesis_weakened()
    test_classify_full_endpoint_success()
    test_classify_retention_regression_non_d3()
    print('ALL_SMOKE_TESTS_PASS')
