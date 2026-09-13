"""Frozen descriptive SF10 reporter. Fixed vs SF9's post-hoc bug: does NOT require
update0_GATES.json (the controller only writes GATES at 100/200, never at the update0 baseline).
Handles partial trajectories (a run that stopped at 100) and full trajectories (100+200) alike.
Smoke-tested against a synthetic no-training fixture BEFORE this file is sealed (see
REPORTER_SMOKE_TEST.py). No inference beyond the preregistered classifications.
"""
import json
import CONTROLLER as C


def load_trajectory(d):
    traj = {}
    for t in [0, 100, 200]:
        if not (d / f'update{t}_train16_RESULT.json').exists():
            continue
        gates_path = d / f'update{t}_GATES.json'
        traj[str(t)] = {
            'train16': C.read(d / f'update{t}_train16_RESULT.json'),
            'dev_surface': C.read(d / f'update{t}_devsurface_RESULT.json'),
            'dev_order': C.read(d / f'update{t}_devorder_RESULT.json'),
            'checks': C.read(d / f'update{t}_checks.json'),
            'd3': C.read(d / f'd3_update{t}_summary.json'),
            'gates': C.read(gates_path) if gates_path.exists() else None,
        }
    return traj


def classify(results):
    """Pure function of {key: {'status': STATUS.json dict, ...}} -> (classification, endpoint_passes).
    Kept standalone (no verify()/filesystem dependency) so it can be smoke-tested pre-seal.
    """
    endpoint_passes = sum(1 for x in results.values() if x['status']['gates']['endpoint_pass'])
    early = any(x['status']['completed'] < 200 and x['status']['status'] not in ('STOP_REGRESSION',) for x in results.values())
    if early:
        return 'INCONCLUSIVE_EARLY_STOP', endpoint_passes
    d3_stops = sum(1 for x in results.values() if x['status']['status'] == 'STOP_REGRESSION' and not x['status']['gates']['d3'])
    if d3_stops >= 2:
        return 'DENSITY_HYPOTHESIS_WEAKENED', endpoint_passes
    if d3_stops == 1:
        return 'MIXED', endpoint_passes
    if any(x['status']['status'] == 'STOP_REGRESSION' for x in results.values()):
        return 'RETENTION_REGRESSION', endpoint_passes
    if endpoint_passes >= 2:
        return 'DENSITY_HYPOTHESIS_SUPPORTED_FULL_ENDPOINT', endpoint_passes
    if endpoint_passes == 1:
        return 'PARTIAL_WIDENING', endpoint_passes
    return 'DENSITY_HYPOTHESIS_UNRESOLVED_WIDENING_STALLED', endpoint_passes


def main():
    p = C.verify()
    results = {}
    for r in p['runs']:
        key = f"seed_{r['seed']}_curriculum"; d = C.H / 'runs' / key
        status = C.read(d / 'STATUS.json'); u = status['completed']
        assert C.sha(status['checkpoint']) == status['checkpoint_sha256']
        results[key] = {'seed': r['seed'], 'parent_checkpoint_sha256': r['parent_checkpoint_sha256'],
                        'status': status, 'trajectory': load_trajectory(d)}

    classification, endpoint_passes = classify(results)
    out = {'classification': classification, 'endpoint_passes': endpoint_passes, 'results': results,
           'transfer': 'LOCKED_UNSCORED', 'final_sacred_accessed': False}
    C.rt.atomic_json(out, C.H / 'RESULTS.json')

    lines = ['# SF10 — reduced-density curriculum widening', '', f'**{classification}**.', '',
             '| Seed | Parent (SF8) | Endpoint | Train16 c/x/rev/fam | DevSurface c/x | DevOrder x (fact/copy) | CE | D3 | Status |',
             '|---|---|---:|---|---|---|---|---:|---:|---|']
    for key, x in results.items():
        st = x['status']; u = st['completed']; z = x['trajectory'][str(u)]
        a = z['train16']; ds = z['dev_surface']; do = z['dev_order']
        fsub = do['subgroups'].get('order0:fact', {}).get('exact', 0) + do['subgroups'].get('order1:fact', {}).get('exact', 0)
        csub = do['subgroups'].get('order0:copy', {}).get('exact', 0) + do['subgroups'].get('order1:copy', {}).get('exact', 0)
        lines.append(f"| {x['seed']} | {x['parent_checkpoint_sha256'][:12]}… | {u} | {a['correct']}/{a['exact']}/{a['reversals']}/{a['families']} | {ds['correct']}/{ds['exact']} | {do['exact']} ({fsub}/{csub}) | {z['checks']['language']['loss']:.4f} | {z['d3']['mean_combined_name_probability']:.6f} | {st['status']} |")

    lines += ['', '## Per-seed endpoint gates', '']
    for key, x in results.items():
        u = x['status']['completed']
        lines.append(f"- {key} @{u}: `{json.dumps(x['status']['gates'])}`")

    lines += ['', 'Frozen SF1 transfer panels remained LOCKED. SF9 remains sealed and unmodified.']
    (C.H / 'FINAL_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')

    files = [C.H / 'RESULTS.json', C.H / 'FINAL_REPORT.md']
    files += sorted(x for x in (C.H / 'runs').rglob('*') if x.is_file() and x.name != 'restart.pt')
    files += sorted(x for x in (C.H / 'logs').glob('*.log') if x.is_file()) if (C.H / 'logs').exists() else []
    receipt = ''.join(f'{C.sha(x)}  {x.relative_to(C.H).as_posix()}\n' for x in files)
    (C.H / 'OUTPUT_SHA256SUMS.txt').write_text(receipt, encoding='utf-8', newline='\n')
    for line in receipt.splitlines():
        h, n = line.split('  ', 1); assert C.sha(C.H / n) == h
    print('STUDY_COMPLETE', classification, 'OUTPUT_MANIFEST', C.sha(C.H / 'OUTPUT_SHA256SUMS.txt'), flush=True)


if __name__ == '__main__':
    main()
