"""Frozen descriptive SF9 reporter. No inference beyond the preregistered classifications."""
import json
import CONTROLLER as C


def main():
    p = C.verify()
    results = {}
    for r in p['runs']:
        key = f"seed_{r['seed']}_curriculum"; d = C.H / 'runs' / key
        status = C.read(d / 'STATUS.json'); u = status['completed']
        assert C.sha(status['checkpoint']) == status['checkpoint_sha256']
        traj = {}
        for t in [0, 100, 200]:
            if (d / f'update{t}_train16_RESULT.json').exists():
                traj[str(t)] = {
                    'train16': C.read(d / f'update{t}_train16_RESULT.json'),
                    'dev_surface': C.read(d / f'update{t}_devsurface_RESULT.json'),
                    'dev_order': C.read(d / f'update{t}_devorder_RESULT.json'),
                    'checks': C.read(d / f'update{t}_checks.json'),
                    'd3': C.read(d / f'd3_update{t}_summary.json'),
                    'gates': C.read(d / f'update{t}_GATES.json'),
                }
        results[key] = {'seed': r['seed'], 'parent_checkpoint_sha256': r['parent_checkpoint_sha256'],
                        'status': status, 'trajectory': traj}

    endpoint_passes = sum(1 for x in results.values() if x['status']['gates']['endpoint_pass'])
    early = any(x['status']['completed'] < 200 for x in results.values())
    if early:
        classification = 'INCONCLUSIVE_EARLY_STOP'
    elif any(x['status']['status'] == 'STOP_REGRESSION' for x in results.values()):
        classification = 'RETENTION_REGRESSION'
    elif endpoint_passes >= 2:
        classification = 'CURRICULUM_WIDENING_SUCCESS'
    elif endpoint_passes == 1:
        classification = 'PARTIAL_WIDENING'
    else:
        classification = 'NO_WIDENING'

    out = {'classification': classification, 'endpoint_passes': endpoint_passes, 'results': results,
           'transfer': 'LOCKED_UNSCORED', 'final_sacred_accessed': False}
    C.rt.atomic_json(out, C.H / 'RESULTS.json')

    lines = ['# SF9 — surface/order curriculum widening', '', f'**{classification}**. Three independent parents (SF8 lambda=0.25 checkpoints).', '',
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

    lines += ['', '## Dev surface subgroup (update 200 or last)', '']
    for key, x in results.items():
        z = x['trajectory'][str(x['status']['completed'])]
        ds = z['dev_surface']
        sg = "  ".join(f"{k}:{v['exact']}/{v['n']}" for k, v in ds['subgroups'].items())
        lines.append(f"- {key}: correct={ds['correct']}/16 exact={ds['exact']}/16  [{sg}]")

    lines += ['', '## Dev order subgroup (update 200 or last)', '']
    for key, x in results.items():
        z = x['trajectory'][str(x['status']['completed'])]
        do = z['dev_order']
        sg = "  ".join(f"{k}:{v['exact']}/{v['n']}" for k, v in do['subgroups'].items())
        lines.append(f"- {key}: exact={do['exact']}/16  [{sg}]")

    lines += ['', 'Frozen transfer panels (HELDOUT/ALTERNATE/COPY/COMPETING) remained LOCKED during training; the single preregistered post-treatment re-evaluation is a separate read-only step.',
              '', 'Dave-coded: this run teaches the same "name the doer" rule through more ways of asking and through mixed-up orderings, to see if Baby stops leaning on the last name it heard.', '',
              'Next action: review this completed study before the preregistered frozen-panel re-evaluation.']
    (C.H / 'FINAL_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')

    files = [C.H / 'RESULTS.json', C.H / 'FINAL_REPORT.md']
    files += sorted(x for x in (C.H / 'runs').rglob('*') if x.is_file() and x.name != 'restart.pt')
    files += sorted(x for x in (C.H / 'logs').glob('*.log') if x.is_file())
    receipt = ''.join(f'{C.sha(x)}  {x.relative_to(C.H).as_posix()}\n' for x in files)
    (C.H / 'OUTPUT_SHA256SUMS.txt').write_text(receipt, encoding='utf-8', newline='\n')
    for line in receipt.splitlines():
        h, n = line.split('  ', 1); assert C.sha(C.H / n) == h
    print('STUDY_COMPLETE', classification, 'OUTPUT_MANIFEST', C.sha(C.H / 'OUTPUT_SHA256SUMS.txt'), flush=True)


if __name__ == '__main__':
    main()
