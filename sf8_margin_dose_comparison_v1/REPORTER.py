"""Frozen descriptive dose-comparison analysis for SF8. No inference or further treatment
selection. Mechanism (first-answer-token gap) is descriptive only, derived from the already-
frozen per-item token_scores_including_eos; never used to gate or reclassify a run.
"""
import json
from pathlib import Path
import CONTROLLER as C

G0_A0 = ['TRAIN:g0:found:small drum:a0', 'TRAIN:g0:found:wooden boat:a0', 'TRAIN:g0:carried:small drum:a0', 'TRAIN:g0:carried:wooden boat:a0']
G0_A1 = ['TRAIN:g0:found:small drum:a1', 'TRAIN:g0:found:wooden boat:a1', 'TRAIN:g0:carried:small drum:a1', 'TRAIN:g0:carried:wooden boat:a1']
RESIDUAL = 'TRAIN:g0:carried:wooden boat:a0'


def mean(v):
    return sum(v) / len(v)


def first_token_gaps(path):
    rows = [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines()]
    out = {}
    for r in rows:
        ci = r['correct_index']; wrong_i = 1 - ci
        out[r['id']] = r['token_scores_including_eos'][ci][0] - r['token_scores_including_eos'][wrong_i][0]
    return out


def mechanism_summary(run_dir, u):
    p = run_dir / f'update{u}_acquisition_RAW.jsonl'
    if not p.exists():
        return None
    gaps = first_token_gaps(p)
    g0a0 = {k: gaps[k] for k in G0_A0}
    g0a1 = [gaps[k] for k in G0_A1]
    g1 = [v for k, v in gaps.items() if k not in G0_A0 and k not in G0_A1]
    return {'g0_a0_gaps': g0a0, 'residual_wooden_boat_carried_a0_gap': gaps[RESIDUAL],
            'mean_g0_a0_gap': mean(list(g0a0.values())), 'min_g0_a0_gap': min(g0a0.values()),
            'mean_g0_a1_gap': mean(g0a1), 'mean_g1_gap': mean(g1)}


def main():
    p = C.verify(); results = {}
    for r in p['runs']:
        key = f"seed_{r['seed']}_{r['arm']}"; d = C.H / 'runs' / key
        status = C.read(d / 'STATUS.json'); u = status['completed']
        assert C.sha(status['checkpoint']) == status['checkpoint_sha256']
        trajectory = {}
        for t in [0, 100, 200]:
            if not (d / f'update{t}_acquisition_RESULT.json').exists(): continue
            trajectory[str(t)] = {'acquisition': C.read(d / f'update{t}_acquisition_RESULT.json'),
                'checks': C.read(d / f'update{t}_checks.json'), 'd3': C.read(d / f'd3_update{t}_summary.json'),
                'mechanism': mechanism_summary(d, t)}
        results[key] = {'seed': r['seed'], 'arm': r['arm'], 'status': status, 'trajectory': trajectory}

    def retention_ok(g):
        return all(g['binding'].values()) and g['language'] and g['d3']

    def classify_stop(status):
        if status['status'] == 'ACQUISITION_SUCCESS': return 'success'
        if status['status'] == 'ACQUISITION_FAIL': return 'acquisition_fail'
        if status['status'] == 'STOP_REGRESSION':
            g = status['gates']
            if not g['d3']: return 'd3_stop'
            return 'other_retention_stop'
        if status['status'] == 'STOP_ACQUISITION_GUARD': return 'acquisition_guard_stop'
        return 'other'

    dose_summary = {}
    for arm in ['control', 'low', 'medium']:
        outcomes = [classify_stop(results[f'seed_{s}_{arm}']['status']) for s in p['seeds']]
        dose_summary[arm] = {
            'full_gate_successes': sum(o == 'success' for o in outcomes),
            'acquisition_failures': sum(o == 'acquisition_fail' for o in outcomes),
            'd3_hard_stops': sum(o == 'd3_stop' for o in outcomes),
            'other_retention_failures': sum(o == 'other_retention_stop' or o == 'acquisition_guard_stop' for o in outcomes),
            'endpoint_availability_200': sum(results[f'seed_{s}_{arm}']['status']['completed'] == 200 for s in p['seeds']),
            'per_seed_outcome': dict(zip(p['seeds'], outcomes)),
        }

    triplets = []
    for seed in p['seeds']:
        row = {'seed': seed}
        for arm in ['control', 'low', 'medium']:
            row[arm] = classify_stop(results[f'seed_{seed}_{arm}']['status'])
        triplets.append(row)

    low_s = dose_summary['low']['full_gate_successes']
    med_s = dose_summary['medium']['full_gate_successes']
    ctrl_s = dose_summary['control']['full_gate_successes']
    low_bad = dose_summary['low']['d3_hard_stops'] + dose_summary['low']['other_retention_failures']
    med_bad = dose_summary['medium']['d3_hard_stops'] + dose_summary['medium']['other_retention_failures']
    any_early_stop_blocks_comparison = any(
        results[f'seed_{s}_{arm}']['status']['completed'] < 200 and classify_stop(results[f'seed_{s}_{arm}']['status']) not in ('d3_stop', 'other_retention_stop', 'acquisition_guard_stop')
        for s in p['seeds'] for arm in ['control', 'low', 'medium']
    )
    if any_early_stop_blocks_comparison:
        classification = 'INCONCLUSIVE_EARLY_STOP'
    elif low_bad > dose_summary['control']['d3_hard_stops'] + dose_summary['control']['other_retention_failures'] and low_s <= ctrl_s and med_bad and med_s <= ctrl_s:
        classification = 'TREATMENT_FAILED_RETENTION'
    elif low_s and med_s and low_bad == 0 and med_bad == 0:
        classification = 'BOTH_DOSES_SUPPORTED'
    elif low_s > ctrl_s and low_s >= med_s and low_bad == 0:
        classification = 'LOW_DOSE_FAVORED'
    elif med_s > ctrl_s and med_s > low_s and med_bad == 0:
        classification = 'MEDIUM_DOSE_FAVORED'
    elif low_s == 0 and med_s == 0 and ctrl_s == 0:
        classification = 'NO_CLEAR_DIFFERENCE'
    elif low_s <= ctrl_s and med_s <= ctrl_s:
        classification = 'CONTROL_FAVORED'
    else:
        classification = 'MIXED'

    out = {'classification': classification, 'dose_summary': dose_summary, 'triplets': triplets, 'runs': results,
           'transfer': 'LOCKED_UNSCORED', 'final_sacred_accessed': False, 'no_followup_experiment': True,
           'sf7_untouched': True}
    C.rt.atomic_json(out, C.H / 'RESULTS.json')

    lines = ['# SF8 — margin dose comparison (control=0.0, low=0.25, medium=0.50; M=1.0 fixed)', '',
             f'**{classification}**. Three fresh same-seed triplets (87017-87019); nine independent Pilot1 starts. SF7 (lambda=1.0) remains sealed and unmodified.', '',
             '| Seed | Arm | Update | Correct/16 | Exact/16 | Rev/8 | Fam/4 | CE | PPL | D3 | Status |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in p['runs']:
        x = results[f"seed_{r['seed']}_{r['arm']}"]; u = x['status']['completed']; z = x['trajectory'][str(u)]
        a = z['acquisition']; l = z['checks']['language']
        lines.append(f"| {r['seed']} | {r['arm']} | {u} | {a['correct']} | {a['exact']} | {a['reversals']} | {a['families']} | {l['loss']:.6f} | {l['perplexity']:.3f} | {z['d3']['mean_combined_name_probability']:.8f} | {x['status']['status']} |")

    lines += ['', '## Dose summary /3', '']
    for arm in ['control', 'low', 'medium']:
        s = dose_summary[arm]
        lines.append(f"- **{arm}**: successes={s['full_gate_successes']}/3, acquisition_fail={s['acquisition_failures']}/3, D3_stops={s['d3_hard_stops']}/3, other_retention={s['other_retention_failures']}/3, endpoint@200={s['endpoint_availability_200']}/3")

    lines += ['', '## Same-seed triplets', '']
    for t in triplets:
        lines.append(f"- {t['seed']}: control={t['control']}, low={t['low']}, medium={t['medium']}")

    lines += ['', '## Mechanism (descriptive only; never a gate)', '']
    for key, x in results.items():
        z200 = x['trajectory'].get('200', {}).get('mechanism')
        z100 = x['trajectory'].get('100', {}).get('mechanism')
        if z200:
            lines.append(f"- {key} @200: residual={z200['residual_wooden_boat_carried_a0_gap']:.4f}, min g0:a0={z200['min_g0_a0_gap']:.4f}, mean g0:a0={z200['mean_g0_a0_gap']:.4f}, mean g0:a1={z200['mean_g0_a1_gap']:.4f}, mean g1={z200['mean_g1_gap']:.4f}, D3={x['trajectory']['200']['d3']['mean_combined_name_probability']:.6f}")
        if z100:
            lines.append(f"  @100: residual={z100['residual_wooden_boat_carried_a0_gap']:.4f}, min g0:a0={z100['min_g0_a0_gap']:.4f}, D3={x['trajectory']['100']['d3']['mean_combined_name_probability']:.6f}")

    lines += ['', 'Transfer/copy/competing-name/FINAL/sacred panels remain locked and unscored. SF7 untouched.', '',
              'Next action: review this completed dose comparison before any transfer evaluation or further treatment. No tenth run, no dose changes, no interim selection occurred.']
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
