"""Frozen descriptive paired analysis for SF7. No inference or further treatment selection.
Adds descriptive-only first-answer-token gap mechanism reporting, derived from the already-
frozen per-item token_scores_including_eos (position 0 of each candidate = first-answer-token
log-probability). Margin/gap is never used to gate or reclassify a run.
"""
import json, math
from pathlib import Path
import CONTROLLER as C

G0_A0 = ['TRAIN:g0:found:small drum:a0', 'TRAIN:g0:found:wooden boat:a0', 'TRAIN:g0:carried:small drum:a0', 'TRAIN:g0:carried:wooden boat:a0']
G0_A1 = ['TRAIN:g0:found:small drum:a1', 'TRAIN:g0:found:wooden boat:a1', 'TRAIN:g0:carried:small drum:a1', 'TRAIN:g0:carried:wooden boat:a1']
RESIDUAL = 'TRAIN:g0:carried:wooden boat:a0'


def mean(v):
    return sum(v) / len(v)


def first_token_gaps(path):
    """gap = logp(correct first token) - logp(wrong first token), read from token_scores_including_eos[*][0]."""
    rows = [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines()]
    out = {}
    for r in rows:
        ci = r['correct_index']
        wrong_i = 1 - ci
        gap = r['token_scores_including_eos'][ci][0] - r['token_scores_including_eos'][wrong_i][0]
        out[r['id']] = gap
    return out


def mechanism_summary(run_dir, u):
    p = run_dir / f'update{u}_acquisition_RAW.jsonl'
    if not p.exists():
        return None
    gaps = first_token_gaps(p)
    g0a0 = {k: gaps[k] for k in G0_A0}
    g0a1 = [gaps[k] for k in G0_A1]
    g1 = [v for k, v in gaps.items() if k not in G0_A0 and k not in G0_A1]
    return {
        'g0_a0_gaps': g0a0,
        'residual_wooden_boat_carried_a0_gap': gaps[RESIDUAL],
        'mean_g0_a0_gap': mean(list(g0a0.values())),
        'min_g0_a0_gap': min(g0a0.values()),
        'mean_g0_a1_gap': mean(g0a1),
        'mean_g1_gap': mean(g1),
    }


def main():
    p = C.verify(); results = {}; pairs = []
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

    for seed in p['seeds']:
        c = results[f'seed_{seed}_control']; t = results[f'seed_{seed}_treatment']
        cp = c['status']['gates']['endpoint_pass']; tp = t['status']['gates']['endpoint_pass']
        outcome = 'both' if cp and tp else 'treatment_only' if tp else 'control_only' if cp else 'neither'
        q = {'seed': seed, 'control_pass': cp, 'treatment_pass': tp, 'outcome': outcome}
        for u in [100, 200]:
            cm = c['trajectory'].get(str(u), {}).get('mechanism')
            tm = t['trajectory'].get(str(u), {}).get('mechanism')
            if cm and tm:
                q.setdefault('mechanism_comparison', {})[str(u)] = {
                    'control_residual_gap': cm['residual_wooden_boat_carried_a0_gap'],
                    'treatment_residual_gap': tm['residual_wooden_boat_carried_a0_gap'],
                    'control_min_g0_a0': cm['min_g0_a0_gap'], 'treatment_min_g0_a0': tm['min_g0_a0_gap'],
                    'control_mean_g0_a0': cm['mean_g0_a0_gap'], 'treatment_mean_g0_a0': tm['mean_g0_a0_gap'],
                    'control_mean_g0_a1': cm['mean_g0_a1_gap'], 'treatment_mean_g0_a1': tm['mean_g0_a1_gap'],
                    'control_mean_g1': cm['mean_g1_gap'], 'treatment_mean_g1': tm['mean_g1_gap'],
                }
        pairs.append(q)

    treatment_only = sum(x['outcome'] == 'treatment_only' for x in pairs)
    control_only = sum(x['outcome'] == 'control_only' for x in pairs)
    control_successes = sum(x['control_pass'] for x in pairs)
    treatment_successes = sum(x['treatment_pass'] for x in pairs)
    def retention_ok(g):
        return all(g['binding'].values()) and g['language'] and g['d3']

    treatment_retention_failed = any(
        (not retention_ok(results[f"seed_{q['seed']}_treatment"]['status']['gates'])) and
        retention_ok(results[f"seed_{q['seed']}_control"]['status']['gates'])
        for q in pairs
    )
    if any(x['status']['completed'] < 200 for x in results.values()):
        classification = 'INCONCLUSIVE_EARLY_STOP'
    elif treatment_retention_failed:
        classification = 'TREATMENT_FAILED_RETENTION'
    elif treatment_only >= 2 and control_only == 0:
        classification = 'TREATMENT_FAVORED'
    elif control_only and treatment_only == 0:
        classification = 'CONTROL_FAVORED'
    elif control_only and treatment_only:
        classification = 'MIXED'
    else:
        classification = 'NO_CLEAR_DIFFERENCE'

    out = {'classification': classification, 'control_successes': control_successes,
           'treatment_successes': treatment_successes, 'treatment_only': treatment_only,
           'control_only': control_only, 'pairs': pairs, 'runs': results,
           'transfer': 'LOCKED_UNSCORED', 'final_sacred_accessed': False, 'no_followup_experiment': True}
    C.rt.atomic_json(out, C.H / 'RESULTS.json')

    lines = ['# SF7 — pairwise first-answer-token margin hinge', '',
             f'**{classification}**. Three preregistered same-seed control/treatment pairs; six independent Pilot1 starts.', '',
             'The only training-variable difference is one added loss term (margin hinge, M=1.0 nat, lambda_margin=1.0) on the first-answer-token pairwise log-prob gap in the treatment arm. English LR is constant 5e-5 in BOTH arms (no annealing). Binding, data, ordering, KL, scope and evaluation are unchanged from SF2/SF6. Historical SF1-SF6 classifications remain unchanged.', '',
             '| Seed | Arm | Update | Correct /16 | Exact /16 | Reversals /8 | Families /4 | CE | PPL | D3 | All gates |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in p['runs']:
        x = results[f"seed_{r['seed']}_{r['arm']}"]; u = x['status']['completed']; z = x['trajectory'][str(u)]
        a = z['acquisition']; l = z['checks']['language']
        lines.append(f"| {r['seed']} | {r['arm']} | {u} | {a['correct']} | {a['exact']} | {a['reversals']} | {a['families']} | {l['loss']:.6f} | {l['perplexity']:.3f} | {z['d3']['mean_combined_name_probability']:.8f} | {x['status']['status']} |")

    lines += ['', '## Paired evidence', '',
              f'Control successes: **{control_successes}/3**. Treatment successes: **{treatment_successes}/3**. Treatment-only: **{treatment_only}/3**. Control-only: **{control_only}/3**.', '']
    for q in pairs:
        lines.append(f"- Seed {q['seed']}: {q['outcome']}. Mechanism comparison: `{json.dumps(q.get('mechanism_comparison', {}))}`")

    lines += ['', '## Mechanism (descriptive only; never a gate)', '']
    for key, x in results.items():
        z200 = x['trajectory'].get('200', {}).get('mechanism')
        z100 = x['trajectory'].get('100', {}).get('mechanism')
        if z200:
            lines.append(f"- {key} @200: residual(wooden boat/Alex) gap={z200['residual_wooden_boat_carried_a0_gap']:.4f}, min g0:a0={z200['min_g0_a0_gap']:.4f}, mean g0:a0={z200['mean_g0_a0_gap']:.4f}, mean g0:a1={z200['mean_g0_a1_gap']:.4f}, mean g1={z200['mean_g1_gap']:.4f}")
        if z100:
            lines.append(f"  @100: residual gap={z100['residual_wooden_boat_carried_a0_gap']:.4f}, min g0:a0={z100['min_g0_a0_gap']:.4f}")

    lines += ['', 'Transfer/copy/competing-name/FINAL/sacred panels remain locked and unscored.', '',
              'Dave-coded: this run tests whether pushing the Alex-vs-Owen decision apart during training keeps the wooden-boat item off the knife edge, without touching anything else Baby learned.', '',
              'Next action: review this completed study before any transfer evaluation or further treatment. No second experiment was started.']
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
