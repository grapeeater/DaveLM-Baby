"""Read-only reconstruction of the inherited selection evidence."""
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

def main():
    panels = json.loads((ROOT / 'data/generated/foundation_v2/panels.json').read_text())
    records = [json.loads(s) for s in (ROOT / 'runs/structured_v2r4_seed106001_from6000_terminal/metrics.jsonl').read_text().splitlines()]
    terminal = records[-1]
    probe = json.loads((ROOT / 'runs/v2r4_isolation_probe/PROBE.json').read_text())
    print('PROBE keys', list(probe))
    for file in ['QUERY_SWAP_ROWS.json', 'ISOLATION_ROWS.json']:
        obj = json.loads((ROOT / 'runs/v2r4_isolation_probe' / file).read_text())
        print(file, type(obj).__name__, list(obj)[:8] if isinstance(obj, dict) else obj[:1])
    for name in ['same_surface_novel', 'short_keyed', 'primitive_keyed']:
        rows = terminal['rows'][name]
        items = panels[name]
        print(name, 'n', len(items), 'K', dict(Counter(x['pair_count'] for x in items)), 'first', sum(x['target_rank']==1 for x in rows), 'rest_lock', sum(all(r==1 for r in x['all_ranks'][1:len(item['target_span'])]) for x,item in zip(rows,items)))
    for name in ['structured_v2r5_seed107001_from6000', 'structured_v2r6_seed108001_from6000', 'diagnostic_v2r5_low_prior_exposure_seed107001_u16000_16600']:
        path = ROOT / 'runs' / name / 'metrics.jsonl'
        if path.exists():
            last=json.loads(path.read_text().splitlines()[-1])
            print(name, last.get('update'), last.get('language_dev_ce'), json.dumps(last.get('summaries',{})))

if __name__ == '__main__':
    main()
