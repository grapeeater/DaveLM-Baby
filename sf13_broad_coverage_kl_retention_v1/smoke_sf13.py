# Engine-level SF13 selection smoke test: updates 1, 2, 100, 180 (no training)
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import SF2_ENGINE as E

H = Path(__file__).resolve().parent
pool, proto = E.load_kl_pool()
entries = pool['entries']
schedule = json.loads((H / 'SF13_KL_SCHEDULE.json').read_text(encoding='utf-8'))['schedule']

for u in [1, 2, 100, 180]:
    sel = E.pick_kl_entries(entries, u)
    assert len(sel) == 160, (u, len(sel))
    assert len(set(e[0] for e in sel)) == 160, u
    frozen = [tuple(e) for e in schedule[str(u)]]
    assert [tuple(e) for e in sel] == frozen, u
    assert all(tuple(e) in {(x[0], x[1]) for x in entries} for e in sel), u
    print(f'update {u}: 160 entries, 160 distinct rows, matches frozen schedule, all in KL_POOL [PASS]')

print('ENGINE-LEVEL SMOKE TEST PASS')
