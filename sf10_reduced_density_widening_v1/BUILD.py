"""One-time SF10 materialization record. Verifies copied frozen inputs are byte-identical to
sealed SF9 (itself SF8/SF6/SF2/SF1), curriculum manifest consistency, and pre-freeze invariants.
"""
import hashlib, json
from pathlib import Path

H = Path(__file__).resolve().parent
ROOT = H.parent
SF9 = ROOT / 'sf9_surface_order_curriculum_v1'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def main():
    manifest_sha = sha(SF9 / 'SHA256SUMS.txt')
    sf9_manifest = {n.replace('\\', '/'): h for h, n in (l.split('  ', 1) for l in (SF9 / 'SHA256SUMS.txt').read_text().splitlines() if l.strip())}
    prov = read(H / 'PROVENANCE.json')
    for r in prov['reused_byte_identical']:
        assert sha(H / r['destination']) == r['sha256'], r['destination']
        assert sf9_manifest.get(r['destination']) == r['sha256'], r['destination']
    cm = read(H / 'CURRICULUM_MANIFEST.json')
    for n, h in cm.items():
        assert sha(H / n) == h, n
    p = read(H / 'PROTOCOL.json')
    external = read(H / 'EXTERNAL_INPUTS.json')
    for n, h in external.items():
        assert sha(n) == h, n
    assert external[p['parent_anchor']] == p['parent_sha256']
    assert external[p['tokenizer']] == p['tokenizer_sha256']
    assert len(p['runs']) == 3 and p['seeds'] == [87023, 87024, 87025]
    assert p['margin']['lambda_margin'] == 0.25 and p['margin']['M'] == 1.0
    for r in p['runs']:
        assert 'sf8_margin_dose_comparison_v1' in r['parent_checkpoint']
    assert not (H / 'runs').exists(), 'No training may occur before freeze'
    print('MATERIALIZATION_VERIFIED', H)


if __name__ == '__main__':
    main()
