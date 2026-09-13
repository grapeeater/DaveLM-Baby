"""One-time, outcome-blind SF8 materialization record. Verifies the already-copied frozen
inputs are byte-identical to the sealed SF7 study (itself byte-identical to sealed SF6/SF2/SF1).
No checkpoint loading or training. Idempotent.
"""
import hashlib, json
from pathlib import Path

H = Path(__file__).resolve().parent
ROOT = H.parent
SF7 = ROOT / 'sf7_pairwise_margin_hinge_v1'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def main():
    assert sha(SF7 / 'FREEZE_RECEIPT.json') == (SF7 / 'FREEZE_RECEIPT.sha256').read_text().split()[0]
    receipt = read(SF7 / 'FREEZE_RECEIPT.json')
    assert receipt['status'] == 'SF7_PROSPECTIVE_PREFLIGHT_PASS'
    assert sha(SF7 / 'SHA256SUMS.txt') == receipt['manifest_sha256']
    sf7_manifest = {n.replace('\\', '/'): h for h, n in (l.split('  ', 1) for l in (SF7 / 'SHA256SUMS.txt').read_text().splitlines() if l.strip())}
    prov = read(H / 'PROVENANCE.json')
    for r in prov['reused_byte_identical']:
        assert sha(H / r['destination']) == r['sha256'], r['destination']
        assert sf7_manifest.get(r['destination']) == r['sha256'], (r['destination'], 'not found or mismatched in sealed SF7 manifest')
    p = read(H / 'PROTOCOL.json')
    external = read(H / 'EXTERNAL_INPUTS.json')
    for n, h in external.items():
        assert sha(n) == h, n
    assert external[p['parent']] == p['parent_sha256']
    assert external[p['tokenizer']] == p['tokenizer_sha256']
    assert len(p['runs']) == 9
    assert p['seeds'] == [87017, 87018, 87019]
    assert p['arms']['control']['lambda_margin'] == 0.0
    assert p['arms']['low']['lambda_margin'] == 0.25
    assert p['arms']['medium']['lambda_margin'] == 0.50
    assert p['margin_M'] == 1.0
    assert not (H / 'runs').exists(), 'No training may occur before freeze'
    print('MATERIALIZATION_VERIFIED', H)


if __name__ == '__main__':
    main()
