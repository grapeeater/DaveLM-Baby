"""One-time, outcome-blind SF7 materialization record. Verifies the already-copied frozen
inputs are byte-identical to the sealed SF6 study (which itself verified them against SF2/SF1
originals), and that PROTOCOL.json/PROVENANCE.json/EXTERNAL_INPUTS.json are internally
consistent. No checkpoint loading or training. Idempotent: safe to re-run before sealing.
"""
import hashlib, json
from pathlib import Path

H = Path(__file__).resolve().parent
ROOT = H.parent
SF6 = ROOT / 'sf6_same_seed_lr_anneal_v1'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def main():
    assert sha(SF6 / 'FREEZE_RECEIPT.json') == (SF6 / 'FREEZE_RECEIPT.sha256').read_text().split()[0]
    receipt = read(SF6 / 'FREEZE_RECEIPT.json')
    assert receipt['status'] == 'SF6_PROSPECTIVE_PREFLIGHT_PASS'
    assert sha(SF6 / 'SHA256SUMS.txt') == receipt['manifest_sha256']
    sf6_manifest = {n.replace('\\', '/'): h for h, n in (l.split('  ', 1) for l in (SF6 / 'SHA256SUMS.txt').read_text().splitlines() if l.strip())}
    prov = read(H / 'PROVENANCE.json')
    for r in prov['reused_byte_identical']:
        assert sha(H / r['destination']) == r['sha256'], r['destination']
        src_rel = Path(r['source']).name if '\\' not in r['destination'] else r['destination']
        assert sf6_manifest.get(r['destination']) == r['sha256'], (r['destination'], 'not found or mismatched in sealed SF6 manifest')
    p = read(H / 'PROTOCOL.json')
    external = read(H / 'EXTERNAL_INPUTS.json')
    for n, h in external.items():
        assert sha(n) == h, n
    assert external[p['parent']] == p['parent_sha256']
    assert external[p['tokenizer']] == p['tokenizer_sha256']
    assert len(p['runs']) == 6
    assert p['seeds'] == [87014, 87015, 87016]
    assert not (H / 'runs').exists(), 'No training may occur before freeze'
    print('MATERIALIZATION_VERIFIED', H)


if __name__ == '__main__':
    main()
