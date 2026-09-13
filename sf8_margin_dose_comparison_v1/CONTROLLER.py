"""Independent full-trajectory SF8 runs. Pinned SF2/SF6/SF7 objectives/evaluators/schedule/
scope/optimizer/gates, unchanged. ONLY new element versus SF7: lambda_margin is a dose
(0.0 / 0.25 / 0.50) instead of the single SF7 value 1.0. M remains fixed at 1.0 nat. The
hinge formula, position (first answer name token only), and integration point are byte-
identical to SF7. Control (lambda=0.0) computes NO hinge term at all.
"""
import argparse, hashlib, json, math, os, platform, sys
from collections import Counter
from pathlib import Path
import torch
import torch.nn.functional as F
import tokenizers
from tokenizers import Tokenizer
import SF2_ENGINE as E

H = Path(__file__).resolve().parent
rt = E.rt
read = E.read
sha = E.sha

MARGIN_M = 1.0
LAMBDA_BY_ARM = {'control': 0.0, 'low': 0.25, 'medium': 0.50}


def margin_gap(logits_row, correct_tok, wrong_tok):
    """First-answer-token pairwise log-prob gap: logp_correct - logp_wrong. No full-string score."""
    logp = logits_row.log_softmax(-1)
    return logp[correct_tok] - logp[wrong_tok]


def margin_hinge_term(logits, ids_in_batch, idx):
    """Mean over the 32 factual sequences in this update of max(0, M - gap)."""
    gaps = []
    for k, rid in enumerate(ids_in_batch):
        r = idx[rid]
        prefix_len = 1 + len(r['prompt_token_ids'])
        ci = r['correct_index']
        correct_tok = r['candidate_token_ids'][ci][0]
        wrong_tok = r['candidate_token_ids'][1 - ci][0]
        gap = margin_gap(logits[k, prefix_len - 1], correct_tok, wrong_tok)
        gaps.append(torch.clamp(MARGIN_M - gap, min=0.0))
    return torch.stack(gaps).mean()


def verify(sealed=True):
    if sealed:
        assert sha(H / 'FREEZE_RECEIPT.json') == (H / 'FREEZE_RECEIPT.sha256').read_text().split()[0]
        receipt = read(H / 'FREEZE_RECEIPT.json')
        assert receipt['status'] == 'SF8_PROSPECTIVE_PREFLIGHT_PASS'
        assert sha(H / 'SHA256SUMS.txt') == receipt['manifest_sha256']
        for line in (H / 'SHA256SUMS.txt').read_text().splitlines():
            h, n = line.split('  ', 1)
            assert sha(H / n) == h, n
    p = read(H / 'PROTOCOL.json')
    for path, h in read(H / 'EXTERNAL_INPUTS.json').items():
        assert sha(path) == h, path
    for r in read(H / 'PROVENANCE.json')['reused_byte_identical']:
        assert sha(H / r['destination']) == r['sha256'], r['destination']
    assert platform.python_version() == p['runtime']['python']
    assert torch.__version__ == p['runtime']['torch']
    assert tokenizers.__version__ == p['runtime']['tokenizers']
    assert Path(sys.executable).resolve() == Path(p['runtime']['executable']).resolve()
    assert torch.cuda.is_available()
    return p


def load_inputs():
    schedule = read(H / 'SCHEDULE.json')
    train = read(H / 'TRAIN.json'); idx = {r['id']: r for r in train}
    pool = read(H / 'data/binding_rehearsal.json')['quartets']
    kl, oldproto = E.load_kl_pool()
    assert len(schedule) == 200 and len(train) == len(idx) == 16
    assert len(pool) == 80 and sum(len(q['docs']) for q in pool) == 320
    ei = 0
    for n, u in enumerate(schedule, 1):
        assert u['update'] == n
        assert u['kind'] == ('binding' if n % 10 == 0 else 'english')
        if u['kind'] == 'english':
            ei += 1
            assert Counter(u['ids']) == Counter({k: 2 for k in idx})
            x, y = E.pad_batch([idx[i] for i in u['ids']], u['pad'])
            assert x.shape == y.shape == (32, u['pad']) and u['pad'] <= 256
            assert int((y != -100).sum()) == 160
            for k, rid in enumerate(u['ids']):
                r = idx[rid]; prefix = [2] + r['prompt_token_ids']; c = r['candidate_token_ids'][r['correct_index']]
                assert len(c) == 4 and y[k, len(prefix) - 1:len(prefix) + 4].tolist() == c + [3]
                assert x[k, :len(prefix) + 4].tolist() == prefix + c
                assert (y[k, :len(prefix) - 1] == -100).all()
                assert (y[k, len(prefix) + 4:] == -100).all()
                assert len(r['candidate_token_ids']) == 2 and all(len(cc) == 4 for cc in r['candidate_token_ids'])
                assert r['candidate_token_ids'][0][0] != r['candidate_token_ids'][1][0]
            assert len(E.pick_kl_entries(kl['entries'], ei)) == 160
        else:
            docs = rt.binding_docs_for_batch(pool, u['quartets'], u['documents'])
            assert len(docs) == 32 and len(set(d['doc_id'] for d in docs)) == 32
    assert ei == 180
    return schedule, train, idx, pool, kl, oldproto


def gates(acq, checks, d3, base_loss, u):
    b = {n: bool(v['gate']) for n, v in checks['binding'].items()}
    lang = checks['language']['loss'] <= base_loss + .25
    pollution = d3['mean_combined_name_probability'] <= .01
    guard = not (u == 100 and acq['correct'] < 10 and acq['exact'] < 10)
    return {'acquisition': bool(E.acquire(acq)), 'binding': b, 'language': lang, 'd3': pollution,
            'update100_acquisition_guard': guard, 'continue': all(b.values()) and lang and pollution and guard,
            'endpoint_pass': u == 200 and E.acquire(acq) and all(b.values()) and lang and pollution}


def run_eval(m, train, tok, device, out, u):
    acq = E.panel(m, H, out, f'update{u}_acquisition', train, tok, device)
    ch = out / f'update{u}_checks.json'
    if not ch.exists(): rt.atomic_json(E.checks(m, H, device), ch)
    rawpath = out / f'd3_update{u}_RAW.json'
    if not rawpath.exists():
        rows = E.measure_d3(m, read(H / 'D3_SELECTION.json'), tok, device)
        rt.atomic_json(rows, rawpath)
    raw = read(rawpath)
    assert len(raw) == 256
    summary = E.d3_summary(raw)
    rt.atomic_json(summary, out / f'd3_update{u}_summary.json')
    return acq, read(ch), summary


def append_metric(out, rec):
    path = out / 'TRAIN_METRICS.jsonl'
    rows = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    assert len({r['update'] for r in rows}) == len(rows)
    if rows and rows[-1]['update'] == rec['update']:
        assert rows[-1] == rec
        return
    assert not rows or rows[-1]['update'] == rec['update'] - 1
    with path.open('a', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(rec, allow_nan=False) + '\n'); f.flush(); os.fsync(f.fileno())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['control', 'low', 'medium'], required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--mode', choices=['preload', 'train'], required=True)
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args()
    p = verify()
    assert {'seed': a.seed, 'arm': a.arm} in p['runs']
    lam = LAMBDA_BY_ARM[a.arm]
    assert os.environ.get('PYTHONHASHSEED') == str(a.seed)
    schedule, train, idx, pool, kl, oldproto = load_inputs()
    rt.configure_runtime(a.seed)
    if a.mode == 'preload':
        print('SF8_PRE_PARENT_LOAD_PASS', a.seed, a.arm, flush=True); return
    out = H / 'runs' / f'seed_{a.seed}_{a.arm}'
    out.mkdir(parents=True, exist_ok=True)
    provenance = {'parent': p['parent_sha256'], 'protocol': sha(H / 'PROTOCOL.json'),
                  'receipt': sha(H / 'FREEZE_RECEIPT.json'), 'schedule': sha(H / 'SCHEDULE.json'),
                  'controller': sha(Path(__file__)),
                  'seed': a.seed, 'arm': a.arm, 'kl_pool': sha(H / 'KL_POOL.json'),
                  'margin_M': MARGIN_M, 'lambda_margin': lam}
    pp = out / 'PROVENANCE.json'
    if pp.exists(): assert read(pp) == provenance
    else: rt.atomic_json(provenance, pp)
    if (out / 'STATUS.json').exists():
        raise RuntimeError('Run already classified; do not replay or extend it')
    restart = out / 'restart.pt'
    assert restart.exists() == a.resume, 'Explicit resume required iff committed restart exists'
    device = torch.device('cuda')
    m = rt.load_model(Path(p['parent']), device, H)
    rt.atomic_json(E.scope_audit(m), out / 'PARAMETER_SCOPE.json')
    tok = Tokenizer.from_file(p['tokenizer'])
    teacher = E.build_teacher(Path(p['parent']), device)
    E.assert_teacher_disjoint(m.base_model, teacher)
    completed = 0; state = None
    if a.resume:
        state = torch.load(restart, map_location=device, weights_only=False)
        assert state['provenance'] == provenance
        completed = state['completed']; assert 0 <= completed <= 200
        m.load_state_dict(state['model'], strict=True)
        if state.get('last_metric'): append_metric(out, state['last_metric'])
    if completed == 0:
        b, c, d = run_eval(m, train, tok, device, out, 0)
        assert b['correct'] == 9 and b['exact'] == 0, b
        assert all(v['gate'] for v in c['binding'].values())
        assert abs(c['language']['loss'] - 3.3907) / 3.3907 < 1e-4
        assert abs(d['mean_combined_name_probability'] - .000907) < .0001
        m.eval()
        with torch.no_grad():
            k0 = float(E.compute_kl(m, teacher, device, kl['rows'], E.pick_kl_entries(kl['entries'], 1)))
        assert abs(k0) < 1e-9, k0
        rt.atomic_json({'status': 'PASS', 'kl0': k0, 'baseline': b, 'language': c['language'], 'd3': d}, out / 'BASELINE_REPRODUCTION.json')
    base_loss = read(out / 'update0_checks.json')['language']['loss']
    opt = torch.optim.AdamW(m.parameters(), lr=5e-5, betas=(.9, .999), eps=1e-8, weight_decay=.05, amsgrad=False, foreach=False, fused=False)
    if state is not None:
        opt.load_state_dict(state['optimizer']); rt.restore_rng(state['rng'])
    pin = rt.pinned_binding(H)

    def commit(u, rec=None):
        rt.atomic_torch_save({'completed': u, 'model': m.state_dict(), 'optimizer': opt.state_dict(),
            'rng': rt.capture_rng(), 'scope': 'binding' if u % 10 == 0 else 'english',
            'provenance': provenance, 'last_metric': rec}, restart)

    def endpoint(u):
        cp = out / f'checkpoint_{u}.pt'
        if not cp.exists():
            rt.atomic_torch_save({'model_state_dict': m.state_dict(), 'update': u, 'provenance': provenance}, cp)
        hp = out / f'checkpoint_{u}.sha256'
        if hp.exists(): assert hp.read_text().split()[0] == sha(cp)
        else: hp.write_text(sha(cp) + '\n', encoding='utf-8')
        acq, ch, d3 = run_eval(m, train, tok, device, out, u)
        g = gates(acq, ch, d3, base_loss, u)
        rt.atomic_json(g, out / f'update{u}_GATES.json')
        print('EVALUATED', a.seed, a.arm, u, json.dumps({'acquisition': acq, 'gates': g, 'language': ch['language'], 'd3': d3}), flush=True)
        if not g['continue'] or u == 200:
            status = ('ACQUISITION_SUCCESS' if g['endpoint_pass'] else 'ACQUISITION_FAIL') if g['continue'] else ('STOP_ACQUISITION_GUARD' if not g['update100_acquisition_guard'] else 'STOP_REGRESSION')
            assert sha(p['parent']) == p['parent_sha256']
            verify()
            rt.atomic_json({'status': status, 'completed': u, 'gates': g, 'checkpoint': str(cp),
                'checkpoint_sha256': sha(cp), 'transfer': 'LOCKED_UNSCORED', 'final_accessed': False}, out / 'STATUS.json')
            return False
        return True

    if completed in [100, 200] and not endpoint(completed): return
    if completed == 0: commit(0)
    english_index = sum(u['kind'] == 'english' for u in schedule[:completed])
    for u in schedule[completed:]:
        opt.zero_grad(set_to_none=True)
        E.set_scope(m, u['kind'] == 'binding')
        inactive = {n: (x.detach().clone(), {k: v.clone() if torch.is_tensor(v) else v for k, v in opt.state.get(x, {}).items()}) for n, x in m.named_parameters() if not x.requires_grad}
        for group in opt.param_groups: group['lr'] = 5e-5  # constant in ALL arms; no annealing in SF8
        ce = kl_val = margin_loss = None
        if u['kind'] == 'english':
            english_index += 1
            m.eval()
            kl_val = E.compute_kl(m, teacher, device, kl['rows'], E.pick_kl_entries(kl['entries'], english_index))
            m.train()
            batch = [idx[i] for i in u['ids']]
            x, y = E.pad_batch(batch, u['pad'])
            x = x.to(device); y = y.to(device)
            logits = m.base_model(x)
            ce = F.cross_entropy(logits.reshape(-1, 1024), y.reshape(-1), ignore_index=-100)
            loss = ce + E.LAMBDA_KL * kl_val
            if lam > 0.0:
                margin_loss = margin_hinge_term(logits, u['ids'], idx)
                loss = loss + lam * margin_loss
        else:
            m.train()
            loss, _ = rt.binding_loss(m, rt.binding_docs_for_batch(pool, u['quartets'], u['documents']), device, pin)
        assert torch.isfinite(loss), 'nonfinite loss'
        loss.backward()
        assert all(x.grad is None for x in m.parameters() if not x.requires_grad)
        norm = torch.nn.utils.clip_grad_norm_([x for x in m.parameters() if x.grad is not None], 2.0)
        assert torch.isfinite(norm), 'nonfinite gradient norm'
        opt.step()
        for n, x in m.named_parameters():
            if n not in inactive: continue
            old, st = inactive[n]; assert torch.equal(x, old), n
            now = opt.state.get(x, {})
            assert now.keys() == st.keys()
            for k, v in st.items(): assert torch.equal(now[k], v) if torch.is_tensor(v) else now[k] == v
        number = u['update']
        rec = {'update': number, 'kind': u['kind'], 'lr': 5e-5, 'loss': float(loss.detach()), 'grad_norm': float(norm)}
        if ce is not None: rec.update(ce=float(ce.detach()), kl=float(kl_val.detach()))
        if margin_loss is not None: rec.update(margin_loss=float(margin_loss.detach()), lambda_margin=lam)
        commit(number, rec); append_metric(out, rec)
        if number % 25 == 0: print('COMMITTED', a.seed, a.arm, number, flush=True)
        if number in [100, 200] and not endpoint(number): return


if __name__ == '__main__': main()
