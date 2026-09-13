"""SF15: sealed SF14 machinery with first-answer-token causal CE weight restored to 0.25.

The sole scientific change from SF14 is the first answer-name CE weight. All later CE,
margin, broad KL, curriculum, optimizer, scope, gates and persistence remain unchanged."""
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
LAMBDA_MARGIN = 0.25


def margin_gap(logits_row, correct_tok, wrong_tok):
    logp = logits_row.log_softmax(-1)
    return logp[correct_tok] - logp[wrong_tok]


def margin_hinge_term(logits, ids_in_batch, idx):
    gaps = []
    for k, rid in enumerate(ids_in_batch):
        r = idx[rid]
        prefix_len = 1 + len(r['prompt_token_ids'])
        ci = r['correct_index']
        correct_tok = r['candidate_token_ids'][ci][0]
        wrong_tok = r['candidate_token_ids'][1 - ci][0]
        gaps.append(torch.clamp(MARGIN_M - margin_gap(logits[k, prefix_len - 1], correct_tok, wrong_tok), min=0.0))
    return torch.stack(gaps).mean()



NAME_TOKEN_IDS = (314, 536, 925, 512)
FIRST_ANSWER_CE_WEIGHT = 0.25


def english_ce_weights(labels, ids_in_batch, idx, first_weight=FIRST_ANSWER_CE_WEIGHT):
    """Return frozen per-label CE weights; ignored labels remain zero."""
    assert 0.0 <= float(first_weight) <= 1.0
    weights = (labels != -100).to(dtype=torch.float32)
    for k, rid in enumerate(ids_in_batch):
        record = idx[rid]
        pos = len(record['prompt_token_ids'])
        expected = record['candidate_token_ids'][record['correct_index']][0]
        assert expected in NAME_TOKEN_IDS
        assert int(labels[k, pos]) == expected
        weights[k, pos] = float(first_weight)
    return weights


def weighted_english_ce(logits, labels, ids_in_batch, idx, first_weight=FIRST_ANSWER_CE_WEIGHT):
    """Weighted mean causal CE; weight 0 reproduces SF14 and weight 1 reproduces SF13."""
    weights = english_ce_weights(labels, ids_in_batch, idx, first_weight).to(device=logits.device, dtype=logits.dtype)
    raw = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
        ignore_index=-100, reduction='none'
    ).reshape_as(labels)
    denominator = weights.sum()
    assert float(denominator.detach()) > 0.0
    return (raw * weights).sum() / denominator, {
        'first_answer_positions': len(ids_in_batch),
        'later_response_eos_positions': int((labels != -100).sum()) - len(ids_in_batch),
        'ce_weight_sum': float(denominator.detach()),
    }


def verify(sealed=True):
    if sealed:
        assert sha(H / 'FREEZE_RECEIPT.json') == (H / 'FREEZE_RECEIPT.sha256').read_text().split()[0]
        receipt = read(H / 'FREEZE_RECEIPT.json')
        assert receipt['status'] == 'SF15_PROSPECTIVE_PREFLIGHT_PASS'
        assert sha(H / 'SHA256SUMS.txt') == receipt['manifest_sha256']
        for line in (H / 'SHA256SUMS.txt').read_text().splitlines():
            h, n = line.split('  ', 1)
            assert sha(H / n) == h, n
    p = read(H / 'PROTOCOL.json')
    for path, h in read(H / 'EXTERNAL_INPUTS.json').items():
        assert sha(path) == h, path
    assert platform.python_version() == p['runtime']['python']
    assert torch.__version__ == p['runtime']['torch']
    assert tokenizers.__version__ == p['runtime']['tokenizers']
    assert Path(sys.executable).resolve() == Path(p['runtime']['executable']).resolve()
    assert torch.cuda.is_available()
    return p


def load_inputs():
    schedule = read(H / 'SCHEDULE.json')
    items = read(H / 'TRAIN.json'); idx = {r['id']: r for r in items}
    train16 = read(H / 'TRAIN16_RETENTION.json')
    dev_surface = read(H / 'DEV_SURFACE.json')
    dev_order = read(H / 'DEV_ORDER.json')
    pool = read(H / 'data/binding_rehearsal.json')['quartets']
    kl, oldproto = E.load_kl_pool()
    assert len(items) == len(idx) == 48
    assert len(train16) == 16 and len(dev_surface) == 16 and len(dev_order) == 16
    assert len(pool) == 80 and sum(len(q['docs']) for q in pool) == 320
    ei = 0
    exposure = Counter()
    for n, u in enumerate(schedule, 1):
        assert u['update'] == n
        assert u['kind'] == ('binding' if n % 10 == 0 else 'english')
        if u['kind'] == 'english':
            ei += 1
            assert len(u['ids']) == 36 and len(set(u['ids'])) == 36
            exposure.update(u['ids'])
            x, y = E.pad_batch([idx[i] for i in u['ids']], u['pad'])
            assert x.shape == y.shape == (36, u['pad']) and u['pad'] <= 256
            assert int((y != -100).sum()) == 36 * 5
            assert len(E.pick_kl_entries(kl['entries'], ei)) == 160
        else:
            docs = rt.binding_docs_for_batch(pool, u['quartets'], u['documents'])
            assert len(docs) == 32 and len(set(d['doc_id'] for d in docs)) == 32
    assert ei == 180
    assert len(exposure) == 48
    assert all(exposure[r['id']] == (45 if r['id'].startswith('TRAIN:') else 180) for r in items)
    return schedule, items, idx, train16, dev_surface, dev_order, pool, kl, oldproto


def retention_acq(acq):
    return acq['correct'] == 16 and acq['exact'] == 16 and acq['reversals'] == 8 and acq['families'] == 4


def dev_surface_gate(agg):
    return agg['correct'] >= 15 and agg['exact'] >= 13 and all(agg['subgroups'][s]['exact'] >= 3 for s in ['cloze','active_qa','passive_cloze','passive_qa'])


def dev_order_gate(agg):
    sg = agg['subgroups']
    fact = sg.get('order0:fact', {}).get('exact', 0) + sg.get('order1:fact', {}).get('exact', 0)
    copy = sg.get('order0:copy', {}).get('exact', 0) + sg.get('order1:copy', {}).get('exact', 0)
    return agg['exact'] >= 13 and fact >= 7 and copy >= 7 and all(sg[o+':fact']['exact']+sg[o+':copy']['exact']>=7 for o in ['order0','order1']) and all(sg[o+':'+q]['exact']>=3 for o in ['order0','order1'] for q in ['fact','copy'])


def gates(ret, dsurf, dord, checks, d3, base_loss, u):
    b = {n: bool(v['gate']) for n, v in checks['binding'].items()}
    lang = checks['language']['loss'] <= base_loss + .25
    pollution = d3['mean_combined_name_probability'] <= .01
    ret_ok = retention_acq(ret)
    continue_ = all(b.values()) and lang and pollution and ret_ok
    endpoint_pass = u == 200 and continue_ and dev_surface_gate(dsurf) and dev_order_gate(dord)
    return {'retention_acquisition': ret_ok, 'binding': b, 'language': lang, 'd3': pollution,
            'dev_surface': dev_surface_gate(dsurf), 'dev_order': dev_order_gate(dord),
            'continue': continue_, 'endpoint_pass': endpoint_pass}


def run_eval(m, train16, dev_surface, dev_order, tok, device, out, u):
    ret = E.panel(m, H, out, f'update{u}_train16', train16, tok, device)
    dsurf = E.panel(m, H, out, f'update{u}_devsurface', dev_surface, tok, device)
    dord = E.panel(m, H, out, f'update{u}_devorder', dev_order, tok, device)
    ch = out / f'update{u}_checks.json'
    if not ch.exists(): rt.atomic_json(E.checks(m, H, device), ch)
    rawpath = out / f'd3_update{u}_RAW.json'
    if not rawpath.exists():
        rows = E.measure_d3(m, read(H / 'D3_SELECTION.json'), tok, device)
        rt.atomic_json(rows, rawpath)
    raw = read(rawpath)
    assert len(raw) == 256
    d3 = E.d3_summary(raw)
    rt.atomic_json(d3, out / f'd3_update{u}_summary.json')
    return ret, dsurf, dord, read(ch), d3


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
    ap.add_argument('--arm', choices=['curriculum'], required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--mode', choices=['preload', 'train'], required=True)
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args()
    p = verify()
    run = next(r for r in p['runs'] if r['seed'] == a.seed)
    assert run['arm'] == a.arm
    assert os.environ.get('PYTHONHASHSEED') == str(a.seed)
    schedule, items, idx, train16, dev_surface, dev_order, pool, kl, oldproto = load_inputs()
    rt.configure_runtime(a.seed)
    if a.mode == 'preload':
        print('SF15_PRE_PARENT_LOAD_PASS', a.seed, flush=True); return
    out = H / 'runs' / f'seed_{a.seed}_curriculum'
    out.mkdir(parents=True, exist_ok=True)
    provenance = {'parent_checkpoint': run['parent_checkpoint'], 'parent_checkpoint_sha256': run['parent_checkpoint_sha256'],
                  'protocol': sha(H / 'PROTOCOL.json'), 'receipt': sha(H / 'FREEZE_RECEIPT.json'),
                  'schedule': sha(H / 'SCHEDULE.json'), 'curriculum': sha(H / 'TRAIN.json'),
                  'controller': sha(Path(__file__)), 'seed': a.seed, 'arm': a.arm,
                  'kl_pool': sha(H / 'KL_POOL.json'), 'margin_M': MARGIN_M, 'lambda_margin': LAMBDA_MARGIN,
                  'first_answer_token_ce_weight': FIRST_ANSWER_CE_WEIGHT, 'later_response_ce_weight': 1.0}
    pp = out / 'PROVENANCE.json'
    if pp.exists(): assert read(pp) == provenance
    else: rt.atomic_json(provenance, pp)
    if (out / 'STATUS.json').exists():
        raise RuntimeError('Run already classified; do not replay or extend it')
    restart = out / 'restart.pt'
    assert restart.exists() == a.resume, 'Explicit resume required iff committed restart exists'
    device = torch.device('cuda')
    m = rt.load_model(Path(p['parent_anchor']), device, H)
    tok = Tokenizer.from_file(p['tokenizer'])
    teacher = E.build_teacher(Path(p['parent_anchor']), device)
    E.assert_teacher_disjoint(m.base_model, teacher)
    ck = torch.load(run['parent_checkpoint'], map_location=device, weights_only=False)
    assert sha(run['parent_checkpoint']) == run['parent_checkpoint_sha256']
    m.load_state_dict(ck['model_state_dict'], strict=True)
    rt.atomic_json(E.scope_audit(m), out / 'PARAMETER_SCOPE.json')
    completed = 0; state = None
    if a.resume:
        state = torch.load(restart, map_location=device, weights_only=False)
        assert state['provenance'] == provenance
        completed = state['completed']; assert 0 <= completed <= 200
        m.load_state_dict(state['model'], strict=True)
        if state.get('last_metric'): append_metric(out, state['last_metric'])
    if completed == 0:
        ret, dsurf, dord, c, d3 = run_eval(m, train16, dev_surface, dev_order, tok, device, out, 0)
        # Parent-specific U0 reproduction: each SF13 branch must reproduce its OWN SF8 parent.
        assert retention_acq(ret), ret
        assert all(v['gate'] for v in c['binding'].values())
        d3_mass = d3['mean_combined_name_probability']
        ref = run['d3_reference']
        assert abs(d3_mass - ref) < 1e-4, f'D3 {d3_mass} != parent reference {ref}'
        assert d3_mass <= .01
        rt.atomic_json({'status': 'PASS', 'seed': a.seed, 'arm': a.arm,
                        'parent_sha256': run['parent_checkpoint_sha256'],
                        'd3_reference': ref, 'd3_measured': d3_mass,
                        'train16_acquired': retention_acq(ret), 'language': c['language'], 'd3': d3},
                       out / 'BASELINE_REPRODUCTION.json')
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
        ret, dsurf, dord, ch, d3 = run_eval(m, train16, dev_surface, dev_order, tok, device, out, u)
        g = gates(ret, dsurf, dord, ch, d3, base_loss, u)
        g['development_gain'] = dsurf['exact'] >= read(out/'update0_devsurface_RESULT.json')['exact']+4 and dord['exact'] >= read(out/'update0_devorder_RESULT.json')['exact']+4
        g['endpoint_pass'] = g['endpoint_pass'] and g['development_gain']
        rt.atomic_json(g, out / f'update{u}_GATES.json')
        print('EVALUATED', a.seed, u, json.dumps({'train16': {'correct': ret['correct'], 'exact': ret['exact'], 'reversals': ret['reversals'], 'families': ret['families']},
            'dev_surface': {'correct': dsurf['correct'], 'exact': dsurf['exact']},
            'dev_order': {'exact': dord['exact']}, 'gates': g, 'language': ch['language'], 'd3': d3}), flush=True)
        if not g['continue'] or u == 200:
            status = ('ACQUISITION_SUCCESS' if g['endpoint_pass'] else 'ACQUISITION_FAIL') if g['continue'] else 'STOP_REGRESSION'
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
        for group in opt.param_groups: group['lr'] = 5e-5
        ce = kl_val = margin_loss = None
        if u['kind'] == 'english':
            english_index += 1
            m.eval()
            kl_val = E.compute_kl(m, teacher, device, kl['rows'], E.pick_kl_entries(kl['entries'], english_index))
            m.train()
            batch = [idx[i] for i in u['ids']]
            x, y = E.pad_batch(batch, u['pad'])
            assert int((y != -100).sum()) == len(u['ids']) * 5
            x = x.to(device); y = y.to(device)
            logits = m.base_model(x)
            ce, ce_meta = weighted_english_ce(logits, y, u['ids'], idx)
            assert ce_meta['first_answer_positions'] == len(u['ids'])
            assert ce_meta['later_response_eos_positions'] == len(u['ids']) * 4
            assert abs(ce_meta['ce_weight_sum'] - len(u['ids']) * 4.25) < 1e-5
            loss = ce + E.LAMBDA_KL * kl_val
            margin_loss = margin_hinge_term(logits, u['ids'], idx)
            loss = loss + LAMBDA_MARGIN * margin_loss
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
        if ce is not None: rec.update(ce=float(ce.detach()), kl=float(kl_val.detach()), shard=u.get('shard'), first_answer_ce_weight=FIRST_ANSWER_CE_WEIGHT, later_response_ce_weight=1.0, ce_weight_sum=ce_meta['ce_weight_sum'], ce_supervised_positions=len(u['ids'])*5)
        if margin_loss is not None: rec.update(margin_loss=float(margin_loss.detach()))
        commit(number, rec); append_metric(out, rec)
        if number % 25 == 0: print('COMMITTED', a.seed, number, flush=True)
        if number in [100, 200] and not endpoint(number): return


if __name__ == '__main__': main()
