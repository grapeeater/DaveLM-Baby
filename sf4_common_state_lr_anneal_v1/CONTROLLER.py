"""SF4: prospective common-state study. Scientific operations are pinned SF2 functions."""
import argparse, json, os, platform, shutil, hashlib
from pathlib import Path
import torch
import torch.nn.functional as F
import tokenizers
import SF2_ENGINE as E

ROOT = Path(__file__).resolve().parent
read, sha, rt = E.read, E.sha, E.rt


def equality(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu())
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equality(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equality(x, y) for x, y in zip(a, b))
    return a == b


def state_digest(value):
    """Serialization-independent digest including dtype, shape, typed keys and RNG state."""
    h = hashlib.sha256()
    def visit(v):
        h.update(type(v).__name__.encode() + b'\0')
        if torch.is_tensor(v):
            h.update(str(v.dtype).encode()); h.update(str(tuple(v.shape)).encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=lambda k: (type(k).__name__, str(k))):
                visit(k); visit(v[k])
        elif isinstance(v, (list, tuple)):
            h.update(str(len(v)).encode())
            for x in v: visit(x)
        else:
            h.update(repr(v).encode())
    visit(value)
    return h.hexdigest()


def verify():
    receipt = ROOT / 'FREEZE_RECEIPT.json'
    assert sha(receipt) == (ROOT / 'FREEZE_RECEIPT.sha256').read_text(encoding='utf-8').split()[0]
    r = read(receipt)
    assert r['status'] == 'SF4_PROSPECTIVE_PROTOCOL_FROZEN'
    assert sha(ROOT / 'PRETRAIN_SHA256SUMS.txt') == r['manifest_sha256']
    for line in (ROOT / 'PRETRAIN_SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        h, n = line.split('  ', 1)
        assert sha(ROOT / n) == h, n
    p = read(ROOT / 'PROTOCOL.json')
    for path, h in read(ROOT / 'INPUT_HASHES.json').items():
        assert sha(path) == h, path
    b = Path(p['base_bundle'])
    base = E.verify(b)
    assert platform.python_version() == '3.12.14'
    assert torch.__version__ == '2.12.0+rocm7.14.0'
    assert tokenizers.__version__ == '0.23.1'
    assert os.environ.get('PYTHONHASHSEED') == '87011'
    rt.configure_runtime(87011)
    assert torch.are_deterministic_algorithms_enabled()
    pool, _ = E.load_kl_pool()
    schedule = read(b / 'SCHEDULE.json')
    lr_rows = read(ROOT / 'LR_SCHEDULE.json')
    assert len(schedule) == len(lr_rows) == 200
    rows = read(b / 'TRAIN.json'); idx = {r['id']: r for r in rows}
    binding = read(b / 'data/binding_rehearsal.json')['quartets']
    for u, l in zip(schedule, lr_rows):
        assert u['update'] == l['update'] and u['kind'] == l['kind']
        if u['kind'] == 'english':
            x, y = E.pad_batch([idx[i] for i in u['ids']], u['pad'])
            assert x.shape == y.shape == (32, u['pad']) and u['pad'] <= 256
        else:
            assert len(rt.binding_docs_for_batch(binding, u['quartets'], u['documents'])) == 32
    return p, b, base, pool, schedule, lr_rows, rows, binding


def provenance():
    return {'protocol': sha(ROOT / 'PROTOCOL.json'), 'manifest': sha(ROOT / 'PRETRAIN_SHA256SUMS.txt'),
            'parent': read(ROOT / 'PROTOCOL.json')['parent_sha256'],
            'schedule': sha(Path(read(ROOT / 'PROTOCOL.json')['base_bundle']) / 'SCHEDULE.json'),
            'lr_schedule': sha(ROOT / 'LR_SCHEDULE.json')}


def check_branch_receipt():
    rpath = ROOT / 'BRANCH_RECEIPT.json'
    assert sha(rpath) == (ROOT / 'BRANCH_RECEIPT.sha256').read_text(encoding='utf-8').split()[0]
    r = read(rpath)
    assert r['status'] == 'COMMON_STATE_EXACT_AND_BRANCHES_FROZEN'
    assert r['provenance'] == provenance()
    for name, h in r['files'].items():
        assert sha(ROOT / name) == h, name
    return r


def eval_at(m, b, out, u, rows, tok, dev, base_loss=None, full=True):
    rng = rt.capture_rng()
    agg = E.panel(m, b, out, f'update{u}_acquisition', rows, tok, dev)
    if full:
        qpath = out / f'update{u}_checks.json'
        if not qpath.exists(): rt.atomic_json(E.checks(m, b, dev), qpath)
        q = read(qpath)
        dpath = out / f'd3_update{u}_summary.json'
        if not dpath.exists():
            raw = E.measure_d3(m, read(ROOT / 'D3_SELECTION.json'), tok, dev)
            rawpath = out / f'd3_update{u}_RAW.jsonl'
            with rawpath.open('x', encoding='utf-8', newline='\n') as f:
                for r in raw: f.write(json.dumps(r) + '\n')
                f.flush(); os.fsync(f.fileno())
            rt.atomic_json(E.d3_summary(raw), dpath)
        d = read(dpath)
        if u == 0:
            assert agg['correct'] == 9 and agg['exact'] == 0
            assert abs(q['language']['loss'] - 3.3907) / 3.3907 < 1e-4
            assert abs(d['mean_combined_name_probability'] - .000907) < .0001
        gates = {'binding': all(v['gate'] for v in q['binding'].values()),
                 'language': base_loss is None or q['language']['loss'] <= base_loss + .25,
                 'd3': d['mean_combined_name_probability'] <= .01,
                 'acquisition_guard': u != 100 or not (agg['correct'] < 10 and agg['exact'] < 10)}
        rt.atomic_json(gates, out / f'update{u}_gates.json')
        assert all(gates.values()), ('RETENTION_OR_ACQUISITION_GUARD_STOP', u, gates)
    assert equality(rng, rt.capture_rng()), 'evaluation consumed RNG'
    return agg


def run(a):
    p, b, base, pool, schedule, lr_rows, rows, binding = verify()
    if a.mode == 'preload':
        print('SF4_REAL_DEPENDENCY_PRELOAD_PASS', flush=True); return
    arm = a.arm
    out = ROOT / arm
    dev = torch.device('cuda')
    if arm in ('control', 'treatment'): check_branch_receipt()
    if a.mode == 'branch-preflight':
        assert arm in ('control', 'treatment')
    out.mkdir(exist_ok=True)
    prov = provenance()
    m = rt.load_model(Path(base['parent']), dev, b)
    scope = E.scope_audit(m)
    rt.atomic_json(scope, out / 'PARAMETER_SCOPE.json')
    tok = E.Tokenizer.from_file(base['tokenizer'])
    teacher = E.build_teacher(Path(base['parent']), dev)
    E.assert_teacher_disjoint(m.base_model, teacher)
    completed = 0
    restart = out / 'restart.pt'
    if arm in ('control', 'treatment') and not a.resume:
        assert not restart.exists()
        source = ROOT / f'{arm}_initial.pt'
        state = torch.load(source, map_location=dev, weights_only=False)
        assert state['arm'] == arm and state['provenance'] == prov
        assert state['completed'] == 100
        m.load_state_dict(state['model']); completed = 100
        for n in ('update0_checks.json',):
            if not (out / n).exists(): shutil.copyfile(ROOT / 'common_a' / n, out / n)
    elif a.resume:
        assert restart.exists()
        state = torch.load(restart, map_location=dev, weights_only=False)
        assert state['provenance'] == prov and state['arm'] == arm
        completed = state['completed']; m.load_state_dict(state['model'])
    else:
        assert arm in ('common_a', 'common_b') and not restart.exists()
        eval_at(m, b, out, 0, rows, tok, dev)
        m.eval()
        with torch.no_grad():
            kl = E.compute_kl(m, teacher, dev, pool['rows'], E.pick_kl_entries(pool['entries'], 1))
        assert abs(float(kl)) < 1e-9
        m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=5e-5, betas=(.9,.999), eps=1e-8,
                           weight_decay=.05, amsgrad=False, foreach=False, fused=False)
    if completed:
        opt.load_state_dict(state['optimizer'])
        E.set_scope(m, state['scope'] == 'binding')
        rt.restore_rng(state['rng'])
        assert equality(m.state_dict(), state['model'])
        assert equality(opt.state_dict(), state['optimizer'])
        assert equality(rt.capture_rng(), state['rng'])
        assert state['english_index'] == sum(u['kind'] == 'english' for u in schedule[:completed])
    if a.mode == 'branch-preflight':
        receipt = {'status':'BRANCH_ACTUAL_LOAD_EXACT', 'arm':arm, 'completed':completed,
                   'model':state_digest(m.state_dict()), 'optimizer':state_digest(opt.state_dict()),
                   'rng':state_digest(rt.capture_rng()), 'provenance':prov, 'optimizer_updates':0}
        rt.atomic_json(receipt, out / 'BRANCH_PREFLIGHT.json')
        print(json.dumps(receipt), flush=True); return
    if arm in ('control','treatment'):
        for ar in ('control','treatment'):
            assert read(ROOT / ar / 'BRANCH_PREFLIGHT.json')['status'] == 'BRANCH_ACTUAL_LOAD_EXACT'
        ap, bp = [read(ROOT / ar / 'BRANCH_PREFLIGHT.json') for ar in ('control','treatment')]
        assert all(ap[k] == bp[k] for k in ('model','optimizer','rng','provenance'))
    pin = rt.pinned_binding(b)
    base_loss = read(out / 'update0_checks.json')['language']['loss']
    english_index = sum(u['kind'] == 'english' for u in schedule[:completed])
    end = 100 if arm.startswith('common') else 200

    def save(u):
        val = {'completed':u, 'model':m.state_dict(), 'optimizer':opt.state_dict(),
               'rng':rt.capture_rng(), 'scope':'binding' if u%10==0 else 'english',
               'english_index':english_index, 'provenance':prov, 'arm':arm}
        rt.atomic_torch_save(val, restart)
        return val

    def scheduled_eval(u):
        if u in (100,125,150,175,200):
            cp = out / f'checkpoint_{u}.pt'
            if not cp.exists():
                rt.atomic_torch_save({'model_state_dict':m.state_dict(),'update':u,'provenance':prov,'arm':arm}, cp)
                (out / f'checkpoint_{u}.sha256').write_text(sha(cp)+'\n')
            else: assert sha(cp) == (out / f'checkpoint_{u}.sha256').read_text(encoding='utf-8').strip()
            eval_at(m,b,out,u,rows,tok,dev,base_loss,full=u in (100,200))

    if completed in (100,125,150,175,200): scheduled_eval(completed)
    if not completed: save(0)
    for u, lrrow in zip(schedule[completed:end], lr_rows[completed:end]):
        lr = lrrow['treatment'] if arm == 'treatment' else lrrow['control']
        for g in opt.param_groups: g['lr'] = lr
        opt.zero_grad(set_to_none=True)
        E.set_scope(m, u['kind']=='binding')
        inactive = {n:(x.detach().clone(),{k:v.clone() if torch.is_tensor(v) else v for k,v in opt.state.get(x,{}).items()})
                    for n,x in m.named_parameters() if not x.requires_grad}
        ce = kl = None
        if u['kind'] == 'english':
            english_index += 1
            m.eval()
            kl = E.compute_kl(m,teacher,dev,pool['rows'],E.pick_kl_entries(pool['entries'],english_index))
            m.train()
            idx = {r['id']:r for r in rows}
            x,y = E.pad_batch([idx[i] for i in u['ids']],u['pad'])
            ce = F.cross_entropy(m.base_model(x.to(dev)).reshape(-1,1024),y.to(dev).reshape(-1),ignore_index=-100)
            loss = ce + E.LAMBDA_KL * kl
        else:
            m.train()
            loss,_ = rt.binding_loss(m,rt.binding_docs_for_batch(binding,u['quartets'],u['documents']),dev,pin)
        loss.backward()
        assert all(x.grad is None for x in m.parameters() if not x.requires_grad)
        norm = torch.nn.utils.clip_grad_norm_([x for x in m.parameters() if x.grad is not None],2.0)
        opt.step()
        for n,x in m.named_parameters():
            if n in inactive:
                old,st=inactive[n]
                assert torch.equal(x,old) and equality(opt.state.get(x,{}),st)
        rec={'update':u['update'],'kind':u['kind'],'lr':lr,'loss':float(loss.detach()),'grad_norm':float(norm)}
        if ce is not None: rec.update(ce=float(ce.detach()),kl=float(kl.detach()))
        committed_state = save(u['update'])
        if u['update'] == 101:
            path101 = out / 'matched_step101_state.pt'
            rt.atomic_torch_save(committed_state, path101)
            if arm == 'treatment':
                ctrl101 = torch.load(ROOT / 'control' / 'matched_step101_state.pt',map_location='cpu',weights_only=False)
                common_keys = [k for k in committed_state if k != 'arm']
                exact = all(equality(committed_state[k],ctrl101[k]) for k in common_keys)
                rt.atomic_json({'exact_model_optimizer_rng_controller':exact,'same_lr':lr==5e-5,
                                'treatment_variable_not_yet_changed':True},ROOT/'MATCHED_STEP101.json')
                assert exact, 'HARD_STOP_SAME_LR_FIRST_BRANCH_STEP_MISMATCH'
        rt.atomic_json(rec,out / f'metric_{u["update"]:03}.json')
        if u['update']%25==0: print(arm,'committed',u['update'],flush=True)
        scheduled_eval(u['update'])
    rt.atomic_json({'status':'COMMON_COMPLETE' if end==100 else 'ARM_ENDPOINT_COMPLETE',
                    'completed':end,'checkpoint_sha256':sha(out / f'checkpoint_{end}.pt'),
                    'restart_sha256':sha(restart),'transfer_opened':False},out / 'STATUS.json')


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--arm',choices=['common_a','common_b','control','treatment'],required=True)
    ap.add_argument('--mode',choices=['preload','train','branch-preflight'],required=True)
    ap.add_argument('--resume',action='store_true')
    a=ap.parse_args()
    try: run(a)
    except Exception as ex:
        directory=ROOT / a.arm
        directory.mkdir(exist_ok=True)
        rt.atomic_json({'status':'HARD_STOP','error_type':type(ex).__name__,'error':str(ex)},directory/'HARD_STOP.json')
        raise
