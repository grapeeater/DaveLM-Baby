"""Fixed three-run execution order; each run in a fresh process, launched exactly once,
sequentially, and BLOCKING (subprocess.run) - no detached/background process ambiguity.
"""
import os, subprocess, sys
import CONTROLLER as C


def main():
    p = C.verify()
    logs = C.H / 'logs'; logs.mkdir(exist_ok=True)
    ledger = []
    for r in p['runs']:
        name = f"seed_{r['seed']}_curriculum"
        out = C.H / 'runs' / name
        if (out / 'STATUS.json').exists():
            status = C.read(out / 'STATUS.json')
            print('ALREADY_CLASSIFIED', name, status['status'], flush=True)
            ledger.append({'seed': r['seed'], 'action': 'already_classified', 'status': status['status']})
            continue
        env = dict(os.environ, PYTHONHASHSEED=str(r['seed']), PYTHONDONTWRITEBYTECODE='1')
        cmd = [sys.executable, '-B', str(C.H / 'CONTROLLER.py'), '--seed', str(r['seed']), '--arm', r['arm'], '--mode', 'train']
        if (out / 'restart.pt').exists(): cmd.append('--resume')
        print('STARTING', name, flush=True)
        ledger.append({'seed': r['seed'], 'action': 'started'})
        with (logs / (name + '.log')).open('a', encoding='utf-8') as log:
            completed = subprocess.run(cmd, cwd=C.H, env=env, stdout=log, stderr=subprocess.STDOUT)
        ledger[-1]['returncode'] = completed.returncode
        if completed.returncode:
            C.rt.atomic_json(ledger, C.H / 'RUN_LEDGER.json')
            raise RuntimeError(f'Execution stopped for {name} (returncode={completed.returncode}); inspect preserved log. No next run launched.')
        status = C.read(out / 'STATUS.json')
        ledger[-1]['status'] = status['status']
        print('CLASSIFIED', name, status['status'], status['completed'], flush=True)
    C.rt.atomic_json(ledger, C.H / 'RUN_LEDGER.json')
    subprocess.run([sys.executable, '-B', str(C.H / 'REPORTER.py')], cwd=C.H, check=True)


if __name__ == '__main__':
    main()
