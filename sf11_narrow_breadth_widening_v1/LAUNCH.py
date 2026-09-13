"""One launch per seed, sequential foreground children, durable ledger, no automatic retry."""
import os,sys,subprocess
from datetime import datetime,timezone
import CONTROLLER as C
def now(): return datetime.now(timezone.utc).isoformat()
def main():
    p=C.verify(); lp=C.H/'RUN_LEDGER.json'
    assert not lp.exists(),'Already launched: inspect ledger, never duplicate launches'
    assert not (C.H/'runs').exists(),'Unexpected preexisting run'
    ledger=[]; logs=C.H/'logs'; logs.mkdir(exist_ok=False)
    for r in p['runs']:
        d=C.H/'runs'/f"seed_{r['seed']}_curriculum"
        env=dict(os.environ,PYTHONHASHSEED=str(r['seed']),PYTHONDONTWRITEBYTECODE='1')
        cmd=[sys.executable,'-B',str(C.H/'CONTROLLER.py'),'--seed',str(r['seed']),'--arm','curriculum','--mode','train']
        rec={'seed':r['seed'],'parent_sha256':r['parent_checkpoint_sha256'],'command':cmd,'start':now(),'state':'LAUNCH_INTENT'}
        ledger.append(rec); C.rt.atomic_json(ledger,lp)
        print('STARTING',r['seed'],flush=True)
        with (logs/f"seed_{r['seed']}.log").open('x',encoding='utf-8') as log:
            child=subprocess.Popen(cmd,cwd=C.H,env=env,stdout=log,stderr=subprocess.STDOUT)
            rec.update(pid=child.pid,state='RUNNING'); C.rt.atomic_json(ledger,lp)
            code=child.wait()  # blocking, attached to this foreground controller; no detached processes
        rec.update(end=now(),returncode=code)
        if code:
            rec['state']='MECHANICAL_STOP'; C.rt.atomic_json(ledger,lp); raise RuntimeError(f'Run {r["seed"]} failed; no next launch')
        st=C.read(d/'STATUS.json')
        assert st['completed'] in [100,200] and st['status'] in ['STOP_REGRESSION','ACQUISITION_SUCCESS','ACQUISITION_FAIL']
        assert C.sha(st['checkpoint'])==st['checkpoint_sha256']
        rec.update(state='CLASSIFIED',status=st['status'],completed=st['completed']); C.rt.atomic_json(ledger,lp)
        print('CLASSIFIED',r['seed'],st['status'],st['completed'],flush=True)
    subprocess.run([sys.executable,'-B',str(C.H/'REPORTER.py')],cwd=C.H,check=True)
if __name__=='__main__': main()
