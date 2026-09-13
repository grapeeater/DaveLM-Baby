"""Sealed sequential SF21 launcher: one fresh process per preregistered seed."""
import json, os, subprocess, sys
from datetime import datetime, timezone
import CONTROLLER as C
def now(): return datetime.now(timezone.utc).isoformat()
def main():
 p=C.verify(); ledger=[]; lp=C.H/'RUN_LEDGER.json'
 assert not lp.exists() and not (C.H/'runs').exists(); logs=C.H/'execution_logs'; logs.mkdir()
 for r in p['runs']:
  seed=r['seed']; cmd=[sys.executable,'-B',str(C.H/'CONTROLLER.py'),'--seed',str(seed),'--arm','curriculum','--mode','train']
  rec={'seed':seed,'parent_sha256':r['parent_checkpoint_sha256'],'command':cmd,'start':now(),'state':'LAUNCH_INTENT'}; ledger.append(rec); C.rt.atomic_json(ledger,lp)
  env=dict(os.environ,PYTHONHASHSEED=str(seed),PYTHONDONTWRITEBYTECODE='1')
  with (logs/f'seed{seed}.out.log').open('x',encoding='utf-8',newline='\n') as out, (logs/f'seed{seed}.err.log').open('x',encoding='utf-8',newline='\n') as err:
   child=subprocess.Popen(cmd,cwd=C.H,env=env,stdout=out,stderr=err); rec.update(pid=child.pid,state='RUNNING'); C.rt.atomic_json(ledger,lp); code=child.wait()
  rec.update(end=now(),returncode=code)
  if code: rec['state']='MECHANICAL_FAILURE'; C.rt.atomic_json(ledger,lp); raise RuntimeError(f'seed {seed} exit {code}')
  st=C.read(C.H/'runs'/f'seed_{seed}_curriculum'/'STATUS.json'); assert C.sha(st['checkpoint'])==st['checkpoint_sha256']
  rec.update(state='CLASSIFIED',status=st['status'],completed=st['completed'],checkpoint_sha256=st['checkpoint_sha256']); C.rt.atomic_json(ledger,lp)
  print('CLASSIFIED',seed,st['status'],st['completed'],flush=True)
 subprocess.run([sys.executable,'-B',str(C.H/'REPORTER.py')],cwd=C.H,check=True)
if __name__=='__main__': main()
