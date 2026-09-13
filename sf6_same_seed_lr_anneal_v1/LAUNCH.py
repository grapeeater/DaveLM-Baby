"""Fixed six-run execution order; each run in a fresh process. No outcome-based additions."""
import os, subprocess, sys
from pathlib import Path
import CONTROLLER as C

def main():
    p=C.verify()
    logs=C.H/'logs'; logs.mkdir(exist_ok=True)
    for r in p['runs']:
        name=f"seed_{r['seed']}_{r['arm']}"
        out=C.H/'runs'/name
        if (out/'STATUS.json').exists():
            prov=C.read(out/'PROVENANCE.json')
            assert prov['receipt']==C.sha(C.H/'FREEZE_RECEIPT.json')
            status=C.read(out/'STATUS.json')
            assert C.sha(status['checkpoint'])==status['checkpoint_sha256']
            print('ALREADY_CLASSIFIED',name,status['status'],flush=True)
            continue
        env=dict(os.environ,PYTHONHASHSEED=str(r['seed']),PYTHONDONTWRITEBYTECODE='1')
        cmd=[sys.executable,'-B',str(C.H/'CONTROLLER.py'),'--seed',str(r['seed']),'--arm',r['arm'],'--mode','train']
        if (out/'restart.pt').exists(): cmd.append('--resume')
        print('STARTING',name,flush=True)
        with (logs/(name+'.log')).open('a',encoding='utf-8') as log:
            completed=subprocess.run(cmd,cwd=C.H,env=env,stdout=log,stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f'Execution stopped for {name}; inspect preserved log. No next run launched.')
        status=C.read(out/'STATUS.json')
        print('CLASSIFIED',name,status['status'],status['completed'],flush=True)
    subprocess.run([sys.executable,'-B',str(C.H/'REPORTER.py')],cwd=C.H,check=True)

if __name__=='__main__': main()
