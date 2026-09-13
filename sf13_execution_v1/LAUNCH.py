"""Mechanical sequential launcher for the sealed SF13 controller; no scientific logic."""
import hashlib,json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent/'sf13_broad_coverage_kl_retention_v1'
PY=Path(r'C:\DaveLM-CADAVER\sf2_runtime\sf2venv\Scripts\python.exe')
SEEDS=(87032,87033,87034)

def now(): return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def atomic(obj,p):
 q=p.with_suffix(p.suffix+'.tmp'); q.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n'); os.replace(q,p)
def verify():
 assert sha(STUDY/'FREEZE_RECEIPT.json')==(STUDY/'FREEZE_RECEIPT.sha256').read_text().split()[0]
 r=json.loads((STUDY/'FREEZE_RECEIPT.json').read_text()); assert r['status']=='SF13_PROSPECTIVE_PREFLIGHT_PASS'
 assert sha(STUDY/'SHA256SUMS.txt')==r['manifest_sha256']
 for line in (STUDY/'SHA256SUMS.txt').read_text().splitlines():
  h,n=line.split('  ',1); assert sha(STUDY/n)==h,n
def main():
 verify(); ledger_path=HERE/'RUN_LEDGER.json'; assert not ledger_path.exists(); assert not (STUDY/'runs').exists()
 logs=HERE/'logs'; logs.mkdir(); ledger=[]
 for seed in SEEDS:
  cmd=[str(PY),'-B',str(STUDY/'CONTROLLER.py'),'--seed',str(seed),'--arm','curriculum','--mode','train']
  rec={'seed':seed,'command':cmd,'start':now(),'state':'LAUNCH_INTENT'}; ledger.append(rec); atomic(ledger,ledger_path)
  env=dict(os.environ,PYTHONHASHSEED=str(seed),PYTHONDONTWRITEBYTECODE='1')
  print('STARTING',seed,flush=True)
  with (logs/f'seed_{seed}.log').open('x',encoding='utf-8',newline='\n') as log:
   child=subprocess.Popen(cmd,cwd=STUDY,env=env,stdout=log,stderr=subprocess.STDOUT)
   rec.update(pid=child.pid,state='RUNNING'); atomic(ledger,ledger_path); code=child.wait()
  rec.update(end=now(),returncode=code)
  if code:
   rec['state']='MECHANICAL_FAILURE'; atomic(ledger,ledger_path); raise RuntimeError(f'seed {seed} controller exit {code}')
  out=STUDY/'runs'/f'seed_{seed}_curriculum'; status=json.loads((out/'STATUS.json').read_text())
  assert status['status'] in ['STOP_REGRESSION','ACQUISITION_SUCCESS','ACQUISITION_FAIL'] and status['completed'] in [100,200]
  assert sha(status['checkpoint'])==status['checkpoint_sha256']
  rec.update(state='CLASSIFIED',status=status['status'],completed=status['completed'],checkpoint=status['checkpoint'],checkpoint_sha256=status['checkpoint_sha256'])
  atomic(ledger,ledger_path); print('CLASSIFIED',seed,status['status'],status['completed'],flush=True)
 print('SF13_THREE_BRANCH_EXECUTION_COMPLETE',flush=True)
if __name__=='__main__': main()
