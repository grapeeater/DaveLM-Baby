import hashlib,json,os,stat
from datetime import datetime,timezone
from pathlib import Path
H=Path(__file__).resolve().parent; X={'SHA256SUMS.txt','FREEZE_RECEIPT.json','FREEZE_RECEIPT.sha256'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def main():
 assert json.loads((H/'PREFLIGHT.json').read_text())['status']=='SF19_PROSPECTIVE_PREFLIGHT_PASS'
 fs=[p for p in sorted(H.rglob('*')) if p.is_file() and p.name not in X and not any(x in p.parts for x in ['runs','logs','__pycache__'])]
 (H/'SHA256SUMS.txt').write_text('\n'.join(f'{sha(p)}  {p.relative_to(H).as_posix()}' for p in fs)+'\n',encoding='utf-8',newline='\n'); mh=sha(H/'SHA256SUMS.txt')
 rec={'status':'SF19_PROSPECTIVE_PREFLIGHT_PASS','study':'SF19_CANDIDATE_MEMBERSHIP_HINGE_V1','manifest_sha256':mh,'sealed_utc':datetime.now(timezone.utc).isoformat(),'sole_scientific_change':'bounded candidate-membership hinge at first answer position','checkpoint_loaded':False,'optimizer_created':False,'updates':0,'final_accessed':False,'sacred_accessed':False}
 (H/'FREEZE_RECEIPT.json').write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n'); rh=sha(H/'FREEZE_RECEIPT.json'); (H/'FREEZE_RECEIPT.sha256').write_text(rh+'  FREEZE_RECEIPT.json\n',encoding='utf-8',newline='\n')
 for p in fs+[H/'SHA256SUMS.txt',H/'FREEZE_RECEIPT.json',H/'FREEZE_RECEIPT.sha256']: os.chmod(p,stat.S_IREAD)
 print('SF19_SEALED',rh,mh)
if __name__=='__main__': main()
