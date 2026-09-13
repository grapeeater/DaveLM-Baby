"""Create the non-circular SF12 integrity chain after successful preflight."""
import hashlib,json,os,stat
from datetime import datetime,timezone
from pathlib import Path

H=Path(__file__).resolve().parent
EXCLUDE={'FREEZE_RECEIPT.json','FREEZE_RECEIPT.sha256','SHA256SUMS.txt'}

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def main():
 pf=json.loads((H/'PREFLIGHT.json').read_text()); assert pf['status']=='SF12_PROSPECTIVE_PREFLIGHT_PASS'
 files=[]
 for p in sorted(H.rglob('*')):
  if p.is_file() and p.name not in EXCLUDE and not any(x in p.parts for x in ['runs','logs','__pycache__']): files.append(p)
 lines=[f"{sha(p)}  {p.relative_to(H).as_posix()}" for p in files]
 (H/'SHA256SUMS.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
 mh=sha(H/'SHA256SUMS.txt')
 receipt={'status':'SF12_PROSPECTIVE_PREFLIGHT_PASS','study':'SF12_ANSWER_VOCAB_RETENTION_V1','manifest_sha256':mh,'sealed_utc':datetime.now(timezone.utc).isoformat(),'scientific_change':'R_name only','updates':0,'checkpoint_loaded':False,'optimizer_created':False,'final_accessed':False,'sacred_accessed':False}
 (H/'FREEZE_RECEIPT.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
 rh=sha(H/'FREEZE_RECEIPT.json'); (H/'FREEZE_RECEIPT.sha256').write_text(rh+'  FREEZE_RECEIPT.json\n',encoding='utf-8',newline='\n')
 for p in files+[H/'SHA256SUMS.txt',H/'FREEZE_RECEIPT.json',H/'FREEZE_RECEIPT.sha256']:
  os.chmod(p,stat.S_IREAD)
 print('SF12_SEALED',rh,mh)

if __name__=='__main__': main()
