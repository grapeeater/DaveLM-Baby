"""Finalize immutable prospective payload; no run may precede this receipt."""
import stat
from datetime import datetime,timezone
import CONTROLLER as C
def main():
    assert not (C.H/'FREEZE_RECEIPT.json').exists() and not (C.H/'runs').exists()
    pre=C.read(C.H/'PREFLIGHT.json')
    assert pre['status']=='PASS' and pre['checkpoint_loaded'] is False and pre['optimizer_created'] is False and pre['updates']==0
    files=sorted(p for p in C.H.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    (C.H/'SHA256SUMS.txt').write_text(''.join(f'{C.sha(f)}  {f.relative_to(C.H).as_posix()}\n' for f in files),encoding='utf-8',newline='\n')
    C.rt.atomic_json({'status':'SF11_PROSPECTIVE_PREFLIGHT_PASS','created_utc':datetime.now(timezone.utc).isoformat(),'manifest_sha256':C.sha(C.H/'SHA256SUMS.txt'),'protocol_sha256':C.sha(C.H/'PROTOCOL.json'),'payload_count':len(files)},C.H/'FREEZE_RECEIPT.json')
    (C.H/'FREEZE_RECEIPT.sha256').write_text(C.sha(C.H/'FREEZE_RECEIPT.json')+'  FREEZE_RECEIPT.json\n',encoding='utf-8')
    for p in files+[C.H/n for n in ['SHA256SUMS.txt','FREEZE_RECEIPT.json','FREEZE_RECEIPT.sha256']]: p.chmod(stat.S_IREAD)
    C.verify()
    print('RECEIPT',C.sha(C.H/'FREEZE_RECEIPT.json'),'MANIFEST',C.sha(C.H/'SHA256SUMS.txt'))
if __name__=='__main__': main()
