"""Non-circular pre-training receipt. Run exactly once after successful preflight/review."""
from datetime import datetime,timezone
from pathlib import Path
import os,stat
import CONTROLLER as C

def main():
    assert not (C.H/'FREEZE_RECEIPT.json').exists()
    assert C.read(C.H/'PREFLIGHT.json')['status']=='PASS'
    assert C.read(C.H/'INDEPENDENT_REVIEW.json')['status']=='PASS'
    files=sorted(p for p in C.H.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    assert not (C.H/'runs').exists()
    text=''.join(f'{C.sha(p)}  {p.relative_to(C.H).as_posix()}\n' for p in files)
    (C.H/'SHA256SUMS.txt').write_text(text,encoding='utf-8',newline='\n')
    C.rt.atomic_json({'status':'SF6_PROSPECTIVE_PREFLIGHT_PASS','created_utc':datetime.now(timezone.utc).isoformat(),
        'manifest_sha256':C.sha(C.H/'SHA256SUMS.txt'),'protocol_sha256':C.sha(C.H/'PROTOCOL.json'),
        'payload_files':len(files),'historical_artifacts_unchanged':True,'no_training_before_freeze':True},C.H/'FREEZE_RECEIPT.json')
    (C.H/'FREEZE_RECEIPT.sha256').write_text(C.sha(C.H/'FREEZE_RECEIPT.json')+'  FREEZE_RECEIPT.json\n',encoding='utf-8',newline='\n')
    for p in files+[C.H/'SHA256SUMS.txt',C.H/'FREEZE_RECEIPT.json',C.H/'FREEZE_RECEIPT.sha256']:
        p.chmod(stat.S_IREAD)
    C.verify()
    print('SEALED_RECEIPT',C.sha(C.H/'FREEZE_RECEIPT.json'))
    print('SEALED_MANIFEST',C.sha(C.H/'SHA256SUMS.txt'))

if __name__=='__main__': main()
