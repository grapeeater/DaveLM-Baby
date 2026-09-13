import hashlib,json,os,stat
from datetime import datetime,timezone
from pathlib import Path
H=Path(__file__).resolve().parent; X={'SHA256SUMS.txt','FREEZE_RECEIPT.json','FREEZE_RECEIPT.sha256'}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 pf=json.loads((H/'PREFLIGHT.json').read_text()); assert pf['status']=='SF21_PROSPECTIVE_PREFLIGHT_PASS'
 fs=[p for p in sorted(H.rglob('*')) if p.is_file() and p.name not in X and not any(x in p.parts for x in ['runs','execution_logs','__pycache__'])]
 (H/'SHA256SUMS.txt').write_text('\n'.join(f'{sha(p)}  {p.relative_to(H).as_posix()}' for p in fs)+'\n',encoding='utf-8',newline='\n'); mh=sha(H/'SHA256SUMS.txt')
 rec={'status':'SF21_PROSPECTIVE_PREFLIGHT_PASS','study':'SF21_PAIRED_COUNTERFACTUAL_REPRESENTATION_CONSISTENCY_V2','manifest_sha256':mh,'sealed_utc':datetime.now(timezone.utc).isoformat(),'sole_scientific_change':'lambda=0.1 paired cosine consistency at final-normalized pre-answer query state','predecessor_failed_attempt':str(H.parent/'sf21_paired_counterfactual_representation_consistency_v1'),'checkpoint_loaded_for_u0_only':True,'optimizer_created':False,'updates':0,'locked_transfer':'LOCKED_UNSCORED','final_accessed':False,'sacred_accessed':False}
 (H/'FREEZE_RECEIPT.json').write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n'); rh=sha(H/'FREEZE_RECEIPT.json'); (H/'FREEZE_RECEIPT.sha256').write_text(rh+'  FREEZE_RECEIPT.json\n',encoding='utf-8',newline='\n')
 for p in fs+[H/'SHA256SUMS.txt',H/'FREEZE_RECEIPT.json',H/'FREEZE_RECEIPT.sha256']: os.chmod(p,stat.S_IREAD)
 print('SF21_SEALED',rh,mh)
if __name__=='__main__': main()
