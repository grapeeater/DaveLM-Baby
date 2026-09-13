import hashlib,json
from pathlib import Path
D=Path(r'C:\DaveLM-CADAVER\fact_supervision_87001_corrected_v6')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 for n in ('SHA256SUMS.txt','FREEZE_RECEIPT.json','FREEZE_RECEIPT.sha256'):
  p=D/n
  if p.exists(): p.chmod(0o666); p.unlink()
 payload=sorted(p for p in D.iterdir() if p.is_file() and p.name not in ('SHA256SUMS.txt','FREEZE_RECEIPT.json','FREEZE_RECEIPT.sha256'))
 (D/'SHA256SUMS.txt').write_bytes(''.join(f'{sha(p)}  {p.name}\n' for p in payload).encode())
 rec={'status':'FULLY_EXECUTABLE_PRETRAINING_FREEZE_COMPLETE','bundle':'fact_supervision_87001_corrected_v6','immutable_predecessor':'fact_supervision_87001_corrected_v5','scientific_specification_changed':False,'mechanical_change':'Controller receipt-status acceptance and status regression test only','manifest_sha256':sha(D/'SHA256SUMS.txt'),'payload_count':len(payload),'checkpoint_loaded':False,'optimizer_created':False,'optimizer_update':False,'behavioral_inference':False,'sacred_access':False,'seed_87002_executed':False,'non_circular_chain':'FREEZE_RECEIPT.sha256 -> FREEZE_RECEIPT.json -> SHA256SUMS.txt -> payload'}
 (D/'FREEZE_RECEIPT.json').write_bytes((json.dumps(rec,sort_keys=True,indent=2)+'\n').encode())
 (D/'FREEZE_RECEIPT.sha256').write_text(sha(D/'FREEZE_RECEIPT.json')+'  FREEZE_RECEIPT.json\n',encoding='utf-8')
 for p in D.iterdir(): p.chmod(0o444)
 print(json.dumps({'status':'RESEALED','manifest_sha256':rec['manifest_sha256'],'receipt_sha256':sha(D/'FREEZE_RECEIPT.json'),'payload_count':len(payload)}))
if __name__=='__main__':main()
