import hashlib,json,os,stat
from pathlib import Path
H=Path(__file__).resolve().parent; X={'SHA256SUMS.txt','FREEZE_RECEIPT.json','FREEZE_RECEIPT.sha256'}
def sha(p): h=hashlib.sha256();h.update(p.read_bytes());return h.hexdigest()
def main():
 assert json.loads((H/'PREFLIGHT.json').read_text())['status']=='PASS_MECHANICAL_REPORTING_HOTFIX'
 fs=[p for p in sorted(H.iterdir()) if p.is_file() and p.name not in X and p.name not in ['RESULTS.json','FINAL_REPORT.md','OUTPUT_SHA256SUMS.txt']]
 (H/'SHA256SUMS.txt').write_text('\n'.join(f'{sha(p)}  {p.name}' for p in fs)+'\n',encoding='utf-8',newline='\n')
 rec={'status':'SEALED_MECHANICAL_REPORTING_HOTFIX','manifest_sha256':sha(H/'SHA256SUMS.txt'),'source_study':'sf12_answer_vocab_retention_v1','scientific_change':False}
 (H/'FREEZE_RECEIPT.json').write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
 (H/'FREEZE_RECEIPT.sha256').write_text(sha(H/'FREEZE_RECEIPT.json')+'  FREEZE_RECEIPT.json\n',encoding='utf-8',newline='\n')
 for p in fs+[H/'SHA256SUMS.txt',H/'FREEZE_RECEIPT.json',H/'FREEZE_RECEIPT.sha256']: os.chmod(p,stat.S_IREAD)
 print('SF12_REPORTING_HOTFIX_SEALED',sha(H/'FREEZE_RECEIPT.json'),sha(H/'SHA256SUMS.txt'))
if __name__=='__main__': main()
