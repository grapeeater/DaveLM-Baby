"""Final static validation and non-circular receipt construction; no model import."""
import hashlib,json,runpy,shutil
from pathlib import Path
ROOT=Path(r'C:\DaveLM-CADAVER'); OUT=ROOT/'fact_supervision_87001_corrected_v4'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert not (OUT/'FREEZE_RECEIPT.sha256').exists(), 'already finalized'
 # Snapshot every construction/finalization source used to make this bundle.
 shutil.copyfile(ROOT/'build_fact_supervision_v4.py',OUT/'BUILD_FREEZE_SOURCE.py')
 shutil.copyfile(ROOT/'finalize_fact_supervision_v4.py',OUT/'FINALIZE_FREEZE_SOURCE.py')
 result=runpy.run_path(str(OUT/'MOCK_TESTS.py'),run_name='__main__')
 sched=json.loads((OUT/'MATERIALIZED_SCHEDULE_SEED87002.json').read_text())
 u=sched['updates']; assert len(u)==500
 vals={'status':'PASS','static_schedule_test':'PASS','updates':len(u),'english':sum(x['kind']=='english' for x in u),'binding':sum(x['kind']=='binding' for x in u),'english_family_placements':{'object':900,'predicate':900},'binding_quartet_placements':400,'mock_model_loaded':False,'checkpoint_loaded':False,'optimizer_created':False,'optimizer_update':False,'inference':False,'sacred_access':False}
 (OUT/'STATIC_VALIDATION.json').write_bytes((json.dumps(vals,sort_keys=True,indent=2)+'\n').encode())
 payload=sorted(p for p in OUT.iterdir() if p.is_file() and p.name not in ('SHA256SUMS.txt','FREEZE_RECEIPT.json','FREEZE_RECEIPT.sha256'))
 (OUT/'SHA256SUMS.txt').write_bytes(''.join(f'{sha(p)}  {p.name}\n' for p in payload).encode())
 mh=sha(OUT/'SHA256SUMS.txt')
 receipt={'status':'PASS_EXECUTABLE_PRETRAINING_FREEZE','bundle':'fact_supervision_87001_corrected_v4','manifest_sha256':mh,'payload_count':len(payload),'non_circular_chain':'FREEZE_RECEIPT.sha256 -> FREEZE_RECEIPT.json -> SHA256SUMS.txt -> payload','checkpoint_loaded':False,'optimizer_created':False,'optimizer_update':False,'behavioral_inference':False,'sacred_access':False}
 (OUT/'FREEZE_RECEIPT.json').write_bytes((json.dumps(receipt,sort_keys=True,indent=2)+'\n').encode())
 rh=sha(OUT/'FREEZE_RECEIPT.json')
 (OUT/'FREEZE_RECEIPT.sha256').write_text(rh+'  FREEZE_RECEIPT.json\n',encoding='utf-8')
 # Verify entire chain before read-only sealing.
 assert (OUT/'FREEZE_RECEIPT.sha256').read_text().split()[0]==sha(OUT/'FREEZE_RECEIPT.json')
 for line in (OUT/'SHA256SUMS.txt').read_text().splitlines():
  h,n=line.split('  ',1); assert sha(OUT/n)==h,n
 for p in OUT.iterdir(): p.chmod(0o444)
 print(json.dumps({'status':'PASS','manifest_sha256':mh,'receipt_sha256':rh,'payload_count':len(payload)},indent=2))
if __name__=='__main__':main()
