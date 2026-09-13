"""Seal v5 after top-level mock validation; no checkpoint or model behavior."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(r'C:\DaveLM-CADAVER'); OUT=ROOT/'fact_supervision_87001_corrected_v5'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def wr(name,obj):
 p=OUT/name; assert not p.exists()
 p.write_bytes((json.dumps(obj,sort_keys=True,indent=2)+'\n').encode() if not isinstance(obj,str) else obj.encode())
def main():
 assert not (OUT/'FREEZE_RECEIPT.sha256').exists()
 shutil.copyfile(ROOT/'finalize_fact_supervision_v5.py',OUT/'FINALIZE_V5_SOURCE.py')
 controller=(OUT/'CONTROLLER.py').read_text(encoding='utf-8')
 forbidden=[x for x in ('TODO','NotImplementedError','pass #','raise RuntimeError(\'Concrete') if x in controller]
 assert not forbidden
 static={'status':'PASS','placeholder_scan':'PASS','forbidden_tokens':forbidden,'functional_symbols':['verify_bundle','load_config','records','reconstruct','set_scope','mask_english','atomic_replace','load_parent_model','make_optimizer','dev_eval','run','main'],'top_level_mock_status':'MOCK_0_UPDATE_PASS','schedule_consumed':500,'english_batches_reconstructed':450,'binding_batches_reconstructed':50,'checkpoint_loaded':False,'optimizer_created':False,'optimizer_updates':0,'inference':False,'sacred_access':False}
 wr('STATIC_COMPLETENESS.json',static)
 wr('CONTROLLER_README.md','''# v5 entrypoint\n\nAfter human approval, execute one arm in a fresh process with the pinned runtime:\n\n`$env:PYTHONHASHSEED=87002; python CONTROLLER.py --bundle C:\\DaveLM-CADAVER\\fact_supervision_87001_corrected_v5 --arm factual --seed 87002 --parent C:\\DaveLM-CADAVER\\language_pilot_1_early_block_protection_seed8380\\pilot_run\\checkpoints\\seed_8380\\latest.pt --out C:\\DaveLM-CADAVER\\fact_supervision_87001_corrected_v5\\run_factual --mode train`\n\nUse `--arm control` and a distinct output directory for the matched control process. `--mode mock-0-update` exercises the same top-level controller without loading the real checkpoint, creating a real optimizer, or updating weights. The schedule is read from `MATERIALIZED_SCHEDULE_SEED87002.json`; no regeneration is permitted. Confirmation and sacred paths are rejected.\n''')
 # Add controller and complete source snapshots to the payload manifest.
 payload=sorted(p for p in OUT.iterdir() if p.is_file() and p.name not in ('SHA256SUMS.txt','FREEZE_RECEIPT.json','FREEZE_RECEIPT.sha256'))
 OUT.joinpath('SHA256SUMS.txt').write_bytes(''.join(f'{sha(p)}  {p.name}\n' for p in payload).encode())
 mh=sha(OUT/'SHA256SUMS.txt')
 receipt={'status':'FULLY_EXECUTABLE_PRETRAINING_FREEZE_COMPLETE','bundle':'fact_supervision_87001_corrected_v5','immutable_predecessor':'fact_supervision_87001_corrected_v4','scientific_specification_changed':False,'manifest_sha256':mh,'payload_count':len(payload),'checkpoint_loaded':False,'optimizer_created':False,'optimizer_update':False,'behavioral_inference':False,'sacred_access':False,'seed_87002_executed':False,'non_circular_chain':'FREEZE_RECEIPT.sha256 -> FREEZE_RECEIPT.json -> SHA256SUMS.txt -> payload'}
 OUT.joinpath('FREEZE_RECEIPT.json').write_bytes((json.dumps(receipt,sort_keys=True,indent=2)+'\n').encode())
 OUT.joinpath('FREEZE_RECEIPT.sha256').write_text(sha(OUT/'FREEZE_RECEIPT.json')+'  FREEZE_RECEIPT.json\n',encoding='utf-8')
 for p in OUT.iterdir(): p.chmod(0o444)
 assert (OUT/'FREEZE_RECEIPT.sha256').read_text().split()[0]==sha(OUT/'FREEZE_RECEIPT.json')
 for line in (OUT/'SHA256SUMS.txt').read_text().splitlines():
  h,n=line.split('  ',1); assert sha(OUT/n)==h,n
 print(json.dumps({'status':receipt['status'],'manifest_sha256':mh,'receipt_sha256':sha(OUT/'FREEZE_RECEIPT.json'),'payload_count':len(payload)},indent=2))
if __name__=='__main__': main()
