"""Status-interface regression; temporary mutation is outside the sealed bundle."""
import json, shutil, tempfile
from pathlib import Path
import CONTROLLER

HERE=Path(__file__).resolve().parent
PARENT=Path(r'C:\DaveLM-CADAVER\language_pilot_1_early_block_protection_seed8380\pilot_run\checkpoints\seed_8380\latest.pt')

def rejected(path):
 try: CONTROLLER.verify_bundle(path)
 except AssertionError: return True
 return False

def make_writable(path):
 for p in path.rglob('*'):
  if p.is_file(): p.chmod(0o666)

assert CONTROLLER.verify_bundle(HERE)[0]['status']=='FULLY_EXECUTABLE_PRETRAINING_FREEZE_COMPLETE'
source=(HERE/'CONTROLLER.py').read_text(encoding='utf-8')
assert "PASS_EXECUTABLE_PRETRAINING_FREEZE" not in source
with tempfile.TemporaryDirectory(prefix='v6_status_regression_') as td:
 t=Path(td)/'bundle'; shutil.copytree(HERE,t,ignore=shutil.ignore_patterns('*.pyc','mock_*'))
 receipt_path=t/'FREEZE_RECEIPT.json'; receipt_path.chmod(0o666)
 receipt=json.loads(receipt_path.read_text()); receipt['status']='PASS_EXECUTABLE_PRETRAINING_FREEZE'; receipt_path.write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n'); assert rejected(t)
 make_writable(t); shutil.rmtree(t); shutil.copytree(HERE,t,ignore=shutil.ignore_patterns('*.pyc','mock_*'))
 receipt_path=t/'FREEZE_RECEIPT.json'; receipt_path.chmod(0o666)
 receipt=json.loads(receipt_path.read_text()); receipt['status']='INVALID_STATUS'; receipt_path.write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n'); assert rejected(t)
print({'status':'PASS','actual_final_receipt_accepted':True,'obsolete_pass_status_not_required':True,'arbitrary_invalid_status_rejected':True,'other_chain_behavior':'unchanged','checkpoint_loaded':False,'inference':False,'optimizer_created':False})
