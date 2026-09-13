"""Build additive SF12 reporting hotfix; never modifies sealed SF12 package."""
import hashlib,json
from pathlib import Path

H=Path(__file__).resolve().parent
SRC=H.parent/'sf12_answer_vocab_retention_v1'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def main():
 assert not any(p.name not in {'BUILD.py','VALIDATE.py','SEAL.py'} for p in H.iterdir())
 text=(SRC/'REPORTER.py').read_text(encoding='utf-8')
 text=text.replace('from pathlib import Path\nimport CONTROLLER as C',"from pathlib import Path\nimport sys\nSOURCE=Path(r'C:\\\\DaveLM-CADAVER\\\\sf12_answer_vocab_retention_v1')\nsys.path.insert(0,str(SOURCE))\nimport CONTROLLER as C")
 text=text.replace("H=Path(__file__).resolve().parent\nread=C.read", "OUT=Path(__file__).resolve().parent\nH=SOURCE\nread=C.read")
 text=text.replace("b='; '.join(f\"{k}:{v['answer']}/{v['both_distinct']}/c{v['collapse']}\" for k,v in r['binding'].items())", "b='; '.join(f\"{k}:{v['answer_exact']}/{v['both_distinct']}/c{v['slot_collapse']}\" for k,v in r['binding'].items())")
 text=text.replace("'binding':ch['binding'],'R_name_endpoint_probe'", "'binding':{k:v['summary'] for k,v in ch['binding'].items()},'R_name_endpoint_probe'")
 text=text.replace("C.rt.atomic_json(result,H/'RESULTS.json'); (H/'FINAL_REPORT.md').write_text(render(result),encoding='utf-8',newline='\\n')", "C.rt.atomic_json(result,OUT/'RESULTS.json'); (OUT/'FINAL_REPORT.md').write_text(render(result),encoding='utf-8',newline='\\n')")
 text=text.replace("f\"- Machine-readable results: `{H/'RESULTS.json'}`\",f\"- Output hashes: `{H/'OUTPUT_SHA256SUMS.txt'}`\"", "f\"- Machine-readable results: `{OUT/'RESULTS.json'}`\",f\"- Output hashes: `{OUT/'OUTPUT_SHA256SUMS.txt'}`\"")
 text=text.replace("for n in ['RESULTS.json','FINAL_REPORT.md','RUN_LEDGER.json']:\n      payload.append(f\"{C.sha(H/n)}  {n}\")\n    (H/'OUTPUT_SHA256SUMS.txt').write_text", "for n in ['RESULTS.json','FINAL_REPORT.md']:\n      payload.append(f\"{C.sha(OUT/n)}  {n}\")\n    payload.append(f\"{C.sha(H/'RUN_LEDGER.json')}  source/RUN_LEDGER.json\")\n    (OUT/'OUTPUT_SHA256SUMS.txt').write_text")
 assert "v['answer']" not in text and "v['summary']" in text
 (H/'REPORTER.py').write_text(text,encoding='utf-8',newline='\n')
 prov={'type':'MECHANICAL_REPORTING_HOTFIX','source_study':str(SRC),'source_receipt_sha256':sha(SRC/'FREEZE_RECEIPT.json'),'source_manifest_sha256':sha(SRC/'SHA256SUMS.txt'),'source_reporter_sha256':sha(SRC/'REPORTER.py'),'defect':'Frozen reporter expected nonexistent short binding summary keys answer/both_distinct/collapse at the gate wrapper level.','correction':'Read the existing authoritative nested binding summary fields answer_exact/both_distinct/slot_collapse. No score, gate, classification rule, checkpoint, or model evaluation changed.','training_replayed':False,'inference_replayed':False}
 (H/'PROVENANCE.json').write_text(json.dumps(prov,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
 print('SF12_REPORTING_HOTFIX_BUILT')
if __name__=='__main__': main()
