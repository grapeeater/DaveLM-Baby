import hashlib,json
from pathlib import Path
r=Path(r'C:\\DaveLM-CADAVER\\human_readiness_hr2_seed87005_v7')
exclude={'MANIFEST.json','SHA256SUMS.txt','FREEZE_RECEIPT.json','FREEZE_RECEIPT.sha256','INDEPENDENT_VALIDATE.py'}
files=sorted(p for p in r.iterdir() if p.is_file() and p.name not in exclude)
man={p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in files}
(r/'MANIFEST.json').write_text(json.dumps(man,indent=2),encoding='utf-8')
lines=''.join(man[n]['sha256']+'  '+n+'\n' for n in sorted(man))
(r/'SHA256SUMS.txt').write_text(lines,encoding='utf-8',newline='')
proto=json.loads((r/'HR2_PROTOCOL.json').read_text())
rec={'status':'HR1_READY_FOR_EXECUTION','version':'HR-1','seed':87004,'parent_sha256':proto['parent_sha256'],'tokenizer_sha256':proto['tokenizer_sha256'],'manifest_sha256':hashlib.sha256((r/'MANIFEST.json').read_bytes()).hexdigest(),'sha256sums_sha256':hashlib.sha256((r/'SHA256SUMS.txt').read_bytes()).hexdigest(),'scientific_scope':'bounded sentence-level language treatment with established T13 rehearsal','final_battery_accessed':False,'checkpoint_loaded':False,'optimizer_created':False,'optimizer_updates':0}
(r/'FREEZE_RECEIPT.json').write_text(json.dumps(rec,indent=2),encoding='utf-8')
(r/'FREEZE_RECEIPT.sha256').write_text(hashlib.sha256((r/'FREEZE_RECEIPT.json').read_bytes()).hexdigest()+'\n',encoding='utf-8')
print(json.dumps(rec,indent=2))

